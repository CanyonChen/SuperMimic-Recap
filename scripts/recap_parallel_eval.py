"""Parallel, resumable inference with one persistent policy process per GPU."""
from __future__ import annotations

from collections import deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time
import types


def support():
    path = Path(__file__).with_name('recap_inference_eval.sh')
    source = path.read_text(encoding='utf-8').split("<<'RECAP_PY'\n", 1)[1].rsplit('\nRECAP_PY', 1)[0]
    module = types.ModuleType('recap_parallel_support')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


recap = support()
# Published Linux/LF evaluator at f6d52d568dda197c893763fa17654edebf80889d.
# Accept its results only while the underlying evaluators/data/physics code is unchanged.
LEGACY_CODE = '1985346f193d0ab5b2f31a126f4f8ed196477291cddc670412fb69970e8722ee'
LEGACY_CORE = '48f94f5451fd8033c243cb8d7ea99c2b2eaa24ef7fa96e9f3cc82a874fdf7ef7'
GPU_SCRIPTS = {'scripts/eval_benchmark.py', 'scripts/diag_drift.py',
               'scripts/media/record_rollout.py', 'scripts/eval_push.py', 'scripts/media/render_rollout.py'}


def profiles(args):
    devices = args.gpu.split(',')
    def values(raw, default, convert, minimum, option):
        parsed = [convert(s) for s in raw.split(',')] if raw else [default]
        if len(parsed) == 1:
            parsed *= len(devices)
        if len(parsed) != len(devices) or any(not math.isfinite(v) or v < minimum for v in parsed):
            raise ValueError(f'{option}: need one value or one valid value per GPU')
        return parsed
    budgets = values(args.vram_gb, 40, float, 4.001, '--vram-gb')
    envs = values(args.num_envs_per_gpu, args.num_envs, int, 1, '--num-envs-per-gpu')
    return [dict(gpu=g, budget_gib=b, num_envs=n) for g, b, n in zip(devices, budgets, envs)]


def code_digest(root):
    paths = sorted([*root.glob('src/supermimic/**/*.py'),
                    *(p for p in root.glob('scripts/**/*.py') if not p.name.startswith('recap_'))])
    paths.append(root / 'uv.lock')
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def result_document(rows):
    summary = recap.stats(rows)
    summary.pop('failure_rate')
    return dict(summary=summary, clips=rows)


class ParallelEvaluation(recap.Evaluation):
    def __init__(self, args):
        self.profiles = profiles(args)
        if args.reuse_from == args.out:
            raise ValueError('--reuse-from must differ from --out; old results stay unchanged')
        super().__init__(args)
        self.cache = {}
        self.execution_started = time.perf_counter()
        for sub in ('workers', 'memory'):
            (self.out / sub).mkdir(exist_ok=True)

    def evaluation_code_sha256(self):
        return code_digest(self.root)

    def environment(self):
        # Do not create a CUDA context in the scheduler; all GPU work belongs to workers.
        import torch
        checkpoint = torch.load(self.a.ckpt, map_location='cpu', weights_only=False)
        if 'policy' not in checkpoint or 'config' not in checkpoint:
            raise ValueError('checkpoint has no policy/config')
        recap.save(self.out / 'environment/checkpoint_config.json',
                   json.loads(json.dumps(checkpoint['config'], default=str)))
        del checkpoint
        devices = []
        for profile in self.profiles:
            probe = subprocess.run(['nvidia-smi', '-i', profile['gpu'],
                '--query-gpu=name,memory.total,memory.free,uuid', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=10)
            name, total, free, uuid = next(csv.reader(probe.stdout.splitlines()))
            if profile['budget_gib'] > float(total)/1024:
                raise ValueError(f'GPU {profile["gpu"]}: budget exceeds physical VRAM')
            devices.append(dict(profile, name=name.strip(), uuid=uuid.strip(), free_gib=float(free)/1024))
            self.log(f'GPU {profile["gpu"]}: {float(free)/1024:.2f} GiB free; '
                     f'budget {profile["budget_gib"]:g} GiB; {profile["num_envs"]} envs')
        if len({d['uuid'] for d in devices}) != len(devices):
            raise ValueError('GPU identifiers resolve to the same physical device')
        recap.save(self.out / 'environment/runtime.json', dict(time=recap.stamp(), python=sys.version,
            platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda, devices=devices))
        for name, command in [('gpu', ['nvidia-smi']), ('packages', ['uv', 'pip', 'freeze']),
                              ('commit', ['git', 'rev-parse', 'HEAD']), ('working_tree', ['git', 'status', '--short'])]:
            r = subprocess.run(command, cwd=self.root, env=self.env, capture_output=True, text=True, timeout=60)
            (self.out / 'environment' / f'{name}.txt').write_text(r.stdout+r.stderr, encoding='utf-8')

    def python(self, name, script, *args, extra_env=None):
        if script not in GPU_SCRIPTS:
            return super().python(name, script, *args, extra_env=extra_env)
        profile = self.profiles[0]
        self.command(name, [sys.executable, '-u', self.root / 'scripts/recap_eval_worker.py',
            '--gpu', profile['gpu'], '--memory-gib', profile['budget_gib'],
            '--memory-out', self.out / 'memory' / f'{name}.json', self.root / script, *args],
            {**(extra_env or {}), 'CUDA_VISIBLE_DEVICES': profile['gpu']})

    def import_completed(self):
        source = self.a.reuse_from
        if source is None:
            return
        old = recap.load(source / 'run_config.json')
        state = recap.load(source / 'status.json')
        if state.get('status') == 'running':
            raise ValueError('stop the old evaluation and wait for its interrupted/completed status before --reuse-from')
        current = recap.load(self.out / 'run_config.json')
        if old['checkpoint_sha256'] != current['checkpoint_sha256'] or old['npz_dir'] != str(self.a.npz_dir):
            raise ValueError('reuse source has a different checkpoint or dataset path')
        core = self.evaluation_code_sha256()
        if old.get('evaluation_code_sha256') != core and not (old['code_sha256'] == LEGACY_CODE and core == LEGACY_CORE):
            raise ValueError('reuse source evaluator code cannot be verified against the current evaluator')
        completed = [(name, value) for name, value in state['stages'].items()
                     if name.startswith('full_') and value.get('status') == 'completed']
        if completed:
            if recap.load(source / 'dataset_identity.json') != recap.load(self.out / 'dataset_identity.json'):
                raise ValueError('reuse source dataset identity differs')
            if recap.digest(source / 'manifest_filtered.csv') != recap.digest(self.out / 'manifest_filtered.csv'):
                raise ValueError('reuse source filtered manifest differs')
        sources = []
        for name, stage in completed:
            path = source / 'full' / f'{name}.json'
            rows = recap.checked_result(path, recap.lines(source / 'lists' / f'{name}.txt'))
            for row in rows:
                if row['name'] in self.cache:
                    raise ValueError(f'duplicate cached motion: {row["name"]}')
                self.cache[row['name']] = row
            sources.append(dict(stage=name, clips=len(rows), sha256=recap.digest(path),
                                gpu=stage.get('gpu', old['gpu']), num_envs=stage.get('num_envs', old['num_envs'])))
        allowed = set(recap.lines(self.out / 'lists/all.txt'))
        if not set(self.cache) <= allowed:
            raise ValueError('reuse source includes motions outside the current evaluation set')
        recap.save(self.out / 'reuse.json', dict(source=str(source), checkpoint_sha256=current['checkpoint_sha256'],
            source_config_sha256=recap.digest(source / 'run_config.json'), reused_clips=len(self.cache), shards=sources))
        self.log(f'[reuse] {len(self.cache):,} completed full-set motions from {source}')

    def reference(self, actual):
        cached = self.root/'logs/repro_e8_auto/reference.json.gz'
        destination = self.out/'reference.json.gz'
        if self.a.reference is None and not destination.exists() and cached.is_file():
            recap.shutil.copyfile(cached, destination)
        return super().reference(actual)

    def pipeline(self):
        a = self.a
        self.step('environment', self.environment, [self.out/'environment/checkpoint_config.json', self.out/'environment/runtime.json'])
        if not a.skip_prep:
            self.step('extract', self.extract, [self.out/'source_csv.txt'])
            self.step('conversion', lambda: self.python('conversion', 'scripts/data/csv_to_npz_batch.py',
                '--input-dir', a.raw_dir, '--output-dir', a.npz_dir, '--input-fps', 120, '--output-fps', 50,
                '--workers', a.workers), [a.npz_dir/'manifest.csv'], self.audit_data)
        else:
            self.audit_data()
        if a.manifest:
            self.step('filter', self.reuse_manifest, [self.out/'manifest_filtered.csv'])
        else:
            self.step('filter', lambda: self.python('filter', 'scripts/data/filter_bones_seed.py', '--npz-dir', a.npz_dir,
                '--workers', a.workers, '--out', self.out/'manifest_filtered.csv'), [self.out/'manifest_filtered.csv'])
        self.step('lists', self.make_lists, [self.out/'lists/index.json', self.out/'lists/all.txt', self.out/'lists/subset.txt'])
        self.import_completed()
        demo = self.root/'data/npz/bones_demo'
        demo = demo if (demo/f'{recap.DEMO}.npz').exists() else a.npz_dir
        jobs = [dict(stage='single', names=[recap.DEMO], data=demo, output=self.out/'single.json', num_envs=1)]
        subset = recap.lines(self.out/'lists/subset.txt')
        jobs += [dict(stage=f'subset_{i}', names=subset, data=a.npz_dir, output=self.out/f'subset_{i}.json')
                 for i in range(1, a.repeats+1)]
        shards = [] if a.skip_full else recap.load(self.out/'lists/index.json')['shards']
        frames = {r['name']: int(r['frames']) for r in recap.read_csv(self.out/'manifest_filtered.csv')}
        full_jobs = [dict(stage=Path(s).stem, names=recap.lines(self.out/'lists'/s), data=a.npz_dir,
                         output=self.out/'full'/f'{Path(s).stem}.json') for s in shards]
        # Longest remaining work first reduces the idle tail; merge keeps canonical order.
        full_jobs.sort(key=lambda j: -sum(frames[n] for n in j['names'] if n not in self.cache))
        jobs += full_jobs
        def validate_jobs():
            for job in jobs:
                recap.checked_result(job['output'], job['names'])
        self.step('benchmarks', lambda: self.run_workers(jobs), [j['output'] for j in jobs], validate_jobs)
        result_path = self.out/'subset_1.json'
        if not a.skip_full:
            result_path = self.out/'full.json'
            self.step('merge', lambda: self.python('merge', 'scripts/merge_benchmark_shards.py', '--shards',
                *[self.out/'full'/f'{Path(s).stem}.json' for s in shards], '--out', result_path),
                [result_path], lambda: recap.checked_result(result_path, recap.lines(self.out/'lists/all.txt')))
        actual = recap.checked_result(result_path, subset if a.skip_full else recap.lines(self.out/'lists/all.txt'))
        selections = recap.select_cases(actual)
        names = [r['name'] for r in selections]
        recap.write_csv(self.out/'cases/selected.csv', selections, ['case_type', 'name', 'norm_return', 'failed', 'steps', 'frames', 'fail_step'])
        self.step('case_drift', lambda: self.python('case_drift', 'scripts/diag_drift.py', '--ckpt', a.ckpt,
            '--npz-dir', a.npz_dir, '--clips', *names, '--out', self.out/'cases/drift.json'), [self.out/'cases/drift.json'])
        self.step('case_rollouts', lambda: self.python('case_rollouts', 'scripts/media/record_rollout.py', '--ckpt', a.ckpt,
            '--npz-dir', a.npz_dir, '--clips', *names, '--out', self.out/'cases/rollouts.npz'), [self.out/'cases/rollouts.npz'])
        if not a.skip_push:
            self.step('push', lambda: self.python('push', 'scripts/eval_push.py', '--ckpt', a.ckpt, '--speeds', 0, .5, 1, 2, 4,
                '--num-envs', self.profiles[0]['num_envs'], '--push-time-s', 1, '--horizon-s', 8), [self.out/'logs/push.log'],
                lambda: recap.parse_push((self.out/'logs/push.log').read_text(encoding='utf-8')), optional=True)
        if not a.skip_reference:
            self.step('reference_subset' if a.skip_full else 'reference_full', lambda: self.reference(actual),
                [self.out/'reference_comparison.json', self.out/'reference_comparison.csv'], optional=True)
        if a.render:
            self.render_cases(names)

    def launch_worker(self, index, profile, attempt):
        log_path = self.out/'logs'/f'worker_{index}.log'
        log = log_path.open('a', encoding='utf-8')
        ready = attempt/f'worker_{index}_ready.json'
        command = [str(x) for x in [sys.executable, '-u', self.root/'scripts/recap_eval_worker.py',
            '--gpu', profile['gpu'], '--memory-gib', profile['budget_gib'], '--memory-out',
            self.out/'memory'/f'worker_{index}.json', '--ckpt', self.a.ckpt, '--ready', ready]]
        self.log('$ ' + shlex.join(command))
        try:
            proc = subprocess.Popen(command, cwd=self.root, env={**self.env, 'CUDA_VISIBLE_DEVICES': profile['gpu']},
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                start_new_session=(os.name != 'nt'))
        except BaseException:
            log.close()
            raise
        self.log(f'[worker] GPU {profile["gpu"]}: PID {proc.pid}; {log_path}')
        return dict(profile=profile, proc=proc, log=log, task=None, ready=ready)

    def finish_task(self, slot):
        job = slot['task']
        runtime = recap.load(job['runtime'])
        if runtime['status'] != 'completed':
            raise RuntimeError(f'{job["stage"]}: {runtime.get("error", "worker failed")}')
        rows = recap.checked_result(job['fresh'], job['remaining'])
        merged = {r['name']: r for r in [*job['cached'], *rows]}
        recap.save(job['output'], result_document([merged[n] for n in job['names']]))
        recap.checked_result(job['output'], job['names'])
        stage = self.state['stages'][job['stage']]
        stage.update(status='completed', elapsed_s=runtime['elapsed_s'], memory=runtime,
                     evaluated_clips=len(rows), reused_clips=len(job['cached']))
        recap.save(self.state_path, self.state)
        self.log(f'[done] GPU {slot["profile"]["gpu"]}: {job["stage"]}; '
                 f'{len(rows):,} evaluated + {len(job["cached"]):,} reused; {runtime["elapsed_s"]:.1f}s')
        slot['task'] = None

    def stop_workers(self, slots, failed):
        for slot in slots:
            proc = slot['proc']
            if proc.poll() is None:
                try:
                    if failed:
                        if os.name == 'nt':
                            proc.terminate()
                        else:
                            os.killpg(proc.pid, signal.SIGTERM)
                    else:
                        proc.stdin.close()
                except (ProcessLookupError, BrokenPipeError):
                    pass
        for slot in slots:
            proc = slot['proc']
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            slot['log'].close()

    def run_workers(self, jobs):
        pending = deque()
        for job in jobs:
            previous = self.state['stages'].get(job['stage'], {})
            if previous.get('status') == 'completed':
                try:
                    recap.checked_result(job['output'], job['names'])
                    self.log('[resume] ' + job['stage'])
                    continue
                except (OSError, ValueError, KeyError):
                    pass
            job['cached'] = [self.cache[n] for n in job['names'] if job['stage'].startswith('full_') and n in self.cache]
            known = {r['name'] for r in job['cached']}
            job['remaining'] = [n for n in job['names'] if n not in known]
            if not job['remaining']:
                recap.save(job['output'], result_document(job['cached']))
                self.state['stages'][job['stage']] = dict(status='completed', elapsed_s=0,
                    reused_clips=len(job['cached']), evaluated_clips=0, source=str(self.a.reuse_from))
            else:
                pending.append(job)
        recap.save(self.state_path, self.state)
        if not pending:
            self.state.setdefault('full_parallel_wall_s', 0.0)
            return
        attempt = self.out/'workers'/f'{time.time_ns()}_{os.getpid()}'
        attempt.mkdir()
        slots, failed = [], True
        full_start = full_end = None
        last_heartbeat = time.monotonic()
        try:
            for index, profile in enumerate(self.profiles[:len(pending)]):
                slots.append(self.launch_worker(index, profile, attempt))
            while pending or any(s['task'] for s in slots):
                for slot in slots:
                    if slot['task'] and slot['task']['runtime'].exists():
                        full = slot['task']['stage'].startswith('full_')
                        self.finish_task(slot)
                        if full:
                            full_end = time.perf_counter()
                    if slot['proc'].poll() is not None:
                        raise RuntimeError(f'GPU {slot["profile"]["gpu"]} worker exited; see logs/worker_*.log')
                    if slot['task'] is None and pending:
                        job = pending.popleft()
                        profile = slot['profile']
                        job['runtime'] = attempt/f'{job["stage"]}.json'
                        job['fresh'] = attempt/f'{job["stage"]}.results.json'
                        clip_list = attempt/f'{job["stage"]}.txt'
                        clip_list.write_text('\n'.join(job['remaining'])+'\n', encoding='utf-8')
                        num_envs = min(job.get('num_envs', profile['num_envs']), len(job['remaining']))
                        self.state['stages'][job['stage']] = dict(status='running', started_at=recap.stamp(),
                            gpu=profile['gpu'], budget_gib=profile['budget_gib'], num_envs=num_envs,
                            reused_clips=len(job['cached']))
                        recap.save(self.state_path, self.state)
                        slot['task'] = job
                        task = dict(stage=job['stage'], clip_list=str(clip_list), data=str(job['data']),
                            output=str(job['fresh']), runtime=str(job['runtime']), num_envs=num_envs)
                        slot['proc'].stdin.write(json.dumps(task)+'\n')
                        slot['proc'].stdin.flush()
                        if job['stage'].startswith('full_') and full_start is None:
                            full_start = time.perf_counter()
                        self.log(f'[dispatch] GPU {profile["gpu"]}: {job["stage"]}; {len(job["remaining"]):,} clips, {num_envs} envs')
                if time.monotonic()-last_heartbeat >= 30:
                    self.log('[progress] ' + '; '.join(f'GPU {s["profile"]["gpu"]}: {s["task"]["stage"] if s["task"] else "idle"}' for s in slots)
                             + f'; {len(pending)} tasks waiting')
                    last_heartbeat = time.monotonic()
                time.sleep(.1)
            failed = False
        finally:
            self.stop_workers(slots, failed)
            if failed:
                for slot in slots:
                    if slot['task']:
                        self.state['stages'][slot['task']['stage']].update(status='failed', error='worker group interrupted; rerun to retry')
            if full_start is not None:
                end = time.perf_counter() if failed else (full_end or time.perf_counter())
                self.state['full_parallel_wall_s'] = self.state.get('full_parallel_wall_s', 0) + end-full_start
            else:
                self.state.setdefault('full_parallel_wall_s', 0.0)
            recap.save(self.state_path, self.state)

    def report(self):
        super().report()
        execution = dict(profiles=self.profiles, shard_size=self.a.shard_size,
            elapsed_this_launch_s=time.perf_counter()-self.execution_started,
            full_parallel_wall_s=self.state.get('full_parallel_wall_s'),
            full_worker_elapsed_s=sum(s.get('elapsed_s', 0) for n, s in self.state['stages'].items()
                                      if n.startswith('full_') and s['status'] == 'completed'),
            reused_clips=sum(s.get('reused_clips', 0) for n, s in self.state['stages'].items()
                             if n.startswith('full_') and s['status'] == 'completed'))
        recap.save(self.out/'execution.json', execution)
        with (self.out/'report.md').open('a', encoding='utf-8') as f:
            f.write('\n**并行执行与计时**\n\n')
            f.write(f'- GPU 配置：`{json.dumps(self.profiles)}`；每片最多 {self.a.shard_size:,} 条动作。\n')
            f.write(f'- 全量复用了 {execution["reused_clips"]:,} 条旧结果；来源见 reuse.json，逐片实际 GPU/环境数见 status.json。\n')
            f.write('- full 耗时是本目录历次全量任务派发到结束的墙钟时间加合并时间；导入结果的历史耗时和前序子集/模型启动时间不计入。全过程本次耗时另见 execution.json；不能把复用后的时间当作从零全量速度。\n')
            f.write('- 重复测试可能使用不同环境批量/GPU；repeatability.json 同时反映数值执行配置的差异。\n')
            f.write('- 预算限制 PyTorch 分配，另留 4 GiB 给 Warp/上下文，并每 5 秒采样进程总显存；不是 MIG 硬隔离。容器 PID 不可见时会在 memory/ 中注明监测不可用。\n')

    def run(self):
        lock = (self.out/'.parallel.lock').open('a')
        acquired = False
        try:
            if os.name != 'nt':
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            (self.out/'launcher.pid').write_text(str(os.getpid()), encoding='ascii')
            return super().run()
        finally:
            if acquired:
                (self.out/'launcher.pid').unlink(missing_ok=True)
            lock.close()


if __name__ == '__main__':
    raise SystemExit(recap.main())
