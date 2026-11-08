"""Automate fresh BONES training, checkpoint evaluation, and reports on one shared GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import traceback
import types


def inference_support():
    """Reuse the tested subprocess, stage-state, data-audit, and result helpers."""
    path = Path(__file__).with_name('recap_inference_eval.sh')
    source = path.read_text(encoding='utf-8').split("<<'RECAP_PY'\n", 1)[1].rsplit('\nRECAP_PY', 1)[0]
    module = types.ModuleType('recap_inference_support')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


recap = inference_support()
load, save, digest = recap.load, recap.save, recap.digest
read_csv, write_csv, lines = recap.read_csv, recap.write_csv, recap.lines


def arguments(argv=None):
    p = argparse.ArgumentParser(prog='recap_training_eval.sh',
        description='从零训练 + 全量推理测评；单 GPU、40 GiB 预算。同配置重跑自动续训。')
    p.add_argument('--gpu', default='1', help='物理 GPU 编号或 UUID；只使用一张卡')
    p.add_argument('--vram-gb', type=float, default=40, help='显存预算，单位 GiB；其中 4 GiB 留给 Warp/上下文')
    p.add_argument('--npz-dir', type=Path, default=Path('data/npz/bones_seed'))
    p.add_argument('--manifest', type=Path, help='已有过滤清单；默认复用上次推理测评的清单，找不到则自动过滤')
    p.add_argument('--library', type=Path, default=Path('data/library/bones_seed_train_40g.pt'))
    p.add_argument('--out', type=Path, default=Path('logs/repro_train_40g'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--policy-batch', type=int, default=2048)
    p.add_argument('--wm-batch', type=int, default=2048)
    p.add_argument('--replay-capacity', type=int, default=1000)
    p.add_argument('--iterations', type=int, default=320000)
    p.add_argument('--checkpoint-interval', type=int, default=4000)
    p.add_argument('--smoke-iterations', type=int, default=100)
    p.add_argument('--smoke-only', action='store_true', help='只完成同规模短训；去掉此参数可继续启动正式从零训练')
    p.add_argument('--eval-every', type=int, default=40000, help='训练结束后测评这些间隔的已保存权重，另含第 0 和 100000 步')
    p.add_argument('--eval-num-envs', type=int, default=256)
    p.add_argument('--subset-size', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--shard-size', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--skip-full', action='store_true', help='最终只测子集，报告明确标注')
    p.add_argument('--skip-push', action='store_true')
    p.add_argument('--skip-reference', action='store_true')
    p.add_argument('--render', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args(argv)
    for key in ('workers', 'num_envs', 'policy_batch', 'wm_batch', 'replay_capacity', 'iterations',
                'checkpoint_interval', 'smoke_iterations', 'eval_every', 'eval_num_envs', 'subset_size', 'repeats', 'shard_size'):
        if getattr(a, key) < 1:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if not math.isfinite(a.vram_gb) or a.vram_gb <= 4:
        p.error('--vram-gb must exceed 4 GiB')
    if ',' in a.gpu or not a.gpu:
        p.error('--gpu accepts one GPU only')
    if a.smoke_iterations < 100 or a.iterations < a.smoke_iterations or a.replay_capacity < 42:
        p.error('need smoke-iterations >= 100, iterations >= smoke-iterations, replay-capacity >= 42')
    if a.manifest is None:
        for path in (Path('logs/repro_e8_auto/manifest_filtered.csv'), a.npz_dir / 'manifest_filtered.csv'):
            if path.is_file():
                a.manifest = path
                break
    for key in ('npz_dir', 'manifest', 'library', 'out'):
        if getattr(a, key) is not None:
            setattr(a, key, getattr(a, key).resolve())
    a.skip_prep = True  # The already converted data are audited by the shared evaluator.
    return a


def training_flags(a, stage, target, resume=None):
    flags = ['--run-name', stage, '--log-root', a.out, '--seed', a.seed,
             '--data.library', a.library, '--data.npz-dir', a.npz_dir,
             '--data.manifest', a.out / 'manifest_filtered.csv', '--env.compact-library',
             '--env.num-envs', a.num_envs,
             '--env.replay-capacity', a.replay_capacity, '--policy-trainer.batch-size', a.policy_batch,
             '--wm-trainer.batch-size', a.wm_batch, '--schedule.total-iterations', target,
             '--schedule.checkpoint-interval', min(a.checkpoint_interval, target), '--schedule.log-interval', 50]
    if resume is not None:
        flags += ['--resume', resume]
    return [str(x) for x in flags]


def profile_matches(config, a, stage, target):
    return (str(config['data']['library']) == str(a.library) and config['run_name'] == stage
            and str(config['log_root']) == str(a.out) and config['seed'] == a.seed
            and config['env']['num_envs'] == a.num_envs and config['env']['replay_capacity'] == a.replay_capacity
            and config['policy_trainer']['batch_size'] == a.policy_batch
            and config['wm_trainer']['batch_size'] == a.wm_batch
            and config['schedule']['total_iterations'] == target)


def selected_checkpoints(directory, interval, final_iteration):
    selected = []
    for path in directory.glob('ckpt_[0-9]*.pt'):
        step = int(path.stem.split('_')[-1])
        if step < final_iteration and (step % interval == 0 or step == 100000):
            selected.append((step, path))
    return sorted(selected)


class TrainingEvaluation(recap.Evaluation):
    def __init__(self, a):
        super().__init__(a)
        self.filtered = self.out / 'manifest_filtered.csv'
        self.library_meta = a.library.with_suffix('.json')
        for directory in ('smoke', 'train', 'checkpoint_eval'):
            (self.out / directory).mkdir(exist_ok=True)

    def bind_config(self):
        code = hashlib.sha256()
        paths = sorted((self.root / 'src').rglob('*.py'))
        paths += sorted(p for p in (self.root / 'scripts').rglob('*') if p.suffix in {'.py', '.sh'})
        paths += [self.root / 'uv.lock']
        for path in paths:
            code.update(str(path.relative_to(self.root)).encode())
            code.update(path.read_bytes())
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.a).items()
                  if k not in {'smoke_only', 'render'}}
        config.update(code_sha256=code.hexdigest(), from_scratch=True,
                      manifest_sha256=digest(self.a.manifest) if self.a.manifest else None)
        path = self.out / 'run_config.json'
        if path.exists() and load(path) != config:
            raise ValueError('output belongs to a different training config/code/data manifest; choose a new --out')
        save(path, config)

    def gpu_info(self):
        result = subprocess.run(['nvidia-smi', '-i', self.a.gpu,
            '--query-gpu=uuid,name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=15)
        rows = list(csv.reader(result.stdout.strip().splitlines()))
        if len(rows) != 1 or len(rows[0]) != 5:
            raise RuntimeError('could not query the selected GPU')
        uuid, name, total, free, driver = [v.strip() for v in rows[0]]
        return dict(uuid=uuid, name=name, total_gib=float(total)/1024, free_gib=float(free)/1024, driver=driver)

    def ensure_memory(self):
        gpu = self.gpu_info()
        self.log(f'GPU {self.a.gpu}: {gpu["free_gib"]:.2f} GiB free; requested budget {self.a.vram_gb:g} GiB')
        if gpu['free_gib'] < self.a.vram_gb:
            raise RuntimeError('not enough free VRAM for the requested budget; wait for the shared GPU, then rerun')
        return gpu

    def environment(self):
        gpu = self.ensure_memory()
        save(self.out / 'environment/runtime.json', dict(time=recap.stamp(), python=sys.version,
             platform=platform.platform(), selected_gpu=self.a.gpu, gpu=gpu, vram_budget_gib=self.a.vram_gb))
        for name, command in [('gpu', ['nvidia-smi']), ('packages', ['uv', 'pip', 'freeze']),
                              ('commit', ['git', 'rev-parse', 'HEAD']), ('working_tree', ['git', 'status', '--short'])]:
            result = subprocess.run(command, cwd=self.root, env=self.env, capture_output=True, text=True)
            (self.out / 'environment' / f'{name}.txt').write_text(result.stdout+result.stderr, encoding='utf-8')

    def prepare_data(self):
        self.audit_data()
        if self.a.manifest:
            self.step('filter', self.reuse_manifest, [self.filtered])
        else:
            self.step('filter', lambda: self.python('filter', 'scripts/data/filter_bones_seed.py',
                      '--npz-dir', self.a.npz_dir, '--out', self.filtered, '--workers', self.a.workers), [self.filtered])
        source = {r['name']: r for r in read_csv(self.a.npz_dir / 'manifest.csv')}
        rows = read_csv(self.filtered)
        names = [r['name'] for r in rows]
        if len(names) != len(set(names)) or set(names) != set(source):
            raise ValueError('filtered manifest and converted NPZ coverage differ')
        for row in rows:
            if row['keep'] not in {'0', '1'} or int(row['frames']) != int(source[row['name']]['frames']):
                raise ValueError(f'invalid filtered row: {row["name"]}')
        self.kept = sorted(r['name'] for r in rows if r['keep'] == '1')
        if len(self.kept) < self.a.subset_size:
            raise ValueError('fewer kept clips than --subset-size')
        self.lengths = [int(source[name]['frames']) for name in self.kept]
        identity = dict(**load(self.out / 'dataset_identity.json'), filter_sha256=digest(self.filtered),
                        clips=len(self.kept), frames=sum(self.lengths))
        path = self.out / 'training_data.json'
        if path.exists() and load(path) != identity:
            raise ValueError('training dataset changed; choose a new --out and --library')
        save(path, identity)
        self.identity = identity
        self.log(f'data: {len(rows)} NPZ clips, {len(self.kept)} kept, {sum(self.lengths):,} frames at 50 FPS')
        (self.out / 'lists/train.txt').write_text('\n'.join(self.kept)+'\n', encoding='utf-8')
        subset = sorted(random.Random(self.a.seed).sample(self.kept, self.a.subset_size))
        (self.out / 'lists/subset.txt').write_text('\n'.join(subset)+'\n', encoding='utf-8')

    def inspect_library(self, path):
        import torch
        pack = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        starts, cursor = [], 0
        for length in self.lengths:
            starts.append(cursor)
            cursor += length
        if (pack['names'] != self.kept or pack['lengths'] != self.lengths or pack['starts'] != starts
                or tuple(pack['ref'].shape) != (cursor, 71) or pack['ref'].dtype != torch.float16):
            raise ValueError('packed library does not match the filtered 50 FPS dataset')

    def validate_library(self):
        metadata = load(self.library_meta)
        if metadata['data'] != self.identity:
            raise ValueError('library was built for different data; choose a new --library')
        stat = self.a.library.stat()
        if [stat.st_size, stat.st_mtime_ns] != metadata['stat'] and digest(self.a.library) != metadata['sha256']:
            raise ValueError('packed library changed after creation')
        self.inspect_library(self.a.library)
        save(self.out / 'library.json', dict(path=str(self.a.library), **metadata))

    def build_library(self):
        if self.a.library.exists():
            if not self.library_meta.exists():
                raise ValueError('existing library has no data identity; choose a new --library path')
            self.validate_library()
            return
        temporary = self.a.library.with_suffix('.building.pt')
        self.python('library', 'scripts/data/build_library.py', '--npz-dir', self.a.npz_dir,
                    '--manifest', self.filtered, '--out', temporary, '--workers', self.a.workers)
        self.inspect_library(temporary)
        temporary.replace(self.a.library)
        stat = self.a.library.stat()
        save(self.library_meta, dict(data=self.identity, sha256=digest(self.a.library), stat=[stat.st_size, stat.st_mtime_ns]))

    def checkpoint_info(self, path, stage, target):
        import torch
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        for key in ('wm', 'policy', 'critic', 'wm_opt', 'policy_opt', 'critic_opt', 'bin_return', 'bin_memory'):
            if key not in ck:
                raise ValueError(f'{path}: missing {key}')
        if not profile_matches(ck['config'], self.a, stage, target):
            raise ValueError(f'{path}: training configuration does not match this run')
        iteration, updates = int(ck['iteration']), int(ck['num_updates'])
        if not 0 <= iteration <= target or updates < 0:
            raise ValueError('invalid checkpoint iteration/update count')
        for key in ('wm', 'policy', 'critic'):
            if not ck[key] or not all(torch.isfinite(value).all().item() for value in ck[key].values()):
                raise ValueError(f'{path}: empty or non-finite {key} weights')
        if updates > 0 and any(not ck[key]['state'] for key in ('wm_opt', 'policy_opt', 'critic_opt')):
            raise ValueError('checkpoint has no optimizer state for a trained model')
        return dict(path=str(path), iteration=iteration, num_updates=updates)

    def resume_checkpoint(self, stage, target):
        directory = self.out / stage
        candidates = [directory / 'ckpt_latest.pt'] + sorted(directory.glob('ckpt_[0-9]*.pt'), reverse=True)
        present = [p for p in candidates if p.exists()]
        for path in present:
            try:
                self.checkpoint_info(path, stage, target)
                return path
            except Exception as exc:
                self.log(f'[checkpoint rejected] {path}: {exc}')
        if present:
            raise RuntimeError(f'{stage}: no valid checkpoint to resume; existing artifacts were preserved')
        return None

    def train(self, stage, target):
        self.ensure_memory()
        resume = self.resume_checkpoint(stage, target)
        self.log(f'{stage}: resume {resume}' if resume else f'{stage}: fresh random initialization; no pretrained weights')
        self.python(stage, 'scripts/recap_training_worker.py', '--gpu', self.a.gpu, '--memory-gib', self.a.vram_gb,
                    *training_flags(self.a, stage, target, resume))

    def completed_training(self, stage, target):
        info = self.checkpoint_info(self.out / stage / 'ckpt_latest.pt', stage, target)
        runtime = load(self.out / stage / 'runtime.json')
        if info['iteration'] != target or info['num_updates'] < 1 or runtime['status'] != 'completed':
            raise ValueError(f'{stage}: training did not finish the requested iterations with parameter updates')
        peak = runtime.get('sampled_process_peak_gib')
        if peak is not None and peak > self.a.vram_gb:
            raise ValueError('measured process memory exceeded the configured budget')
        save(self.out / stage / 'checkpoint_metadata.json', info)

    def export_curves(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(str(self.out / 'train/tb'), size_guidance={'scalars': 0}).Reload()
        latest = {}
        for tag in acc.Tags().get('scalars', []):
            for event in acc.Scalars(tag):
                if math.isfinite(event.value):
                    key = (event.step, tag)
                    row = dict(iteration=event.step, tag=tag, value=event.value, wall_time=event.wall_time)
                    if key not in latest or event.wall_time >= latest[key]['wall_time']:
                        latest[key] = row
        rows = [latest[k] for k in sorted(latest)]
        if not rows:
            raise ValueError('training produced no TensorBoard scalar records')
        write_csv(self.out / 'training_curves.csv', rows, ['iteration', 'tag', 'value', 'wall_time'])
        final = {}
        for row in rows:
            final[row['tag']] = dict(iteration=row['iteration'], value=row['value'])
        save(self.out / 'training_metrics.json', final)

    def checkpoint_evaluations(self):
        subset = lines(self.out / 'lists/subset.txt')
        for iteration, ckpt in selected_checkpoints(self.out / 'train', self.a.eval_every, self.a.iterations):
            output = self.out / 'checkpoint_eval' / f'iter_{iteration:07d}.json'
            name = f'checkpoint_eval_{iteration:07d}'
            self.step(name, lambda ckpt=ckpt, output=output, name=name: self.python(name, 'scripts/eval_benchmark.py',
                '--ckpt', ckpt, '--npz-dir', self.a.npz_dir, '--clip-list', self.out / 'lists/subset.txt',
                '--num-envs', self.a.eval_num_envs, '--out', output), [output], lambda output=output: recap.checked_result(output, subset))

    def evaluate_final(self):
        a = self.a
        command = ['bash', self.root / 'scripts/recap_inference_eval.sh', '--gpu', a.gpu,
            '--ckpt', self.out / 'train/ckpt_latest.pt', '--npz-dir', a.npz_dir, '--manifest', self.filtered,
            '--out', self.out / 'evaluation', '--skip-prep', '--workers', a.workers,
            '--num-envs', a.eval_num_envs, '--subset-size', a.subset_size, '--repeats', a.repeats,
            '--shard-size', a.shard_size, '--seed', a.seed, '--render' if a.render else '--no-render']
        for flag in ('skip_full', 'skip_push', 'skip_reference'):
            if getattr(a, flag):
                command.append('--' + flag.replace('_', '-'))
        # Reuse the downloaded reference when available, avoiding another network request.
        reference = self.root / 'logs/repro_e8_auto/reference.json.gz'
        if reference.is_file() and not a.skip_reference:
            command += ['--reference', reference]
        self.command('evaluation', command)

    def validate_evaluation(self):
        directory = self.out / 'evaluation'
        state = load(directory / 'status.json')
        if state['status'] not in {'completed', 'completed_with_warnings'}:
            raise ValueError('final inference evaluation did not complete')
        if lines(directory / 'lists/all.txt') != self.kept:
            raise ValueError('training and evaluation used different motion sets')
        result = directory / ('subset_1.json' if self.a.skip_full else 'full.json')
        expected = lines(self.out / 'lists/subset.txt') if self.a.skip_full else self.kept
        recap.checked_result(result, expected)

    def pipeline(self):
        self.step('environment', self.environment, [self.out / 'environment/runtime.json'])
        self.prepare_data()
        self.step('library', self.build_library, [self.a.library, self.library_meta], self.validate_library)
        for stage, target in [('smoke', self.a.smoke_iterations), ('train', self.a.iterations)]:
            if stage == 'train' and self.a.smoke_only:
                return
            directory = self.out / stage
            self.step(stage, lambda stage=stage, target=target: self.train(stage, target),
                [directory / 'ckpt_latest.pt', directory / 'runtime.json'],
                lambda stage=stage, target=target: self.completed_training(stage, target))
        self.step('curves', self.export_curves, [self.out / 'training_curves.csv', self.out / 'training_metrics.json'])
        self.checkpoint_evaluations()
        inner = self.out / 'evaluation/status.json'
        if inner.exists():
            previous = load(inner)
            needs_video = self.a.render and previous['stages'].get('render', {}).get('status') != 'completed'
            if previous['status'] == 'completed_with_warnings' or needs_video:
                self.state['stages'].pop('evaluation', None)  # Inner stages resume independently.
        self.step('evaluation', self.evaluate_final, [inner, self.out / 'evaluation/report.md'], self.validate_evaluation)

    def report(self):
        a = self.a
        attempts = [load(p) for p in sorted((self.out / 'train').glob('runtime_*.json'))]
        checkpoint = self.out / 'train/checkpoint_metadata.json'
        data = load(self.out / 'training_data.json') if (self.out / 'training_data.json').exists() else {}
        summary = dict(status=self.state['status'], from_scratch=True, requested_iterations=a.iterations,
            checkpoint=load(checkpoint) if checkpoint.exists() else None, training_clips=data.get('clips'),
            training_process_wall_s=sum(r['process_wall_s'] for r in attempts),
            training_loop_s=sum(r['loop_s'] for r in attempts), attempts=len(attempts),
            profile=dict(gpu=a.gpu, vram_budget_gib=a.vram_gb, num_envs=a.num_envs,
                         policy_batch=a.policy_batch, wm_batch=a.wm_batch, replay_capacity=a.replay_capacity))
        peaks = [r['sampled_process_peak_gib'] for r in attempts if r.get('sampled_process_peak_gib') is not None]
        summary['sampled_process_peak_gib'] = max(peaks) if peaks else None
        save(self.out / 'training_summary.json', summary)
        curve = []
        for path in sorted((self.out / 'checkpoint_eval').glob('iter_*.json')):
            step = int(path.stem.split('_')[-1])
            if self.state['stages'].get(f'checkpoint_eval_{step:07d}', {}).get('status') == 'completed':
                curve.append(dict(iteration=step, **recap.stats(load(path)['clips'])))
        final_summary = self.out / 'evaluation/summary.json'
        if self.state['stages'].get('evaluation', {}).get('status') == 'completed' and final_summary.exists():
            metrics = load(final_summary)
            summary['evaluation'] = metrics
            save(self.out / 'training_summary.json', summary)
            for row in metrics:
                if row['run'] == 'subset_1':
                    curve.append(dict(iteration=a.iterations, **{k: v for k, v in row.items() if k not in {'run', 'elapsed_s'}}))
        fields = ['iteration', 'clips', 'failed', 'failure_rate', 'mean_norm_return', 'step_weighted_norm_return']
        write_csv(self.out / 'checkpoint_eval.csv', curve, fields)
        text = ['**SuperMimic 从零训练与推理测评报告**', '', f'状态：`{self.state["status"]}`；生成时间：{recap.stamp()}。',
            f'正式训练目标 {a.iterations:,} 次迭代；先独立试训 {a.smoke_iterations} 次，正式训练不继承试训权重。',
            f'训练动作数：{data.get("clips", "尚未校验")}；数据与评测共用 manifest_filtered.csv。', '',
            f'单卡 GPU {a.gpu}，预算 {a.vram_gb:g} GiB；{a.num_envs} 个环境，策略批 {a.policy_batch}，世界模型批 {a.wm_batch}。',
            '保留 H=32、k=8、UTD=0.4、Critic 和仓库的网络/噪声/随机化设置；完整参数见 train/config.json。',
            '该配置用于共享显存条件下复现第三方训练流程；单卡、策略批量等与原论文/第三方四卡 e8 有差别，不保证相同迭代达到相同指标。', '',
            f'正式训练进程累计耗时 {summary["training_process_wall_s"]/3600:.3f} h，其中训练循环 {summary["training_loop_s"]/3600:.3f} h；尝试次数 {len(attempts)}。',
            f'正式训练进程采样显存峰值：{summary["sampled_process_peak_gib"]} GiB；完整记录见 train/memory.csv 和 runtime_*.json。',
            '此计时排除数据打包、独立试训和推理评测；含重启/失败尝试，不直接等同论文的训练耗时。',
            'PyTorch 分配器额度为预算减 4 GiB；进程显存按日志点采样，容器可能看不到匹配 PID。这不是硬件隔离或连续峰值证明。', '',
            '**训练记录**', '', '- [TensorBoard 标量导出](training_curves.csv)；原始事件位于 train/tb/。',
            '- [最终训练指标](training_metrics.json)：世界模型 loss、在线 8 步误差、训练回报、策略梯度和 Critic loss。',
            '- [检查点子集评测曲线](checkpoint_eval.csv)：先完成训练，再顺序测试已有检查点，避免和训练同时占用显存。',
            '- train/ckpt_latest.pt 是正式训练权重；smoke/ 仅用于流程和显存验证。',
            '- 中断后恢复模型、优化器、采样器状态；回放缓冲区重新预热，不是逐位一致的无缝续训。', '',
            '**训练后测评**', '', '- [完整推理测评报告](evaluation/report.md)',
            '- [失败率与回报](evaluation/summary.csv)；逐动作结果、扰动、第三方对照见 evaluation/。',
            '- [案例清单](evaluation/cases/selected.csv)；MP4 位于 evaluation/cases/videos/。', '',
            '论文参照：25/129,514（单 H100 训练 3 h）、0/129,514（双 H100 训练 9.8 h）。第三方 e8：145/129,491，均值约 0.932。',
            '训练回报和世界模型在线误差来自本次训练日志；基准回报按实际执行步数归一化，与论文完整回合口径存在差异。',
            'BONES 测评衡量训练动作覆盖；这里没有评估未见动作泛化、真机部署或双卡训练效率。', '']
        for name, stage in self.state['stages'].items():
            if stage.get('status') == 'failed':
                text.append(f'- 未完成阶段 {name}：{stage.get("error", "interrupted")}')
        if self.state.get('error'):
            text.append(f'- 错误：{self.state["error"]}')
        if summary.get('evaluation'):
            text += ['', '**本次最终权重的实测结果**', '',
                     '| 测试 | 动作数 | 失败数 | 失败率 | 平均归一化回报 | 步数加权回报 |',
                     '|---|---:|---:|---:|---:|---:|']
            for row in summary['evaluation']:
                text.append(f'| {row["run"]} | {row["clips"]} | {row["failed"]} | '
                            f'{row["failure_rate"]:.4%} | {row["mean_norm_return"]:.6f} | '
                            f'{row["step_weighted_norm_return"]:.6f} |')
        (self.out / 'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')

    def run(self):
        try:
            self.bind_config()
        except Exception:
            self.log_file.close()
            raise
        self.state.update(status='running', updated_at=recap.stamp())
        self.state.pop('error', None)
        save(self.state_path, self.state)
        code = 0
        try:
            self.pipeline()
            inner = self.out / 'evaluation/status.json'
            self.state['status'] = ('smoke_completed' if self.a.smoke_only else
                'completed_with_warnings' if load(inner)['status'] == 'completed_with_warnings' else 'completed')
        except (Exception, KeyboardInterrupt) as exc:
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            self.state.update(status='interrupted' if code == 130 else 'failed', error=str(exc) or 'interrupted')
            if self.current in self.state['stages'] and self.state['stages'][self.current]['status'] == 'running':
                self.state['stages'][self.current].update(status='failed', error=str(exc) or 'interrupted')
            self.log(traceback.format_exc())
        finally:
            self.state['updated_at'] = recap.stamp()
            save(self.state_path, self.state)
            self.report()
            self.log(f'\nStatus: {self.state["status"]}; report: {self.out / "report.md"}')
            self.log_file.close()
        return code


def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for name in ('SIGTERM', 'SIGHUP'):
        sig = getattr(signal, name, None)
        if sig is not None and signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, interrupted)
    return TrainingEvaluation(arguments(argv)).run()


if __name__ == '__main__':
    sys.exit(main())
