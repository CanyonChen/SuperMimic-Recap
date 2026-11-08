#!/usr/bin/env bash
# Run from any directory: bash scripts/recap_inference_eval.sh --gpu 1
# The embedded Python only orchestrates the repository's existing evaluators.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'uv is required; use the configured SuperMimic environment.' >&2; exit 1; }
exec env -u PYTHONPATH uv run --locked python -u - "$@" <<'RECAP_PY'
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from datetime import datetime, timezone


REFERENCE_URL = ('https://github.com/CuboiLeo/SuperMimic-Reproduced/releases/'
                 'download/e8-boneseed-100k/e8_boneseed_benchmark_full.json.gz')
DEMO = 'high_jump_R_001__A450'


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def lines(path):
    return Path(path).read_text(encoding='utf-8').splitlines()


def stats(rows):
    if not rows:
        return None
    steps = sum(r['steps'] for r in rows)
    failed = sum(r['failed'] for r in rows)
    return dict(clips=len(rows), failed=failed, failure_rate=failed / len(rows),
                mean_norm_return=sum(r['norm_return'] for r in rows) / len(rows),
                step_weighted_norm_return=sum(r['norm_return'] * r['steps'] for r in rows) / steps)


def checked_result(path, expected):
    result = load(path)
    rows = result['clips']
    names = [r['name'] for r in rows]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError(f'{path}: missing/extra/duplicate clips; expected {len(expected)}, got {len(names)}')
    for r in rows:
        if not isinstance(r['failed'], bool) or r['steps'] < 1 or r['frames'] < 2:
            raise ValueError(f'{path}: invalid clip record: {r["name"]}')
        values = [r['norm_return'], *(r.get('terms') or {}).values()]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'{path}: non-finite metrics: {r["name"]}')
        if r['fail_step'] != (r['steps'] if r['failed'] else -1):
            raise ValueError(f'{path}: inconsistent fail_step: {r["name"]}')
    for key, value in stats(rows).items():
        if key != 'failure_rate' and not math.isclose(result['summary'][key], value, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError(f'{path}: summary mismatch: {key}')
    return rows


def group(name):
    name = name.lower()
    return 'flip' if 'flip' in name else 'side_roll_or_sideway' if ('side_roll' in name or 'sideway' in name) else 'other'


def select_cases(rows):
    good = sorted((r for r in rows if not r['failed']), key=lambda r: (r['norm_return'], r['name']))
    failed = sorted((r for r in rows if r['failed']), key=lambda r: (r['fail_step'], r['name']))
    selections, used = [], set()
    for role, pool in [('low_return_completed', good),
                       ('typical_completed', sorted(good, key=lambda r: abs(r['norm_return'] - good[len(good)//2]['norm_return'])) if good else []),
                       ('early_failure', failed)]:
        chosen = [r for r in pool if r['name'] not in used][:3]
        selections += [dict(r, case_type=role) for r in chosen]
        used.update(r['name'] for r in chosen)
    return selections


def parse_push(text):
    pattern = (r'push ([\d.]+) m/s: fell ([\d.]+)%, settled ([\d.]+)%, '
               r'settle time mean ([\d.nan-]+) s \(max ([\d.nan-]+)\)')
    rows = []
    for speed, fell, settled, mean, maximum in re.findall(pattern, text):
        finite = lambda x: float(x) if math.isfinite(float(x)) else None
        rows.append(dict(speed_m_s=float(speed), failure_rate=float(fell)/100,
                         settled_rate=float(settled)/100, mean_settle_s=finite(mean), max_settle_s=finite(maximum)))
    if [r['speed_m_s'] for r in rows] != [0, .5, 1, 2, 4]:
        raise ValueError('push.log does not contain all five requested speeds')
    return rows


def arguments(argv=None):
    p = argparse.ArgumentParser(prog='recap_inference_eval.sh', description='自动评测已下载的 SuperMimic 权重；不训练。相同配置重跑可续跑。')
    p.add_argument('--gpu', default='1', help='单张物理 GPU 的编号或 UUID（默认 1）')
    p.add_argument('--ckpt', type=Path, default=Path('data/checkpoints/e8/e8_boneseed_ckpt_0100000.pt'))
    p.add_argument('--archive', type=Path, default=Path('data/raw/bones_seed/g1.tar.gz.part'))
    p.add_argument('--raw-dir', type=Path, default=Path('data/raw/bones_seed'))
    p.add_argument('--npz-dir', type=Path, default=Path('data/npz/bones_seed'))
    p.add_argument('--manifest', type=Path, help='复用已有过滤清单（name,frames,keep），保证训练与评测使用同一集合')
    p.add_argument('--out', type=Path, default=Path('logs/repro_e8_auto'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--num-envs', type=int, default=256)
    p.add_argument('--subset-size', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--shard-size', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0, help='仅用于选择固定子集；评测器的环境 seed 固定为 0')
    p.add_argument('--skip-prep', action='store_true', help='使用已经完整转换的 NPZ 和 manifest.csv，不解压/转换')
    p.add_argument('--skip-full', action='store_true', help='只做子集评测；报告不会称为全量复现')
    p.add_argument('--skip-push', action='store_true', help='不做可选的扰动恢复测试')
    p.add_argument('--skip-reference', action='store_true', help='不下载第三方逐动作基准')
    p.add_argument('--reference', type=Path, help='使用本地第三方 benchmark JSON 或 JSON.gz')
    p.add_argument('--render', action=argparse.BooleanOptionalAction, default=True,
                   help='默认生成案例 MP4；用 --no-render 跳过视频渲染')
    p.add_argument('--render-only', action='store_true',
                   help='仅为 --out 中已完成的测评补做视频并更新报告；沿用原评测配置，不重新推理')
    a = p.parse_args(argv)
    if a.render_only and not a.render:
        p.error('--render-only cannot be combined with --no-render')
    for key in ['workers', 'num_envs', 'subset_size', 'repeats', 'shard_size']:
        if getattr(a, key) < 1:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if ',' in a.gpu:
        p.error('--gpu accepts one GPU; this script processes shards sequentially')
    for key in ['ckpt', 'archive', 'raw_dir', 'npz_dir', 'out', 'reference', 'manifest']:
        if getattr(a, key) is not None:
            setattr(a, key, getattr(a, key).resolve())
    return a


class Evaluation:
    def __init__(self, a):
        self.a, self.root, self.out = a, Path.cwd(), a.out
        self.out.mkdir(parents=True, exist_ok=True)
        for sub in ['logs', 'environment', 'lists', 'full', 'cases']:
            (self.out / sub).mkdir(exist_ok=True)
        self.state_path = self.out / 'status.json'
        self.state = load(self.state_path) if self.state_path.exists() else {'stages': {}}
        self.log_file = (self.out / 'run.log').open('a', encoding='utf-8')
        self.current = 'initialization'
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED='1')
        self.env.pop('PYTHONPATH', None)
        self.env.setdefault('OMP_NUM_THREADS', '1')
        os.environ.update({'CUDA_VISIBLE_DEVICES': a.gpu})

    def log(self, message):
        print(message, flush=True)
        self.log_file.write(str(message) + '\n')
        self.log_file.flush()

    def command(self, name, command, extra_env=None):
        command = [str(c) for c in command]
        self.log('$ ' + shlex.join(command))
        with (self.out / 'logs' / f'{name}.log').open('w', encoding='utf-8') as log:
            log.write('$ ' + shlex.join(command) + '\n')
            with subprocess.Popen(command, cwd=self.root, env={**self.env, **(extra_env or {})},
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding='utf-8', errors='replace', start_new_session=(os.name != 'nt')) as proc:
                try:
                    for line in proc.stdout:
                        self.log(line.rstrip('\n'))
                        log.write(line)
                        log.flush()
                    code = proc.wait()
                except BaseException:
                    if os.name == 'nt':
                        proc.terminate()
                    else:
                        os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait()
                    raise
            if code:
                raise RuntimeError(f'{name} exited with code {code}; see logs/{name}.log')

    def python(self, name, script, *args, extra_env=None):
        self.command(name, [sys.executable, '-u', self.root / script, *args], extra_env)

    def step(self, name, action, outputs=(), validate=None, optional=False):
        self.current = name
        previous = self.state['stages'].get(name, {})
        if previous.get('status') == 'completed' and all(Path(p).exists() for p in outputs):
            try:
                if validate:
                    validate()
                self.log(f'[resume] {name}')
                return True
            except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
                self.log(f'[rerun] {name}: {exc}')
        start = time.perf_counter()
        self.state['stages'][name] = {'status': 'running', 'started_at': stamp()}
        save(self.state_path, self.state)
        self.log(f'\n[{stamp()}] {name}')
        try:
            action()
            if not all(Path(p).exists() for p in outputs):
                raise RuntimeError(f'{name}: expected output is missing')
            if validate:
                validate()
        except Exception as exc:
            self.state['stages'][name].update(status='failed', error=str(exc), elapsed_s=time.perf_counter()-start)
            save(self.state_path, self.state)
            if not optional:
                raise
            self.log(f'[optional stage failed] {name}: {exc}')
            return False
        self.state['stages'][name].update(status='completed', elapsed_s=time.perf_counter()-start)
        save(self.state_path, self.state)
        return True

    def bind_config(self):
        a = self.a
        if not a.ckpt.is_file():
            raise FileNotFoundError(f'checkpoint missing: {a.ckpt}')
        code = hashlib.sha256()
        paths = sorted((self.root / 'src/supermimic').rglob('*.py')) + sorted((self.root / 'scripts').rglob('*.py'))
        paths += [self.root / 'uv.lock', self.root / 'scripts/recap_inference_eval.sh']
        for p in paths:
            code.update(str(p.relative_to(self.root)).encode())
            code.update(p.read_bytes())
        cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in {'render', 'render_only'}}
        cfg.update(checkpoint_sha256=digest(a.ckpt), code_sha256=code.hexdigest())
        cfg['reference_sha256'] = digest(a.reference) if a.reference and a.reference.is_file() else None
        cfg['manifest_sha256'] = digest(a.manifest) if a.manifest else None
        cfg['archive_stat'] = [a.archive.stat().st_size, a.archive.stat().st_mtime_ns] if a.archive.exists() and not a.skip_prep else None
        path = self.out / 'run_config.json'
        if path.exists() and load(path) != cfg:
            raise ValueError('output directory belongs to a different checkpoint/config/code; use a new --out directory')
        save(path, cfg)

    def restore_render_config(self):
        if self.state.get('status') not in {'completed', 'completed_with_warnings'}:
            raise ValueError('--render-only requires a completed evaluation; resume unfinished evaluation with its original code/config')
        if self.state['stages'].get('case_rollouts', {}).get('status') != 'completed':
            raise ValueError('--render-only requires completed case_rollouts')
        for path in [self.out / 'cases/selected.csv', self.out / 'cases/rollouts.npz']:
            if not path.is_file():
                raise FileNotFoundError(path)
        cfg = load(self.out / 'run_config.json')
        # Rendering consumes saved trajectories, not the checkpoint or source dataset.
        # Keep the original evaluation identity intact even after a renderer fix.
        for key in vars(self.a):
            if key in cfg and key not in {'gpu', 'out', 'render', 'render_only'}:
                value = cfg[key]
                if key in {'ckpt', 'archive', 'raw_dir', 'npz_dir', 'reference', 'manifest'} and value is not None:
                    value = Path(value)
                setattr(self.a, key, value)

    def environment(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f'CUDA unavailable for --gpu {self.a.gpu}')
        probe = torch.zeros(1, device='cuda:0')
        torch.cuda.synchronize()
        checkpoint = torch.load(self.a.ckpt, map_location='cpu', weights_only=False)
        if 'policy' not in checkpoint or 'config' not in checkpoint:
            raise ValueError('checkpoint has no policy/config')
        # Training saves dataclasses.asdict(), including Path-valued data/log paths.
        checkpoint_config = json.loads(json.dumps(checkpoint['config'], default=str))
        save(self.out / 'environment/checkpoint_config.json', checkpoint_config)
        save(self.out / 'environment/runtime.json', dict(time=stamp(), python=sys.version,
             platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0), visible_gpu=self.a.gpu))
        for name, command in [('gpu', ['nvidia-smi']), ('packages', ['uv', 'pip', 'freeze']),
                              ('commit', ['git', 'rev-parse', 'HEAD']), ('working_tree', ['git', 'status', '--short'])]:
            result = subprocess.run(command, cwd=self.root, env=self.env, capture_output=True, text=True)
            (self.out / 'environment' / f'{name}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
        del probe, checkpoint

    def extract(self):
        a = self.a
        if not a.archive.is_file():
            raise FileNotFoundError(f'archive missing: {a.archive}; for existing NPZ use --skip-prep')
        a.raw_dir.mkdir(parents=True, exist_ok=True)
        extracted, stems = [], set()
        # Stream large archives without keeping all member metadata in memory.
        with tarfile.open(a.archive, 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith('.csv'):
                    continue
                stem = Path(member.name).stem
                if stem in stems:
                    raise ValueError(f'duplicate CSV stem would collide in the converter: {stem}')
                stems.add(stem)
                archive.extract(member, path=a.raw_dir, filter='data')
                extracted.append(str((a.raw_dir / member.name).resolve()))
                if len(extracted) % 5000 == 0:
                    self.log(f'extracted {len(extracted)} CSV files')
        if not extracted:
            raise ValueError('archive contains no CSV files')
        (self.out / 'source_csv.txt').write_text('\n'.join(extracted) + '\n', encoding='utf-8')

    def audit_data(self):
        a = self.a
        rows = read_csv(a.npz_dir / 'manifest.csv')
        by_name = {r['name']: r for r in rows}
        files = sorted(a.npz_dir.glob('*.npz'))
        if not rows or len(rows) != len(by_name) or set(by_name) != {p.stem for p in files}:
            raise ValueError('NPZ files and manifest.csv disagree, or names are duplicated')
        for r in rows:
            if float(r['fps']) != 50 or float(r['input_fps']) != 120 or int(r['frames']) < 2:
                raise ValueError(f'invalid FPS/frames: {r["name"]}; re-convert with input 120 / output 50 FPS')
        if not a.skip_prep:
            expected = [Path(p).stem for p in lines(self.out / 'source_csv.txt')]
            if set(expected) != set(by_name):
                raise ValueError(f'CSV/NPZ coverage mismatch: {len(expected)} source, {len(by_name)} converted; check conversion.log')
        inventory = hashlib.sha256()
        for p in files:
            st = p.stat()
            inventory.update(f'{p.name}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
        identity = dict(manifest_sha256=digest(a.npz_dir / 'manifest.csv'), npz_inventory_sha256=inventory.hexdigest())
        identity_path = self.out / 'dataset_identity.json'
        if identity_path.exists() and load(identity_path) != identity:
            raise ValueError('dataset changed since this run; use a new --out directory')
        save(identity_path, identity)
        save(self.out / 'data_audit.json', dict(npz_clips=len(rows), source_coverage_checked=not a.skip_prep,
             duration_hours=sum(float(r['duration_s']) for r in rows)/3600, input_fps=120, output_fps=50))

    def make_lists(self):
        a = self.a
        names = sorted(r['name'] for r in read_csv(self.out / 'manifest_filtered.csv') if r['keep'] == '1')
        if len(names) < a.subset_size or len(names) != len(set(names)):
            raise ValueError(f'need {a.subset_size} unique kept clips; got {len(names)}')
        def put(name, values):
            (self.out / 'lists' / name).write_text('\n'.join(values) + '\n', encoding='utf-8')
        put('all.txt', names)
        put('subset.txt', sorted(random.Random(a.seed).sample(names, a.subset_size)))
        shards = []
        for i in range(0, len(names), a.shard_size):
            name = f'full_{i//a.shard_size:03d}.txt'
            put(name, names[i:i+a.shard_size])
            shards.append(name)
        save(self.out / 'lists/index.json', dict(kept=len(names), shards=shards))

    def reuse_manifest(self):
        rows = read_csv(self.a.manifest)
        source = {r['name']: r for r in read_csv(self.a.npz_dir / 'manifest.csv')}
        names = [r['name'] for r in rows]
        if not rows or len(names) != len(set(names)) or set(names) != set(source):
            raise ValueError('filter manifest does not cover the NPZ manifest exactly')
        for r in rows:
            if r['keep'] not in {'0', '1'} or int(r['frames']) != int(source[r['name']]['frames']):
                raise ValueError(f'invalid filter row: {r["name"]}')
        destination = self.out / 'manifest_filtered.csv'
        if self.a.manifest.resolve() != destination.resolve():
            shutil.copyfile(self.a.manifest, destination)

    def benchmark(self, stage, names, data, output, num_envs):
        path = self.out / output
        clip_list = self.out / 'lists' / f'{stage}.txt'
        clip_list.write_text('\n'.join(names) + '\n', encoding='utf-8')
        self.step(stage, lambda: self.python(stage, 'scripts/eval_benchmark.py',
                  '--ckpt', self.a.ckpt, '--npz-dir', data, '--clip-list', clip_list,
                  '--num-envs', num_envs, '--out', path), [path], lambda: checked_result(path, names))

    def reference(self, actual):
        path = self.a.reference or self.out / 'reference.json.gz'
        if not path.exists():
            if self.a.reference:
                raise FileNotFoundError(path)
            tmp = path.with_suffix('.part')
            with urllib.request.urlopen(REFERENCE_URL, timeout=60) as response, tmp.open('wb') as f:
                shutil.copyfileobj(response, f)
            tmp.replace(path)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as f:
            published = json.load(f)['clips']
        ref = {r['name']: r for r in published}
        if len(ref) != len(published):
            raise ValueError('reference contains duplicate clip names')
        matched = [r for r in actual if r['name'] in ref]
        if not matched:
            raise ValueError('no clip names match the published benchmark')
        paired = [dict(name=r['name'], actual_return=r['norm_return'], reference_return=ref[r['name']]['norm_return'],
                       return_delta=r['norm_return']-ref[r['name']]['norm_return'], actual_failed=r['failed'],
                       reference_failed=ref[r['name']]['failed'], frames_match=r['frames']==ref[r['name']]['frames']) for r in matched]
        write_csv(self.out / 'reference_comparison.csv', paired, list(paired[0]))
        save(self.out / 'reference_comparison.json', dict(matched=len(matched), evaluated=len(actual),
             source=str(path), sha256=digest(path), actual=stats(matched), reference=stats([ref[r['name']] for r in matched]),
             failure_disagreements=sum(r['actual_failed'] != r['reference_failed'] for r in paired),
             frame_mismatches=sum(not r['frames_match'] for r in paired),
             mean_abs_return_delta=sum(abs(r['return_delta']) for r in paired)/len(paired),
             missing_names=[r['name'] for r in actual if r['name'] not in ref]))

    def pipeline(self):
        a = self.a
        self.step('environment', self.environment, [self.out / 'environment/checkpoint_config.json', self.out / 'environment/runtime.json'])
        demo_dir = self.root / 'data/npz/bones_demo'
        if (demo_dir / f'{DEMO}.npz').exists():
            self.benchmark('single', [DEMO], demo_dir, 'single.json', 1)
        if not a.skip_prep:
            def check_sources():
                paths = lines(self.out / 'source_csv.txt')
                if not paths or not all(Path(p).is_file() for p in paths):
                    raise ValueError('extracted CSV missing')
            self.step('extract', self.extract, [self.out / 'source_csv.txt'], check_sources)
            self.step('conversion', lambda: self.python('conversion', 'scripts/data/csv_to_npz_batch.py',
                      '--input-dir', a.raw_dir, '--output-dir', a.npz_dir, '--input-fps', 120, '--output-fps', 50,
                      '--workers', a.workers), [a.npz_dir / 'manifest.csv'], self.audit_data)
        else:
            self.audit_data()
        if a.manifest:
            self.step('filter', self.reuse_manifest, [self.out / 'manifest_filtered.csv'])
        else:
            self.step('filter', lambda: self.python('filter', 'scripts/data/filter_bones_seed.py', '--npz-dir', a.npz_dir,
                      '--workers', a.workers, '--out', self.out / 'manifest_filtered.csv'), [self.out / 'manifest_filtered.csv'])
        self.step('lists', self.make_lists, [self.out / 'lists/index.json', self.out / 'lists/all.txt', self.out / 'lists/subset.txt'])
        if not (demo_dir / f'{DEMO}.npz').exists():
            self.benchmark('single', [DEMO], a.npz_dir, 'single.json', 1)
        subset = lines(self.out / 'lists/subset.txt')
        for i in range(1, a.repeats+1):
            self.benchmark(f'subset_{i}', subset, a.npz_dir, f'subset_{i}.json', a.num_envs)
        result_path = self.out / 'subset_1.json'
        if not a.skip_full:
            shards = load(self.out / 'lists/index.json')['shards']
            for shard in shards:
                self.benchmark(Path(shard).stem, lines(self.out / 'lists' / shard), a.npz_dir,
                               f'full/{Path(shard).stem}.json', a.num_envs)
            result_path = self.out / 'full.json'
            self.step('merge', lambda: self.python('merge', 'scripts/merge_benchmark_shards.py', '--shards',
                      *[self.out / 'full' / (Path(s).stem + '.json') for s in shards], '--out', result_path),
                      [result_path], lambda: checked_result(result_path, lines(self.out / 'lists/all.txt')))
        actual = checked_result(result_path, subset if a.skip_full else lines(self.out / 'lists/all.txt'))
        selections = select_cases(actual)
        case_names = [r['name'] for r in selections]
        write_csv(self.out / 'cases/selected.csv', selections, ['case_type', 'name', 'norm_return', 'failed', 'steps', 'frames', 'fail_step'])
        self.step('case_drift', lambda: self.python('case_drift', 'scripts/diag_drift.py', '--ckpt', a.ckpt,
                  '--npz-dir', a.npz_dir, '--clips', *case_names, '--out', self.out / 'cases/drift.json'), [self.out / 'cases/drift.json'])
        self.step('case_rollouts', lambda: self.python('case_rollouts', 'scripts/media/record_rollout.py', '--ckpt', a.ckpt,
                  '--npz-dir', a.npz_dir, '--clips', *case_names, '--out', self.out / 'cases/rollouts.npz'), [self.out / 'cases/rollouts.npz'])
        if not a.skip_push:
            self.step('push', lambda: self.python('push', 'scripts/eval_push.py', '--ckpt', a.ckpt, '--speeds', 0, .5, 1, 2, 4,
                      '--num-envs', a.num_envs, '--push-time-s', 1, '--horizon-s', 8), [self.out / 'logs/push.log'],
                      lambda: parse_push((self.out / 'logs/push.log').read_text(encoding='utf-8')), optional=True)
        if not a.skip_reference:
            self.step('reference_full' if not a.skip_full else 'reference_subset', lambda: self.reference(actual),
                      [self.out / 'reference_comparison.json', self.out / 'reference_comparison.csv'], optional=True)
        if a.render:
            self.render_cases(case_names)

    def render_cases(self, case_names):
        def render():
            if not shutil.which('ffmpeg'):
                raise RuntimeError('ffmpeg is missing; trajectories are already saved')
            self.python('render', 'scripts/media/render_rollout.py', '--traj', self.out / 'cases/rollouts.npz',
                        '--out-dir', self.out / 'cases/videos', '--fps', 25,
                        extra_env={'MUJOCO_GL': os.environ.get('MUJOCO_GL', 'egl')})
        self.step('render', render, [self.out / 'cases/videos' / f'{n}.mp4' for n in case_names], optional=True)

    def report(self):
        a, summary = self.a, []
        paths = [('single', self.out/'single.json')] + [(f'subset_{i}', self.out/f'subset_{i}.json') for i in range(1, a.repeats+1)]
        if not a.skip_full:
            paths.append(('full', self.out/'full.json'))
        valid = {}
        for name, path in paths:
            stage = 'merge' if name == 'full' else name
            if self.state['stages'].get(stage, {}).get('status') != 'completed':
                continue
            expected = lines(self.out/'lists/all.txt') if name == 'full' else [DEMO] if name == 'single' else lines(self.out/'lists/subset.txt')
            rows = checked_result(path, expected)
            valid[name] = rows
            elapsed = self.state['stages'][stage]['elapsed_s']
            if name == 'full':
                elapsed += sum(v.get('elapsed_s', 0) for k, v in self.state['stages'].items() if k.startswith('full_') and v['status'] == 'completed')
            summary.append(dict(run=name, **stats(rows), elapsed_s=elapsed))
        fields = ['run', 'clips', 'failed', 'failure_rate', 'mean_norm_return', 'step_weighted_norm_return', 'elapsed_s']
        write_csv(self.out/'summary.csv', summary, fields)
        save(self.out/'summary.json', summary)
        main_name = 'full' if 'full' in valid else 'subset_1' if 'subset_1' in valid else 'single' if 'single' in valid else None
        if main_name:
            rows = [dict(r, group=group(r['name']), fail_time_s=r['fail_step']*.02 if r['failed'] else None) for r in valid[main_name]]
            columns = ['name', 'group', 'frames', 'steps', 'norm_return', 'failed', 'fail_step', 'fail_time_s']
            write_csv(self.out/'per_clip.csv', rows, columns)
            write_csv(self.out/'failures.csv', [r for r in rows if r['failed']], columns)
            grouped = [dict(group=g, **stats([r for r in rows if r['group']==g])) for g in sorted({r['group'] for r in rows})]
            write_csv(self.out/'groups.csv', grouped, ['group', *fields[1:-1]])
        repeat = []
        if 'subset_1' in valid:
            base = {r['name']: r for r in valid['subset_1']}
            for i in range(2, a.repeats+1):
                if f'subset_{i}' in valid:
                    r = valid[f'subset_{i}']
                    diffs = [abs(v['norm_return']-base[v['name']]['norm_return']) for v in r]
                    repeat.append(dict(run=f'subset_{i}', reference='subset_1', mean_abs_return_delta=sum(diffs)/len(diffs),
                         max_abs_return_delta=max(diffs), failure_disagreements=sum(v['failed']!=base[v['name']]['failed'] for v in r)))
        save(self.out/'repeatability.json', repeat)
        push = []
        if not a.skip_push and self.state['stages'].get('push', {}).get('status') == 'completed':
            push = parse_push((self.out/'logs/push.log').read_text(encoding='utf-8'))
            save(self.out/'push.json', push)
            write_csv(self.out/'push.csv', push, list(push[0]))
        text = ['**SuperMimic 预训练策略自动测评报告**', '', f'生成时间：{stamp()}；流程状态：`{self.state["status"]}`。',
                f'主要结果范围：`{main_name or "尚无有效结果"}`。全量表示当前输入目录中过滤后的全部动作；数量需与论文核对。', '',
                '| 运行 | 动作数 | 失败数 | 失败率 | 均值回报 | 步数加权回报 | 耗时 s |',
                '|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            text.append(f'| {r["run"]} | {r["clips"]} | {r["failed"]} | {r["failure_rate"]:.4%} | {r["mean_norm_return"]:.6f} | {r["step_weighted_norm_return"]:.6f} | {r["elapsed_s"]:.1f} |')
        text += ['', '**论文与发布权重的参照**', '',
                 '- 论文表 A.8：25/129,514（1×H100 训练 3 h）；0/129,514（2×H100 训练 9.8 h）。',
                 '- 第三方 e8 发布记录：145/129,491，回报均值 0.932、步数加权 0.925；不是本次实测值。',
                 '- 论文图 3(a) 的回报没有该点的精确表值；本仓库按实际执行步数归一化，提前终止的惩罚口径未对齐。',
                 '- 分组规则：名称含 flip、含 side_roll/sideway、其余；groups.csv 不是论文精确分组。论文单卡翻转 1/44、侧滚 2/22 仅作参照。',
                 '- 计时包含初始化/加载，full 为所有分片及合并耗时之和；不是训练时间、纯推理延迟或浏览器 FPS。',
                 '- BONES 基准衡量训练动作覆盖；本次不验证训练效率、世界模型误差、未见动作泛化或真机迁移。', '',
                 '**重复性**', '', f'同一固定子集的重复运行（不是独立训练种子）：`{json.dumps(repeat, ensure_ascii=False)}`', '']
        refstage = 'reference_subset' if a.skip_full else 'reference_full'
        if not a.skip_reference and self.state['stages'].get(refstage, {}).get('status') == 'completed':
            ref = load(self.out/'reference_comparison.json')
            text += ['**同名动作对照第三方发布结果**', '',
                     f'匹配 {ref["matched"]}/{ref["evaluated"]}；失败判定不同 {ref["failure_disagreements"]} 条；帧数不同 {ref["frame_mismatches"]} 条。',
                     f'回报绝对差均值：{ref["mean_abs_return_delta"]:.6f}；完整数据见 reference_comparison.json/csv。', '']
        text += ['**案例与扰动**', '', 'cases/selected.csv 记录最低回报完成、其余完成动作的回报中位附近、最早失败，各至多 3 条；类别不足时如实减少。',
                 'cases/drift.json 为位置/偏航诊断，cases/rollouts.npz 为实际与参考轨迹；观察性失败原因需查看轨迹后填写，未自动推断。']
        if push:
            text += ['', '| 扰动 m/s | 失败率 | 稳定比例 | 恢复时间均值 s | 最大 s |', '|---|---:|---:|---:|---:|']
            for r in push:
                text.append(f'| {r["speed_m_s"]} | {r["failure_rate"]:.2%} | {r["settled_rate"]:.2%} | {r["mean_settle_s"]} | {r["max_settle_s"]} |')
            text += ['', '0 m/s 为站立基线，不称为恢复时间；无稳定样本的时间为 null。数据来自控制台输出，百分比精度为 0.01%。',
                     '稳定判据：根部速度 <0.1 m/s、位置误差 <0.05 m，持续 0.5 s；与论文 0.5–2.4 s 只作参考对照。']
        else:
            text += ['', '扰动测试：未请求或未成功，见 status.json。']
        render_state = self.state['stages'].get('render', {}).get('status')
        if a.render and render_state == 'completed':
            videos = sorted((self.out / 'cases/videos').glob('*.mp4'))
            text += ['', '**案例视频**', '', '25 FPS；实体为策略执行结果，半透明为参考动作。']
            text += [f'- [{p.name}](cases/videos/{p.name})' for p in videos]
        elif a.render:
            text += ['', '案例视频尚未生成成功；轨迹见 cases/rollouts.npz，原因见 status.json 和 logs/render.log（若已启动渲染）。']
        else:
            text += ['', '案例视频：本次通过 --no-render 跳过。']
        text += ['', '**流程状态**', '', f'请求选项：skip_full={a.skip_full}, skip_push={a.skip_push}, skip_reference={a.skip_reference}, render={a.render}。',
                 '完整阶段状态与耗时见 status.json；命令及错误见 run.log 和 logs/。']
        for name, stage in self.state['stages'].items():
            if stage['status'] == 'failed':
                text.append(f'- {name}: {stage.get("error", "interrupted")}')
        if self.state.get('error'):
            text.append(f'- 流程错误：{self.state["error"]}')
        (self.out/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')

    def run(self):
        # Bind before changing status so a rejected resume cannot rewrite an older report.
        try:
            if self.a.render_only:
                self.restore_render_config()
            else:
                self.bind_config()
        except Exception:
            self.log_file.close()
            raise
        self.state.update(status='running', updated_at=stamp())
        self.state.pop('error', None)
        save(self.state_path, self.state)
        code = 0
        try:
            if self.a.render_only:
                self.render_cases([r['name'] for r in read_csv(self.out / 'cases/selected.csv')])
            else:
                self.pipeline()
            failures = [v for v in self.state['stages'].values() if v['status'] == 'failed']
            self.state['status'] = 'completed_with_warnings' if failures else 'completed'
        except (Exception, KeyboardInterrupt) as exc:
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            self.state.update(status='interrupted' if code == 130 else 'failed', error=str(exc) or 'interrupted')
            stage = self.state['stages'].get(self.current)
            if stage and stage['status'] == 'running':
                stage.update(status='failed', error=str(exc) or 'interrupted')
            self.log(traceback.format_exc())
        finally:
            self.state['updated_at'] = stamp()
            save(self.state_path, self.state)
            self.report()
            self.log(f'\nStatus: {self.state["status"]}; report: {self.out / "report.md"}')
            self.log_file.close()
        return code


def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for name in ['SIGTERM', 'SIGHUP']:
        sig = getattr(signal, name, None)
        if sig is not None and signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, interrupted)
    return Evaluation(arguments(argv)).run()


if __name__ == '__main__':
    sys.exit(main())
RECAP_PY
