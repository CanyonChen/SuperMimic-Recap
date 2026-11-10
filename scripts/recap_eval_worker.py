"""Persistent benchmark worker, or budgeted wrapper for the other GPU evaluators.

The allowance limits PyTorch's allocator. Warp, EGL and the CUDA context are
outside that allocator: reserve 4 GiB and monitor this process with nvidia-smi.
This is a sampled budget check, not a hardware memory partition.
"""
from __future__ import annotations

import _thread
import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import threading
import time

GIB = 1024**3
RESERVE_GIB = 4.0


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def process_memory_gib(text, pid):
    # Some drivers report the same PID twice; these are not additive allocations.
    values = []
    for row in csv.reader(text.splitlines()):
        if len(row) == 2 and row[0].strip() == str(pid):
            try:
                values.append(float(row[1]) / 1024)
            except ValueError:
                pass
    return max(values) if values else None


class MemoryBudget:
    def __init__(self, torch, gpu, budget):
        self.torch, self.gpu, self.budget = torch, gpu, budget
        self.peak = None
        self.monitor_error = None
        self.exceeded = None
        self.stop = threading.Event()
        total = torch.cuda.get_device_properties(0).total_memory
        if not math.isfinite(budget) or not RESERVE_GIB < budget <= total / GIB:
            raise ValueError('budget must exceed 4 GiB and fit on this GPU')
        self.allowance = budget - RESERVE_GIB
        torch.cuda.set_per_process_memory_fraction(self.allowance * GIB / total, 0)
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def sample(self):
        try:
            result = subprocess.run(['nvidia-smi', '-i', self.gpu,
                '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=5)
            value = process_memory_gib(result.stdout, os.getpid())
            if value is None:
                self.monitor_error = 'PID not visible (possible container PID namespace); process memory unavailable'
            else:
                self.monitor_error = None
                self.peak = max(self.peak or 0, value)
                if value > self.budget:
                    self.exceeded = f'process uses {value:.2f} GiB, above GPU {self.gpu} budget {self.budget:g} GiB'
        except (OSError, subprocess.SubprocessError) as exc:
            self.monitor_error = str(exc)

    def watch(self):
        while not self.stop.wait(5):
            self.sample()
            if self.exceeded:
                print('[memory budget] ' + self.exceeded, flush=True)
                _thread.interrupt_main()
                return

    def check(self):
        self.sample()
        if self.exceeded:
            raise RuntimeError(self.exceeded)

    def snapshot(self):
        cuda = self.torch.cuda
        return dict(pid=os.getpid(), physical_gpu=self.gpu, budget_gib=self.budget,
            torch_allowance_gib=self.allowance, warp_context_reserve_gib=RESERVE_GIB,
            torch_peak_reserved_gib=cuda.max_memory_reserved()/GIB,
            sampled_process_peak_gib=self.peak, monitor_error=self.monitor_error,
            budget_error=self.exceeded)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
        self.sample()


def serve(args, torch, monitor):
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results

    started = time.perf_counter()
    policy, cfg = load_policy_from_checkpoint(args.ckpt, 'cuda:0')
    monitor.check()
    save(args.ready, dict(status='ready', initialization_s=time.perf_counter()-started, **monitor.snapshot()))
    inventories = {}
    for line in sys.stdin:
        task = json.loads(line)
        started = time.perf_counter()
        runtime = dict(stage=task['stage'], status='failed', num_envs=task['num_envs'])
        try:
            names = Path(task['clip_list']).read_text(encoding='utf-8').splitlines()
            if task['data'] not in inventories:
                # Listing 142k files over NFS for every shard is avoidable.
                paths = discover_npz(Path(task['data']))
                inventories[task['data']] = {p.stem: (i, str(p)) for i, p in enumerate(paths)}
            inventory = inventories[task['data']]
            if len(set(names)) != len(names) or any(n not in inventory for n in names):
                raise ValueError(f'{task["stage"]}: clip list and NPZ files disagree')
            files = [inventory[n][1] for n in sorted(names, key=lambda n: inventory[n][0])]
            print(f'\n[GPU {args.gpu}] {task["stage"]}: {len(files)} clips, {task["num_envs"]} envs', flush=True)
            results = run_benchmark(policy, files, device='cuda:0', num_envs=task['num_envs'],
                future_frames=cfg['env']['future_frames'], anchor_obs=cfg['env'].get('anchor_obs', False))
            # Environment references/cycles must be released between tasks.
            gc.collect()
            torch.cuda.empty_cache()
            monitor.check()
            out = Path(task['output'])
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_name(out.name + '.tmp')
            save_results(results, temporary)
            temporary.replace(out)
            runtime['status'] = 'completed'
        except BaseException as exc:
            runtime['error'] = monitor.exceeded or str(exc) or type(exc).__name__
            raise
        finally:
            runtime.update(elapsed_s=time.perf_counter()-started, **monitor.snapshot())
            save(task['runtime'], runtime)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--memory-gib', type=float, required=True)
    parser.add_argument('--memory-out', type=Path, required=True)
    parser.add_argument('--ckpt', type=Path)
    parser.add_argument('--ready', type=Path)
    parser.add_argument('script', nargs='?')
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.script is None and (args.ckpt is None or args.ready is None):
        parser.error('persistent mode needs --ckpt and --ready')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.gpu.isdigit():
        os.environ['MUJOCO_EGL_DEVICE_ID'] = args.gpu
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'):
        os.environ.pop(key, None)  # These are independent evaluators, not DDP ranks.

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for name in ('SIGTERM', 'SIGHUP'):
        sig = getattr(signal, name, None)
        if sig is not None and signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, interrupted)

    import torch
    monitor = MemoryBudget(torch, args.gpu, args.memory_gib)
    monitor.thread.start()
    print(f'[GPU {args.gpu}] PID {os.getpid()}: budget {args.memory_gib:g} GiB; '
          f'PyTorch {monitor.allowance:g} GiB + {RESERVE_GIB:g} GiB reserve', flush=True)
    try:
        if args.script:
            sys.argv = [args.script, *args.script_args]
            runpy.run_path(args.script, run_name='__main__')
            monitor.check()
        else:
            serve(args, torch, monitor)
    finally:
        monitor.close()
        save(args.memory_out, monitor.snapshot())
    if monitor.exceeded:
        raise RuntimeError(monitor.exceeded)


if __name__ == '__main__':
    main()
