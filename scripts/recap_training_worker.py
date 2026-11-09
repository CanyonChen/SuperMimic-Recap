"""One training rank with a PyTorch memory allowance and per-process measurements.

Called by recap_training_eval.sh; remaining CLI arguments are parsed by the real TrainConfig.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

GIB = 1024**3
WARP_RESERVE_GIB = 4.0


def pin_rank(gpus, local_rank_arg=None):
    """Pin MuJoCo Warp and torch to the same GPU before either library is imported."""
    devices = [g.strip() for g in gpus.split(',')]
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', str(world)))
    if (world != len(devices) or local_world != world or rank != local_rank
            or not 0 <= local_rank < world or any(not d for d in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError('GPU list must match this single-node torchrun worker group')
    if local_rank_arg is not None and local_rank_arg != local_rank:
        raise ValueError('--local-rank differs from LOCAL_RANK')
    selected = devices[local_rank]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = selected
    os.environ['SM_GPU_PINNED'] = '1'
    if world > 1:
        os.environ['SM_DIST_BACKEND'] = 'nccl'
        try:
            cores = sorted(os.sched_getaffinity(0))
            per = len(cores) // world
            if per >= 2:
                os.sched_setaffinity(0, set(cores[local_rank*per:(local_rank+1)*per]))
        except (AttributeError, OSError, ValueError):
            pass
    return rank, world, selected


def process_memory_gib(text, pid):
    """A PID can appear twice in nvidia-smi; never sum duplicate context rows."""
    values = []
    for row in csv.reader(text.splitlines()):
        if len(row) == 2 and row[0].strip() == str(pid):
            try:
                values.append(float(row[1].strip()) / 1024)
            except ValueError:
                pass
    return max(values) if values else None


class BudgetWriter:
    def __init__(self, writer, torch, directory, gpu, budget):
        self.writer, self.torch, self.gpu, self.budget = writer, torch, gpu, budget
        self.started = time.perf_counter()
        self.samples = []
        self.monitor_error = None
        path = directory / 'memory.csv'
        needs_header = not path.exists()
        self.file = path.open('a', newline='', encoding='utf-8')
        self.csv = csv.DictWriter(self.file, fieldnames=['pid', 'iteration', 'elapsed_s', 'process_gib',
            'torch_allocated_gib', 'torch_reserved_gib', 'torch_peak_allocated_gib', 'torch_peak_reserved_gib'])
        if needs_header:
            self.csv.writeheader()

    def sample(self, iteration):
        process = None
        try:
            result = subprocess.run(['nvidia-smi', '-i', self.gpu,
                '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=5)
            process = process_memory_gib(result.stdout, os.getpid())
            if process is None:
                self.monitor_error = 'PID not visible in nvidia-smi (possible container PID namespace); process peak unavailable'
        except (OSError, subprocess.SubprocessError) as exc:
            self.monitor_error = str(exc)
        cuda = self.torch.cuda
        row = dict(pid=os.getpid(), iteration=iteration, elapsed_s=time.perf_counter()-self.started,
            process_gib=process, torch_allocated_gib=cuda.memory_allocated()/GIB,
            torch_reserved_gib=cuda.memory_reserved()/GIB,
            torch_peak_allocated_gib=cuda.max_memory_allocated()/GIB,
            torch_peak_reserved_gib=cuda.max_memory_reserved()/GIB)
        self.samples.append(row)
        self.csv.writerow(row)
        self.file.flush()
        if process is not None and process > self.budget:
            raise RuntimeError(f'training process uses {process:.2f} GiB, exceeding {self.budget:g} GiB budget; stopping without changing batches')

    def add_scalar(self, tag, value, step, *args, **kwargs):
        if tag in {'wm/loss', 'wm/grad_norm', 'policy/policy_grad_norm', 'policy/critic_loss'}:
            if not math.isfinite(float(value)):
                raise RuntimeError(f'non-finite training metric at iteration {step}: {tag}={value}')
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step, *args, **kwargs)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.file.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--gpus', '--gpu', dest='gpus', required=True)
    parser.add_argument('--local-rank', '--local_rank', type=int)
    parser.add_argument('--memory-gib', type=float, required=True)
    args, training_args = parser.parse_known_args(argv)
    if not math.isfinite(args.memory_gib) or args.memory_gib <= WARP_RESERVE_GIB:
        parser.error('--memory-gib must exceed the 4 GiB reserve for Warp/context')
    rank, world, gpu = pin_rank(args.gpus, args.local_rank)
    print(f'[rank {rank}/{world}] PID {os.getpid()}, physical GPU {gpu} -> cuda:0; importing training libraries', flush=True)

    # Each rank sees only its own GPU, including MuJoCo Warp's implicit first device.
    import torch
    import tyro
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = tyro.cli(TrainConfig, args=training_args)
    if cfg.device != 'cuda:0':
        raise ValueError('use cuda:0 inside the selected CUDA_VISIBLE_DEVICES mapping')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    total = torch.cuda.get_device_properties(0).total_memory
    torch_allowance = (args.memory_gib - WARP_RESERVE_GIB) * GIB
    if args.memory_gib * GIB > total:
        raise ValueError('memory budget exceeds physical GPU memory')
    torch.cuda.set_per_process_memory_fraction(torch_allowance / total, 0)
    torch.cuda.reset_peak_memory_stats()
    print(f'[rank {rank}] GPU {gpu}: budget {args.memory_gib:g} GiB, PyTorch allowance {torch_allowance/GIB:g} GiB', flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}; restart the parent command to resume its last checkpoint')
    for name in ('SIGTERM', 'SIGHUP'):
        sig = getattr(signal, name, None)
        if sig is not None and signal.getsignal(sig) != signal.SIG_IGN:
            signal.signal(sig, interrupted)

    directory = cfg.log_root / cfg.run_name
    if rank != 0:
        directory = directory / f'rank_{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    attempt = dict(pid=os.getpid(), rank=rank, world_size=world, physical_gpu=gpu,
        started_at=time.time(), budget_gib=args.memory_gib,
        torch_allowance_gib=torch_allowance/GIB, warp_context_reserve_gib=WARP_RESERVE_GIB,
        resume=str(cfg.resume) if cfg.resume else None, status='failed')
    trainer = monitor = None
    loop_start = None
    try:
        trainer = Trainer(cfg)
        attempt['start_iteration'] = trainer.iteration
        monitor = BudgetWriter(trainer.writer, torch, directory, gpu, args.memory_gib)
        if trainer.is_main:
            trainer.writer = monitor
        monitor.sample(trainer.iteration)
        if cfg.resume is None:
            trainer.save('ckpt_0000000.pt')
        attempt['initialization_s'] = time.perf_counter()-started
        loop_start = time.perf_counter()
        trainer.run(on_log=monitor.sample)
        attempt['status'] = 'completed'
    except BaseException as exc:
        attempt['error'] = str(exc) or type(exc).__name__
        raise
    finally:
        attempt['process_wall_s'] = time.perf_counter()-started
        attempt['loop_s'] = time.perf_counter()-loop_start if loop_start is not None else 0
        if trainer is not None:
            attempt.update(iteration=trainer.iteration, num_updates=trainer.num_updates)
        if monitor is not None:
            # Never hide the original training error with a final telemetry error.
            try:
                monitor.sample(trainer.iteration)
            except Exception as exc:
                attempt['final_memory_error'] = str(exc)
                if attempt['status'] == 'completed':
                    attempt['status'] = 'failed'
            values = [s['process_gib'] for s in monitor.samples if s['process_gib'] is not None]
            attempt['sampled_process_peak_gib'] = max(values) if values else None
            attempt['monitor_error'] = monitor.monitor_error
            monitor.close()
        attempt.update(torch_peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                       torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB)
        payload = json.dumps(attempt, ensure_ascii=False, indent=2, allow_nan=False)
        for name in (f'runtime_{int(attempt["started_at"]*1000)}_{os.getpid()}.json', 'runtime.json'):
            path = directory / name
            temporary = path.with_name(path.name + '.tmp')
            temporary.write_text(payload, encoding='utf-8')
            temporary.replace(path)
    if attempt['status'] != 'completed':
        raise RuntimeError(attempt.get('final_memory_error', 'training did not complete'))


if __name__ == '__main__':
    main()
