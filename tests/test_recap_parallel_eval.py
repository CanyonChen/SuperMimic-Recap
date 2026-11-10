"""CPU tests of real persistent subprocess scheduling, reuse and memory accounting."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import test_recap_inference_eval as original

FakeGPU = original.FakeGPU

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


parallel = module(ROOT/'scripts/recap_parallel_eval.py', 'parallel_fixture')
worker = module(ROOT/'scripts/recap_eval_worker.py', 'eval_worker_fixture')
recap = parallel.recap

FAKE_WORKER = r'''
import argparse, json, os
from pathlib import Path
import sys, time
p = argparse.ArgumentParser()
p.add_argument('--gpu'); p.add_argument('--memory-gib'); p.add_argument('--memory-out')
p.add_argument('--ckpt'); p.add_argument('--ready')
a = p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] == a.gpu
Path(a.ready).write_text('{}')
for line in sys.stdin:
    j = json.loads(line)
    started = time.time()
    names = Path(j['clip_list']).read_text().splitlines()
    time.sleep(.04 if a.gpu == '1' else .12)
    if j['stage'] == os.environ.get('FAIL_STAGE'):
        Path(j['runtime']).write_text(json.dumps(dict(status='failed', error='injected worker failure')))
        sys.exit(7)
    rows = [dict(name=n, frames=100, steps=10 if 'bad' in n else 99,
            norm_return=.3 if 'bad' in n else .9, failed='bad' in n,
            fail_step=10 if 'bad' in n else -1, terms={'body_pos': .01}) for n in names]
    steps = sum(r['steps'] for r in rows)
    summary = dict(clips=len(rows), failed=sum(r['failed'] for r in rows),
        mean_norm_return=sum(r['norm_return'] for r in rows)/len(rows),
        step_weighted_norm_return=sum(r['norm_return']*r['steps'] for r in rows)/steps)
    Path(j['output']).write_text(json.dumps(dict(summary=summary, clips=rows)))
    runtime = dict(status='completed', elapsed_s=time.time()-started, pid=os.getpid(),
        physical_gpu=a.gpu, budget_gib=float(a.memory_gib), num_envs=j['num_envs'],
        start=started, end=time.time())
    temp = Path(j['runtime']+'.tmp')
    temp.write_text(json.dumps(runtime)); temp.replace(j['runtime'])
Path(a.memory_out).write_text('{}')
'''


class FakeParallel(parallel.ParallelEvaluation):
    environment = FakeGPU.environment
    python = FakeGPU.python

    def __init__(self, args, fail_at=None):
        super().__init__(args)
        self.calls, self.fail_at, self.launches = [], fail_at, []

    def launch_worker(self, index, profile, attempt):
        slot = super().launch_worker(index, profile, attempt)
        self.launches.append(slot)
        return slot


class ParallelTests(unittest.TestCase):
    setUp = original.RecapTests.setUp
    tearDown = original.RecapTests.tearDown
    run_quietly = original.RecapTests.run_quietly

    def args(self, *extra):
        return original.RecapTests.args(self, '--gpus', '0,1', '--vram-gb', '40,70',
                               '--num-envs-per-gpu', '1,2', *extra)

    def prepare(self):
        (self.root/'scripts/recap_eval_worker.py').write_text(FAKE_WORKER, encoding='utf-8')

    def test_profiles_reject_mismatched_budgets_and_duplicates(self):
        self.assertEqual(parallel.profiles(self.args()), [dict(gpu='0', budget_gib=40, num_envs=1),
                                                        dict(gpu='1', budget_gib=70, num_envs=2)])
        for extra in [('--vram-gb', '40,70,40'), ('--vram-gb', 'nan'),
                      ('--vram-gb', '4'), ('--num-envs-per-gpu', '0,2')]:
            with self.assertRaises(ValueError):
                parallel.profiles(self.args(*extra))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--gpus', '0,0')

    def test_real_workers_overlap_persist_and_resume(self):
        self.prepare()
        runner = FakeParallel(self.args())
        self.assertEqual(self.run_quietly(runner), 0)
        self.assertEqual(len(runner.launches), 2)
        stages = recap.load(runner.state_path)['stages']
        tasks = [s for s in stages.values() if 'memory' in s]
        self.assertEqual(len({s['memory']['pid'] for s in tasks}), 2)
        self.assertTrue(any(a['memory']['start'] < b['memory']['end'] and b['memory']['start'] < a['memory']['end']
            for a in tasks for b in tasks if a['gpu'] != b['gpu']))
        self.assertEqual({s['budget_gib'] for s in tasks}, {40, 70})
        self.assertEqual(recap.load(runner.out/'full.json')['summary']['clips'], len(self.names))
        self.assertEqual(recap.load(runner.out/'full.json')['summary']['failed'], 1)
        self.assertFalse((runner.out/'launcher.pid').exists())
        again = FakeParallel(self.args())
        self.assertEqual(self.run_quietly(again), 0)
        self.assertEqual(again.launches, [])
        self.assertEqual(again.calls, [])

    def test_failure_stops_both_workers_and_preserves_completed_shards(self):
        self.prepare()
        runner = FakeParallel(self.args())
        runner.env['FAIL_STAGE'] = 'full_001'
        self.assertEqual(self.run_quietly(runner), 1)
        self.assertTrue(all(s['proc'].poll() is not None for s in runner.launches))
        completed = {n: (runner.out/'full'/f'{n}.json').read_bytes() for n, s in runner.state['stages'].items()
                     if n.startswith('full_') and s['status'] == 'completed'}
        self.assertFalse((runner.out/'full.json').exists())
        again = FakeParallel(self.args())
        self.assertEqual(self.run_quietly(again), 0)
        for n, before in completed.items():
            self.assertEqual((again.out/'full'/f'{n}.json').read_bytes(), before)
        self.assertFalse(any(s['status'] == 'running' for s in again.state['stages'].values()))

    def test_reuse_legacy_shards_with_different_boundaries(self):
        self.prepare()
        old = FakeGPU(original.RecapTests.args(self), fail_at='full_002')
        self.assertEqual(self.run_quietly(old), 1)
        old_cfg = recap.load(old.out/'run_config.json')
        old_cfg['code_sha256'] = parallel.LEGACY_CODE
        recap.save(old.out/'run_config.json', old_cfg)
        before = {p: p.read_bytes() for p in old.out.rglob('*') if p.is_file()}
        runner = FakeParallel(self.args('--reuse-from', 'out', '--out', 'new', '--shard-size', '3'))
        with patch.object(runner, 'evaluation_code_sha256', return_value=parallel.LEGACY_CORE):
            self.assertEqual(self.run_quietly(runner), 0)
        actual = recap.load(runner.out/'full.json')
        self.assertEqual({r['name'] for r in actual['clips']}, set(self.names))
        self.assertEqual(actual['summary']['failed'], 1)
        self.assertEqual(recap.load(runner.out/'reuse.json')['reused_clips'], 4)
        self.assertEqual(recap.load(runner.out/'execution.json')['reused_clips'], 4)
        evaluated = sum(s.get('evaluated_clips', 0) for n, s in runner.state['stages'].items() if n.startswith('full_'))
        self.assertEqual(evaluated, 2)
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_reuse_rejects_unknown_core_and_running_source(self):
        self.prepare()
        old = FakeGPU(original.RecapTests.args(self))
        self.assertEqual(self.run_quietly(old), 0)
        cfg = recap.load(old.out/'run_config.json')
        cfg['code_sha256'] = 'unknown'
        recap.save(old.out/'run_config.json', cfg)
        runner = FakeParallel(self.args('--reuse-from', 'out', '--out', 'new'))
        self.assertEqual(self.run_quietly(runner), 1)
        self.assertIn('evaluator code', runner.state['error'])
        self.assertEqual(runner.launches, [])
        state = recap.load(old.state_path)
        state['status'] = 'running'
        recap.save(old.state_path, state)
        again = FakeParallel(self.args('--reuse-from', 'out', '--out', 'new2'))
        self.assertEqual(self.run_quietly(again), 1)
        self.assertIn('stop the old', again.state['error'])

    def test_budget_fraction_and_process_memory(self):
        calls = []
        cuda = types.SimpleNamespace(get_device_properties=lambda _: types.SimpleNamespace(total_memory=80*worker.GIB),
            set_per_process_memory_fraction=lambda f, d: calls.append((f, d)))
        budget = worker.MemoryBudget(types.SimpleNamespace(cuda=cuda), '0', 40)
        self.assertEqual(calls, [(36/80, 0)])
        text = f'{os.getpid()}, 45056\n{os.getpid()}, 45056\n999999, 70000\n'
        self.assertEqual(worker.process_memory_gib(text, os.getpid()), 44)
        with patch.object(worker.subprocess, 'run', return_value=types.SimpleNamespace(stdout=text)):
            with self.assertRaisesRegex(RuntimeError, 'above GPU 0 budget'):
                budget.check()

    def test_persistent_worker_loads_policy_and_lists_directory_once(self):
        from unittest.mock import Mock
        benchmark = types.ModuleType('supermimic.eval.benchmark')
        benchmark.load_policy_from_checkpoint = Mock(return_value=(object(), {'env': {'future_frames': 10}}))
        benchmark.run_benchmark = Mock(return_value=[object()])
        benchmark.save_results = lambda rows, path: Path(path).write_text('{}')
        library = types.ModuleType('supermimic.envs.mdp.motion_library')
        library.discover_npz = Mock(return_value=[self.root/'npz/a.npz', self.root/'npz/b.npz'])
        tasks = []
        for i in range(2):
            clips = self.root/f'clips{i}.txt'
            clips.write_text('b\na\n')
            tasks.append(dict(stage=f'job{i}', num_envs=i+1, clip_list=str(clips), data=str(self.root/'npz'),
                output=str(self.root/f'result{i}.json'), runtime=str(self.root/f'runtime{i}.json')))
        monitor = types.SimpleNamespace(check=lambda: None, snapshot=lambda: {})
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
        args = types.SimpleNamespace(ckpt='model.pt', ready=self.root/'ready.json', gpu='1')
        import json
        with patch.dict(sys.modules, {'supermimic.eval.benchmark': benchmark, 'supermimic.envs.mdp.motion_library': library}), \
             patch.object(sys, 'stdin', io.StringIO('\n'.join(json.dumps(j) for j in tasks))), \
             contextlib.redirect_stdout(io.StringIO()):
            worker.serve(args, torch, monitor)
        self.assertEqual(benchmark.load_policy_from_checkpoint.call_count, 1)
        self.assertEqual(library.discover_npz.call_count, 1)
        self.assertEqual(benchmark.run_benchmark.call_count, 2)
        for call in benchmark.run_benchmark.call_args_list:
            self.assertEqual(call.args[1], [str(self.root/'npz/a.npz'), str(self.root/'npz/b.npz')])
        self.assertEqual(recap.load(self.root/'runtime1.json')['status'], 'completed')

    def test_budgeted_wrapper_pins_gpu_and_preserves_script_flags(self):
        from unittest.mock import Mock
        cuda = types.SimpleNamespace(get_device_properties=lambda _: types.SimpleNamespace(total_memory=80*worker.GIB),
            set_per_process_memory_fraction=Mock(), max_memory_reserved=lambda: 0)
        fake_torch = types.SimpleNamespace(cuda=cuda)
        captured = []
        def run_script(path, run_name):
            captured.append((os.environ['CUDA_VISIBLE_DEVICES'], list(sys.argv), run_name))
        with patch.dict(os.environ), patch.dict(sys.modules, {'torch': fake_torch}), \
             patch.object(worker.runpy, 'run_path', side_effect=run_script), \
             patch.object(worker.MemoryBudget, 'sample'), patch.object(sys, 'argv', ['worker.py']), \
             contextlib.redirect_stdout(io.StringIO()):
            worker.main(['--gpu', '1', '--memory-gib', '70', '--memory-out', str(self.root/'memory.json'),
                         'render.py', '--traj', 'trajectory.npz', '--out-dir', 'videos'])
        cuda.set_per_process_memory_fraction.assert_called_once_with(66/80, 0)
        self.assertEqual(captured, [('1', ['render.py', '--traj', 'trajectory.npz', '--out-dir', 'videos'], '__main__')])


if __name__ == '__main__':
    unittest.main()
