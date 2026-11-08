"""CPU orchestration and real Tyro contract tests; no CUDA training is claimed here."""
from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass, field
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_recap_inference_eval import FakeGPU, result

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


training = module(ROOT / 'scripts/recap_training_eval.py', 'training_fixture')
worker = module(ROOT / 'scripts/recap_training_worker.py', 'training_worker_fixture')


class FakeTraining(training.TrainingEvaluation):
    def __init__(self, args, fail_at=None, inner_failure=None, free_gib=44.5):
        super().__init__(args)
        self.calls, self.fail_at, self.inner_failure, self.free_gib = [], fail_at, inner_failure, free_gib

    def gpu_info(self):
        return dict(name='fixture GPU', free_gib=self.free_gib, total_gib=80, uuid='GPU-fixture', driver='fixture')

    def environment(self):
        training.save(self.out / 'environment/runtime.json', self.ensure_memory())

    def inspect_library(self, path):
        data = training.load(path)
        if data['names'] != self.kept or data['lengths'] != self.lengths:
            raise ValueError('library data mismatch')

    def checkpoint_info(self, path, stage, target):
        checkpoint = training.load(path)
        if not training.profile_matches(checkpoint['config'], self.a, stage, target):
            raise ValueError('profile mismatch')
        return dict(path=str(path), iteration=checkpoint['iteration'], num_updates=checkpoint['num_updates'])

    def export_curves(self):
        training.write_csv(self.out / 'training_curves.csv',
            [dict(iteration=self.a.iterations, tag='wm/loss', value=.01, wall_time=1)],
            ['iteration', 'tag', 'value', 'wall_time'])
        training.save(self.out / 'training_metrics.json', {'wm/loss': {'value': .01, 'iteration': self.a.iterations}})

    def python(self, name, script, *args, extra_env=None):
        args = [str(x) for x in args]
        self.calls.append((name, args))
        def arg(flag):
            return args[args.index(flag)+1]
        if script.endswith('build_library.py'):
            training.save(arg('--out'), dict(names=self.kept, lengths=self.lengths))
        elif script.endswith('filter_bones_seed.py'):
            training.write_csv(arg('--out'), [dict(name=p.stem, frames=100, keep=1)
                for p in self.a.npz_dir.glob('*.npz')], ['name', 'frames', 'keep'])
        elif script.endswith('recap_training_worker.py'):
            target = int(arg('--schedule.total-iterations'))
            config = dict(data={'library': str(self.a.library)}, run_name=name, log_root=str(self.out), seed=self.a.seed,
                env={'num_envs': self.a.num_envs, 'replay_capacity': self.a.replay_capacity},
                policy_trainer={'batch_size': self.a.policy_batch}, wm_trainer={'batch_size': self.a.wm_batch},
                schedule={'total_iterations': target})
            def ck(step):
                return dict(config=config, iteration=step, num_updates=max(0, step-40)//3)
            directory = self.out / name
            if '--resume' not in args:
                training.save(directory / 'ckpt_0000000.pt', ck(0))
            end = target//2 if name == self.fail_at else target
            training.save(directory / 'ckpt_latest.pt', ck(end))
            for step in range(self.a.checkpoint_interval, end+1, self.a.checkpoint_interval):
                training.save(directory / f'ckpt_{step:07d}.pt', ck(step))
            runtime = dict(status='failed' if name == self.fail_at else 'completed',
                iteration=end, num_updates=ck(end)['num_updates'], process_wall_s=10, loop_s=8,
                sampled_process_peak_gib=23.5, resume=arg('--resume') if '--resume' in args else None)
            training.save(directory / 'runtime.json', runtime)
            number = len(list(directory.glob('runtime_*.json')))
            training.save(directory / f'runtime_{number}.json', runtime)
            if name == self.fail_at:
                raise RuntimeError('injected interrupted training')
        elif script.endswith('eval_benchmark.py'):
            training.save(arg('--out'), result(training.lines(arg('--clip-list'))))
        else:
            raise AssertionError(script)

    def command(self, name, command, extra_env=None):
        self.calls.append((name, [str(c) for c in command]))
        if name != 'evaluation':
            raise AssertionError(name)
        # Run the actual inference orchestrator with its existing GPU fixture.
        evaluator = FakeGPU(training.recap.arguments([str(c) for c in command[2:]]), fail_at=self.inner_failure)
        with patch.object(training.shutil, 'which', return_value='fixture_ffmpeg'):
            code = evaluator.run()
        if code:
            raise RuntimeError('inference subprocess failed')


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.old_cwd, self.old_gpu = Path.cwd(), os.environ.get('CUDA_VISIBLE_DEVICES')
        os.chdir(self.directory)
        (self.directory / 'scripts').mkdir()
        for name in ('recap_inference_eval.sh', 'recap_training_eval.sh', 'recap_training_eval.py', 'recap_training_worker.py'):
            (self.directory / 'scripts' / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
        (self.directory / 'uv.lock').write_text('fixture lock')
        (self.directory / 'npz').mkdir()
        self.names = [training.recap.DEMO, 'walk_a', 'walk_b', 'walk_c', 'flip_bad', 'excluded']
        for name in self.names:
            (self.directory / 'npz' / f'{name}.npz').write_bytes(b'fixture NPZ')
        training.write_csv(self.directory / 'npz/manifest.csv',
            [dict(name=n, frames=100, fps=50, input_fps=120, duration_s=1.98) for n in self.names],
            ['name', 'frames', 'fps', 'input_fps', 'duration_s'])
        training.write_csv(self.directory / 'filter.csv',
            [dict(name=n, frames=100, keep=int(n != 'excluded')) for n in self.names], ['name', 'frames', 'keep'])

    def tearDown(self):
        os.chdir(self.old_cwd)
        if self.old_gpu is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.old_gpu
        self.tmp.cleanup()

    def args(self, *extra):
        return training.arguments(['--npz-dir', 'npz', '--manifest', 'filter.csv', '--library', 'library.pt',
            '--out', 'out', '--iterations', '200', '--checkpoint-interval', '50', '--eval-every', '100',
            '--num-envs', '8', '--policy-batch', '4', '--wm-batch', '4', '--replay-capacity', '64',
            '--eval-num-envs', '2', '--subset-size', '3', '--shard-size', '2', '--repeats', '2',
            '--skip-reference', '--skip-push', *extra])

    def run_quietly(self, runner):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run()

    def test_fresh_training_full_evaluation_videos_and_cached_rerun(self):
        runner = FakeTraining(self.args())
        self.assertEqual(self.run_quietly(runner), 0)
        state = training.load(self.directory / 'out/status.json')
        self.assertEqual(state['status'], 'completed')
        for name, args in runner.calls:
            if name in ('smoke', 'train'):
                self.assertNotIn('--resume', args)
                self.assertIn('--memory-gib', args)
            if name == 'evaluation':
                self.assertIn('--manifest', args)
                self.assertIn('--skip-prep', args)
        summary = training.load(self.directory / 'out/training_summary.json')
        self.assertEqual(summary['checkpoint']['iteration'], 200)
        self.assertEqual(summary['training_clips'], 5)
        self.assertEqual(summary['training_process_wall_s'], 10)
        report = (self.directory / 'out/report.md').read_text(encoding='utf-8')
        self.assertIn('| full | 5 | 1 | 20.0000%', report)
        full = training.load(self.directory / 'out/evaluation/full.json')
        self.assertEqual(full['summary']['clips'], 5)
        self.assertEqual(full['summary']['failed'], 1)
        self.assertTrue(list((self.directory / 'out/evaluation/cases/videos').glob('*.mp4')))
        self.assertEqual([int(r['iteration']) for r in training.read_csv(self.directory / 'out/checkpoint_eval.csv')], [0, 100, 200])
        again = FakeTraining(self.args())
        self.assertEqual(self.run_quietly(again), 0)
        self.assertEqual(again.calls, [])

    def test_failed_training_resumes_own_checkpoint_without_repeating_smoke(self):
        first = FakeTraining(self.args(), fail_at='train')
        self.assertEqual(self.run_quietly(first), 1)
        self.assertFalse((self.directory / 'out/evaluation/full.json').exists())
        second = FakeTraining(self.args())
        self.assertEqual(self.run_quietly(second), 0)
        self.assertNotIn('smoke', [name for name, _ in second.calls])
        args = next(args for name, args in second.calls if name == 'train')
        self.assertEqual(Path(args[args.index('--resume')+1]), self.directory / 'out/train/ckpt_latest.pt')
        self.assertEqual(training.load(self.directory / 'out/training_summary.json')['attempts'], 2)

    def test_smoke_only_then_formal_run_starts_from_zero(self):
        first = FakeTraining(self.args('--smoke-only'))
        self.assertEqual(self.run_quietly(first), 0)
        self.assertEqual(training.load(self.directory / 'out/status.json')['status'], 'smoke_completed')
        self.assertNotIn('train', [name for name, _ in first.calls])
        second = FakeTraining(self.args())
        self.assertEqual(self.run_quietly(second), 0)
        args = next(args for name, args in second.calls if name == 'train')
        self.assertNotIn('--resume', args)
        self.assertNotIn('smoke', [name for name, _ in second.calls])

    def test_low_vram_stops_before_library_or_training(self):
        runner = FakeTraining(self.args(), free_gib=32)
        self.assertEqual(self.run_quietly(runner), 1)
        self.assertEqual(runner.calls, [])
        self.assertIn('not enough free VRAM', training.load(self.directory / 'out/status.json')['error'])

    def test_config_change_cannot_overwrite_completed_experiment(self):
        self.run_quietly(FakeTraining(self.args()))
        paths = [self.directory / 'out' / name for name in ['run_config.json', 'report.md', 'train/ckpt_latest.pt']]
        before = [p.read_bytes() for p in paths]
        with self.assertRaisesRegex(ValueError, 'different training config'):
            self.run_quietly(FakeTraining(self.args('--policy-batch', '8')))
        self.assertEqual([p.read_bytes() for p in paths], before)

    def test_optional_video_failure_can_retry_without_retraining(self):
        self.assertEqual(self.run_quietly(FakeTraining(self.args(), inner_failure='render')), 0)
        self.assertEqual(training.load(self.directory / 'out/status.json')['status'], 'completed_with_warnings')
        checkpoint = (self.directory / 'out/train/ckpt_latest.pt').read_bytes()
        again = FakeTraining(self.args())
        self.assertEqual(self.run_quietly(again), 0)
        self.assertEqual([name for name, _ in again.calls], ['evaluation'])
        self.assertEqual((self.directory / 'out/train/ckpt_latest.pt').read_bytes(), checkpoint)
        self.assertEqual(training.load(self.directory / 'out/status.json')['status'], 'completed')

    def test_add_videos_after_no_render_without_retraining(self):
        self.assertEqual(self.run_quietly(FakeTraining(self.args('--no-render'))), 0)
        again = FakeTraining(self.args())
        self.assertEqual(self.run_quietly(again), 0)
        self.assertEqual([name for name, _ in again.calls], ['evaluation'])
        self.assertTrue(list((self.directory / 'out/evaluation/cases/videos').glob('*.mp4')))

    def test_corrupt_latest_checkpoint_falls_back_to_numbered_checkpoint(self):
        self.run_quietly(FakeTraining(self.args(), fail_at='train'))
        (self.directory / 'out/train/ckpt_latest.pt').write_bytes(b'incomplete checkpoint')
        again = FakeTraining(self.args())
        self.assertEqual(self.run_quietly(again), 0)
        args = next(args for name, args in again.calls if name == 'train')
        self.assertTrue(args[args.index('--resume')+1].endswith('ckpt_0000100.pt'))

    def test_filter_mismatch_rejected_before_library(self):
        rows = training.read_csv(self.directory / 'filter.csv')[:-1]
        training.write_csv(self.directory / 'filter.csv', rows, ['name', 'frames', 'keep'])
        runner = FakeTraining(self.args())
        self.assertEqual(self.run_quietly(runner), 1)
        self.assertFalse(runner.calls)

    def test_real_tyro_accepts_training_flags_and_data_builder_flags(self):
        try:
            import tyro
        except ImportError:
            self.skipTest('install tyro for CLI contract checks')
        # Compile the real dataclass definitions, with unrelated model architecture defaults stubbed.
        @dataclass
        class ModelConfig:
            pass
        namespace = dict(__name__=__name__, dataclass=dataclass, field=field, Path=Path,
                         PolicyCfg=ModelConfig, CriticCfg=ModelConfig, WorldModelCfg=ModelConfig)
        for source, names in [('src/supermimic/world_model/wm_trainer.py', {'WMTrainerCfg'}),
                              ('src/supermimic/imagination/policy_trainer.py', {'PolicyTrainerCfg'}),
                              ('src/supermimic/config.py', {'DataCfg', 'EnvCfg', 'ScheduleCfg', 'TrainConfig'})]:
            tree = ast.parse((ROOT / source).read_text(encoding='utf-8'))
            definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), source, 'exec', dont_inherit=True), namespace)
        config = tyro.cli(namespace['TrainConfig'], args=training.training_flags(self.args(), 'train', 200))
        self.assertTrue(training.profile_matches(vars_tree(config), self.args(), 'train', 200))
        self.assertEqual(config.policy_trainer.horizon, 32)
        self.assertEqual(config.wm_trainer.window, 8)
        self.assertTrue(config.env.compact_library)
        self.assertEqual(config.data.manifest, self.directory / 'out/manifest_filtered.csv')
        self.assertIsNone(config.resume)
        source = ROOT / 'scripts/data/build_library.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        main.body = [ast.Return(value=ast.Constant(value=None))]
        signature = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
        exec(compile(signature, str(source), 'exec', dont_inherit=True), namespace)
        self.assertIsNone(tyro.cli(namespace['main'], args=['--npz-dir', 'npz', '--manifest', 'filter.csv',
                                                        '--out', 'packed.pt', '--workers', '8']))


def vars_tree(config):
    from dataclasses import asdict
    return asdict(config)


class WorkerTests(unittest.TestCase):
    def test_duplicate_nvml_rows_are_not_added_and_other_jobs_are_excluded(self):
        self.assertEqual(worker.process_memory_gib('11, 4096\n11, 4096\n12, 64000\n', 11), 4)
        self.assertIsNone(worker.process_memory_gib('12, 64000\n', 11))

    def test_budget_and_non_finite_training_loss_are_detected(self):
        cuda = types.SimpleNamespace(**{name: lambda: 20*worker.GIB for name in
            ['memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved']})
        writer = types.SimpleNamespace(add_scalar=lambda *a, **kw: None, close=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            monitor = worker.BudgetWriter(writer, types.SimpleNamespace(cuda=cuda), Path(directory), '1', 40)
            try:
                with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                    monitor.add_scalar('wm/loss', float('nan'), 50)
                output = types.SimpleNamespace(stdout=f'{os.getpid()}, 45056\n')
                with patch.object(worker.subprocess, 'run', return_value=output), self.assertRaisesRegex(RuntimeError, 'exceeding'):
                    monitor.sample(50)
            finally:
                monitor.close()

    def test_interrupted_checkpoint_write_preserves_last_good_checkpoint(self):
        source = ROOT / 'src/supermimic/training/loop.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        save = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'save')
        def incomplete_save(payload, path):
            Path(path).write_bytes(b'partial checkpoint')
            raise RuntimeError('interrupted write')
        namespace = dict(torch=types.SimpleNamespace(save=incomplete_save), asdict=lambda cfg: {})
        exec(compile(ast.Module(body=[save], type_ignores=[]), str(source), 'exec', dont_inherit=True), namespace)
        model = types.SimpleNamespace(state_dict=lambda: {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ckpt_latest.pt'
            path.write_bytes(b'last good checkpoint')
            trainer = types.SimpleNamespace(is_main=True, log_dir=Path(directory), iteration=100, num_updates=24,
                wm=model, policy=model, critic=model, wm_trainer=types.SimpleNamespace(opt=model),
                policy_trainer=types.SimpleNamespace(opt=model, critic_opt=model),
                command=types.SimpleNamespace(bin_return=0, bin_memory=0), cfg={})
            with self.assertRaisesRegex(RuntimeError, 'interrupted write'):
                namespace['save'](trainer)
            self.assertEqual(path.read_bytes(), b'last good checkpoint')


if __name__ == '__main__':
    unittest.main()
