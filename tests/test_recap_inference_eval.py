"""CPU-only orchestration tests; simulator/torch subprocesses are replaced by fixtures.

Run without the robotics dependencies:
  python -m unittest discover -s tests -p test_recap_inference_eval.py -v
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recap_inference_eval.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8").split("<<'RECAP_PY'\n", 1)[1].rsplit("\nRECAP_PY", 1)[0]
recap = types.ModuleType("recap_fixture_module")
exec(compile(SOURCE, str(SCRIPT), "exec"), recap.__dict__)


def result(names):
    rows = []
    for name in names:
        failed = "bad" in name
        rows.append(dict(name=name, frames=100, steps=10 if failed else 99,
                         norm_return=.3 if failed else .9, failed=failed,
                         fail_step=10 if failed else -1, terms={"body_pos": .01}))
    summary = recap.stats(rows)
    summary.pop("failure_rate")
    return dict(summary=summary, clips=rows)


class FakeGPU(recap.Evaluation):
    """Run real file/state/report logic, substituting only GPU evaluator commands."""
    def __init__(self, args, fail_at=None):
        super().__init__(args)
        self.calls, self.fail_at = [], fail_at

    def environment(self):
        recap.save(self.out / "environment/checkpoint_config.json", {"env": {"anchor_obs": False}})
        recap.save(self.out / "environment/runtime.json", {"test_fixture": True})

    def python(self, name, script, *args, extra_env=None):
        self.calls.append(name)
        if name == self.fail_at:
            raise RuntimeError("injected evaluator failure")
        def arg(flag):
            return args[args.index(flag) + 1]
        if script.endswith("csv_to_npz_batch.py"):
            return  # Fixture already has converted payloads and a manifest.
        if script.endswith("filter_bones_seed.py"):
            recap.write_csv(arg("--out"), [dict(name=p.stem, frames=100, keep=1, reason="")
                                           for p in self.a.npz_dir.glob("*.npz")], ["name", "frames", "keep", "reason"])
        elif script.endswith("eval_benchmark.py"):
            recap.save(arg("--out"), result(recap.lines(arg("--clip-list"))))
        elif script.endswith("merge_benchmark_shards.py"):
            rows = []
            for p in args[1:args.index("--out")]:
                rows.extend(recap.load(p)["clips"])
            summary = recap.stats(rows)
            summary.pop("failure_rate")
            recap.save(arg("--out"), dict(clips=rows, summary=summary))
        elif script.endswith("diag_drift.py"):
            recap.save(arg("--out"), {"summary": {}, "clips": []})
        elif script.endswith("record_rollout.py"):
            Path(arg("--out")).write_bytes(b"fixture trajectory")
        elif script.endswith("render_rollout.py"):
            output = Path(args[1])
            output.mkdir(parents=True, exist_ok=True)
            for row in recap.read_csv(self.out / "cases/selected.csv"):
                (output / f'{row["name"]}.mp4').write_bytes(b"fixture video")
        elif script.endswith("eval_push.py"):
            text = "\n".join(f"push {s:.1f} m/s: fell 1.00%, settled 99.00%, settle time mean 0.60 s (max 1.20)"
                             for s in [0, .5, 1, 2, 4])
            (self.out / "logs/push.log").write_text(text, encoding="utf-8")
        else:
            raise AssertionError(script)


class RecapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_cwd = Path.cwd()
        self.old_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.chdir(self.root)
        (self.root / "scripts").mkdir()
        (self.root / "scripts/recap_inference_eval.sh").write_bytes(SCRIPT.read_bytes())
        (self.root / "uv.lock").write_text("fixture")
        (self.root / "model.pt").write_bytes(b"fixture checkpoint")
        data = self.root / "npz"
        data.mkdir()
        self.names = [recap.DEMO, "flip_bad", "sideway_ok", "walk_a", "walk_b", "walk_c"]
        for name in self.names:
            (data / f"{name}.npz").write_bytes(b"fixture NPZ")
        recap.write_csv(data / "manifest.csv", [dict(name=n, frames=100, fps=50, input_fps=120, duration_s=1.98)
                                                for n in self.names], ["name", "frames", "fps", "input_fps", "duration_s"])
        with gzip.open(self.root / "reference.json.gz", "wt", encoding="utf-8") as f:
            json.dump(result(self.names), f)

    def tearDown(self):
        os.chdir(self.old_cwd)
        if self.old_gpu is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.old_gpu
        self.tmp.cleanup()

    def args(self, *extra):
        return recap.arguments(["--ckpt", "model.pt", "--npz-dir", "npz", "--out", "out", "--skip-prep",
                                "--subset-size", "3", "--shard-size", "2", "--num-envs", "2",
                                "--reference", "reference.json.gz", "--no-render", *extra])

    def run_quietly(self, runner):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run()

    def test_complete_report_and_cached_rerun(self):
        runner = FakeGPU(self.args())
        self.assertEqual(self.run_quietly(runner), 0)
        data = recap.load(self.root / "out/full.json")
        self.assertEqual(data["summary"]["clips"], 6)
        self.assertEqual(data["summary"]["failed"], 1)  # Policy failure is NOT a failed script.
        self.assertNotEqual(data["summary"]["mean_norm_return"], data["summary"]["step_weighted_norm_return"])
        self.assertEqual(recap.load(self.root / "out/status.json")["status"], "completed")
        self.assertEqual(recap.load(self.root / "out/reference_comparison.json")["matched"], 6)
        self.assertEqual(len(recap.read_csv(self.root / "out/failures.csv")), 1)
        self.assertEqual(len(recap.load(self.root / "out/push.json")), 5)
        self.assertTrue((self.root / "out/cases/rollouts.npz").exists())
        self.assertIn("129,514", (self.root / "out/report.md").read_text(encoding="utf-8"))
        for r in recap.load(self.root / "out/repeatability.json"):
            self.assertEqual(r["max_abs_return_delta"], 0)
        again = FakeGPU(self.args())
        self.assertEqual(self.run_quietly(again), 0)
        self.assertEqual(again.calls, [])

    def test_failed_shard_resumes_without_repeating_completed_shard(self):
        failed = FakeGPU(self.args(), fail_at="full_001")
        self.assertEqual(self.run_quietly(failed), 1)
        self.assertFalse((self.root / "out/full.json").exists())
        self.assertNotIn("merge", failed.calls)
        self.assertEqual(recap.load(self.root / "out/status.json")["status"], "failed")
        again = FakeGPU(self.args())
        self.assertEqual(self.run_quietly(again), 0)
        self.assertNotIn("full_000", again.calls)
        self.assertIn("full_001", again.calls)
        self.assertIn("full_002", again.calls)

    def test_config_mismatch_preserves_existing_report(self):
        self.run_quietly(FakeGPU(self.args()))
        before = (self.root / "out/report.md").read_bytes()
        (self.root / "model.pt").write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "different checkpoint/config/code"):
            self.run_quietly(FakeGPU(self.args()))
        self.assertEqual(before, (self.root / "out/report.md").read_bytes())

    def test_subset_scope_and_optional_failure_are_explicit(self):
        runner = FakeGPU(self.args("--skip-full"), fail_at="push")
        self.assertEqual(self.run_quietly(runner), 0)
        status = recap.load(self.root / "out/status.json")
        self.assertEqual(status["status"], "completed_with_warnings")
        self.assertEqual(status["stages"]["push"]["status"], "failed")
        self.assertFalse((self.root / "out/full.json").exists())
        self.assertEqual(len(recap.read_csv(self.root / "out/per_clip.csv")), 3)

    def test_coverage_duplicates_and_nan_are_rejected(self):
        path = self.root / "bad.json"
        for broken in [result(self.names[:-1]), result([self.names[0], self.names[0]])]:
            recap.save(path, broken)
            with self.assertRaises(ValueError):
                recap.checked_result(path, self.names)
        broken = result(self.names)
        broken["clips"][0]["norm_return"] = float("nan")
        path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            recap.checked_result(path, self.names)

    def test_missing_conversion_is_rejected_even_if_converter_exits_zero(self):
        args = self.args()
        args.skip_prep = False
        runner = FakeGPU(args)
        try:
            (runner.out / "source_csv.txt").write_text("\n".join(n + ".csv" for n in self.names + ["missing"]) + "\n")
            with self.assertRaisesRegex(ValueError, "CSV/NPZ coverage mismatch"):
                runner.audit_data()
        finally:
            runner.log_file.close()

    def test_fps_mismatch_is_rejected(self):
        rows = recap.read_csv(self.root / "npz/manifest.csv")
        rows[0]["fps"] = 120
        recap.write_csv(self.root / "npz/manifest.csv", rows, list(rows[0]))
        runner = FakeGPU(self.args())
        try:
            with self.assertRaisesRegex(ValueError, "invalid FPS"):
                runner.audit_data()
        finally:
            runner.log_file.close()

    def test_default_extraction_and_preparation_flow(self):
        archive_path = self.root / "g1.tar.gz.part"
        with tarfile.open(archive_path, "w:gz") as archive:
            for n in self.names:
                info = tarfile.TarInfo(f"g1/csv/day/{n}.csv")
                payload = b"fixture CSV"
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        args = self.args()
        args.skip_prep = False
        args.archive = archive_path
        runner = FakeGPU(args)
        self.assertEqual(self.run_quietly(runner), 0)
        self.assertTrue((args.raw_dir / "g1/csv/day/flip_bad.csv").exists())
        self.assertEqual(len(recap.lines(runner.out / "source_csv.txt")), 6)
        self.assertTrue(recap.load(runner.out / "data_audit.json")["source_coverage_checked"])

    def test_real_subprocess_failure_is_not_hidden(self):
        runner = FakeGPU(self.args())
        try:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "code 7"):
                runner.command("broken_child", [sys.executable, "-c", "print('child error'); raise SystemExit(7)"])
            self.assertIn("child error", (runner.out / "logs/broken_child.log").read_text())
        finally:
            runner.log_file.close()

    def test_checkpoint_config_with_path_fields_is_exported(self):
        runner = FakeGPU(self.args())
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, synchronize=lambda: None,
                                       get_device_name=lambda _: "fixture GPU"),
            zeros=lambda *args, **kwargs: object(), __version__="fixture", version=types.SimpleNamespace(cuda="fixture"),
            load=lambda *args, **kwargs: {"policy": {}, "config": {"data": {"npz_dir": Path("data/npz/bones_seed")},
                                                                 "log_root": Path("logs")}})
        try:
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(recap.subprocess, "run", return_value=types.SimpleNamespace(stdout="fixture", stderr="")):
                recap.Evaluation.environment(runner)
            config = recap.load(runner.out / "environment/checkpoint_config.json")
            self.assertEqual(config["log_root"], "logs")
            self.assertEqual(config["data"]["npz_dir"], str(Path("data/npz/bones_seed")))
        finally:
            runner.log_file.close()

    def test_push_no_survivors_and_case_selection(self):
        text = "\n".join(f"push {s:.1f} m/s: fell 100.00%, settled 0.00%, settle time mean nan s (max nan)" for s in [0, .5, 1, 2, 4])
        self.assertIsNone(recap.parse_push(text)[-1]["mean_settle_s"])
        chosen = recap.select_cases(result(self.names)["clips"])
        self.assertEqual(len(chosen), len({r["name"] for r in chosen}))
        self.assertTrue(any(r["case_type"] == "early_failure" for r in chosen))

    def test_render_default_and_add_videos_to_completed_run(self):
        self.assertTrue(recap.arguments([]).render)
        self.assertFalse(recap.arguments(["--no-render"]).render)
        self.assertEqual(self.run_quietly(FakeGPU(self.args())), 0)
        with patch.object(recap.shutil, "which", return_value="fixture_ffmpeg"):
            runner = FakeGPU(self.args("--render"))
            self.assertEqual(self.run_quietly(runner), 0)
        self.assertEqual(runner.calls, ["render"])
        selected = recap.read_csv(self.root / "out/cases/selected.csv")
        videos = list((self.root / "out/cases/videos").glob("*.mp4"))
        self.assertEqual(len(videos), len(selected))
        report = (self.root / "out/report.md").read_text(encoding="utf-8")
        self.assertIn(f"cases/videos/{videos[0].name}", report)

    def test_render_failure_preserves_numerical_results(self):
        with patch.object(recap.shutil, "which", return_value="fixture_ffmpeg"):
            runner = FakeGPU(self.args("--render"), fail_at="render")
            self.assertEqual(self.run_quietly(runner), 0)
        status = recap.load(self.root / "out/status.json")
        self.assertEqual(status["status"], "completed_with_warnings")
        self.assertEqual(status["stages"]["render"]["status"], "failed")
        self.assertEqual(recap.load(self.root / "out/full.json")["summary"]["clips"], 6)
        self.assertTrue((self.root / "out/cases/rollouts.npz").exists())


if __name__ == "__main__":
    unittest.main()
