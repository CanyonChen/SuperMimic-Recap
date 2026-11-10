# Dual-GPU inference evaluation

The checkpoint under evaluation is `ckpt_0152000.pt`; training was stopped before the 320,000-iteration target.

- `scripts/recap_inference_eval.sh` keeps the original single-GPU interface and adds `--gpus`, per-GPU budgets/environment counts, and `--reuse-from`.
- `scripts/recap_parallel_eval.py` schedules one persistent worker per GPU and merges partial cached results by motion name. Only verified, completed full-set shards are imported; the source directory stays unchanged.
- `scripts/recap_eval_worker.py` reuses the policy and file inventory, applies the PyTorch memory allowance, and samples total process GPU memory. The application budget is not a hard GPU partition.
- Requested server profile: GPU 0 = 40 GiB / 1,024 environments; GPU 1 = 70 GiB / 2,048 environments; 8,192 motions per shard. All original evaluation projects and MP4 output remain enabled.
- Server stop/pull/launch instructions and result paths are in `docs/RECAP_INFERENCE_EVAL.md`.
- Validation uses CPU fixture subprocesses for concurrency, persistence, interrupted-run recovery, cross-shard reuse and memory-budget plumbing. H100 throughput, CUDA execution and EGL rendering require the server run; no measured GPU speedup is claimed.
- The IDE's original local workspace path was inaccessible in this session. Changes were prepared in a clean temporary checkout of `CanyonChen/SuperMimic-Recap`, based on `f6d52d5`.
