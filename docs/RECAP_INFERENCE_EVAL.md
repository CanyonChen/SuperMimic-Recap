**预训练权重自动测评**

在已经配置好 uv 环境的 H100 服务器上，从仓库根目录执行：

```bash
bash scripts/recap_inference_eval.sh --gpu 1
```

默认读取 `data/checkpoints/e8/e8_boneseed_ckpt_0100000.pt` 和 `data/raw/bones_seed/g1.tar.gz.part`。流程包括 CSV 解压、120→50 FPS 转换、过滤、单动作、2,000 条固定子集重复 3 次、全量分片、案例诊断与录制、扰动恢复、发布结果对照和 MP4 渲染。不进行训练。

| 选项 | 用途 |
|---|---|
| `--skip-prep` | 已有完整 `data/npz/bones_seed/` 和 `manifest.csv` 时跳过解压、转换 |
| `--manifest PATH` | 复用已有过滤清单，保证训练和评测使用同一批动作；必须覆盖转换清单的全部行并保留 `keep` 标记 |
| `--skip-full --out logs/repro_e8_subset` | 只做子集，使用独立结果目录 |
| `--no-render` | 跳过默认开启的视频渲染 |
| `--render-only` | 为已完成的测评补做视频并更新报告；只需原输出目录及轨迹，不重新推理 |
| `--skip-push` / `--skip-reference` | 跳过扰动 / 第三方结果下载对照 |
| `--ckpt` / `--archive` / `--npz-dir` / `--out` | 改用指定路径；相对路径按仓库根目录解释 |
| `--help` | 查看全部配置选项 |

**双 H100 并行验证：GPU 0 预算 40 GiB，GPU 1 预算 70 GiB**

两张卡分别运行独立的确定性动作测评，动态领取不同分片，不需要 DDP 或梯度同步。
每卡只启动一个常驻进程，策略只加载一次；数据目录也只列举一次。分片仍分别建立仿真环境。
下面配置使用 1,024 / 2,048 个并行环境，每片最多 8,192 条动作；任务不足时自动降低实际环境数。
这些参数增加吞吐量，具体 H100 加速比和峰值显存需要服务器实测。

若旧的单卡验证仍在运行，先停止它；在 `tail -f` 中按 Ctrl+C 只退出日志查看。
以下针对之前 `--gpu 0 --ckpt logs/repro_train_2xh100/train/ckpt_0152000.pt` 的启动命令，只匹配该 Python 调度进程。
SIGTERM 会让它停止自己的评测子进程，并保存中断状态和已完成结果。

```bash
cd /home/jovyan/canyon/SuperMimic-Recap

OLD_EVAL_PATTERN='^/home/jovyan/canyon/SuperMimic-Recap/\.venv/bin/python[^ ]* -u - --gpu 0 --ckpt logs/repro_train_2xh100/train/ckpt_0152000\.pt([[:space:]]|$)'
pkill -TERM -f "$OLD_EVAL_PATTERN" || true
while pgrep -f "$OLD_EVAL_PATTERN" >/dev/null; do
  echo '等待旧验证退出……'
  sleep 3
done

git pull --ff-only origin main
```

确认上面的 `git pull` 成功后，启动新验证：

```bash
nohup bash scripts/recap_inference_eval.sh \
  --gpus 0,1 --vram-gb 40,70 \
  --num-envs 1024 --num-envs-per-gpu 1024,2048 \
  --ckpt logs/repro_train_2xh100/train/ckpt_0152000.pt \
  --npz-dir data/npz/bones_seed \
  --manifest logs/repro_train_2xh100/manifest_filtered.csv \
  --skip-prep --subset-size 2000 --repeats 3 --shard-size 8192 \
  --reuse-from logs/repro_train_2xh100/eval_0152000 \
  --out logs/repro_train_2xh100/eval_0152000_dual \
  --render \
  > logs/recap_eval_0152000_dual_launcher.log 2>&1 < /dev/null &

tail -f logs/recap_eval_0152000_dual_launcher.log
```

`--reuse-from` 校验权重、数据文件清单、过滤清单和评测实现，按动作名导入旧目录中已完成的全量分片。
支持将原来每片 2,000 条动作重新组合到 8,192 条的新分片，缺失部分才重新计算。
原目录保留不动，新的单动作/三轮子集/案例/扰动测试仍执行，报告注明每片 GPU、环境数和旧结果来源。
允许复用本项目发布版本 `f6d52d5` 的原始测评，以及记录了相同核心评测指纹的新版本；代码无法核验时不会混用结果。
原始数据目录只审计一次，不会每卡重复审计、转换或过滤。
如果旧任务还没完成任何全量分片，会记录复用数量为 0 并执行全量测试。

每张卡的 PyTorch 分配上限分别为 36 / 66 GiB，另外预留 4 GiB 给 Warp、CUDA 上下文等。
进程总显存每 5 秒通过 `nvidia-smi` 采样，发现超预算会停止并保留已完成分片。
这是应用预算，不是 MIG 硬隔离；容器内 PID 无法映射时，`memory/` 会明确记录总显存监测不可用。
预算不会预先占满显存，也不代表越接近预算就越快。

新目录增加：

| 文件 | 内容 |
|---|---|
| `execution.json` | 每卡环境数/预算、并行墙钟时间、各 worker 耗时之和、复用动作数 |
| `reuse.json` | 旧结果目录、校验信息、导入分片及其原有执行配置 |
| `memory/*.json` | PyTorch 峰值、采样到的进程显存峰值、监测限制 |
| `logs/worker_0.log`、`logs/worker_1.log` | 每卡的初始化和逐片动作进度 |
| `workers/` | 每次启动的任务清单、临时结果、逐片执行与显存记录 |
| `launcher.pid` | 当前调度进程 PID；正常退出后删除 |

相同新命令重跑可续接新目录；不要对同一个输出目录同时启动两份调度器。
需要再次停止新任务时，用 `kill -TERM "$(cat logs/repro_train_2xh100/eval_0152000_dual/launcher.pid)"`。
`summary.csv` 的 `full` 耗时按并行墙钟时间统计，导入结果的旧耗时不计入；完整本次耗时另见 `execution.json`。
不同 GPU/环境批量可能带来浮点数值差异，报告中的重复性结果也会包含这些执行配置的差异。
之前 e8 测评的 `reference.json.gz` 若存在，会直接复用，无需再次下载。

**输出：`logs/repro_e8_auto/`**

| 文件或目录 | 内容 |
|---|---|
| `report.md` | 中文汇总报告、论文参照、视频链接及可比性限制 |
| `summary.csv`、`summary.json` | 各轮失败率、回报、总耗时 |
| `single.json`、`subset_*.json`、`full.json`、`full/` | 原始逐动作及分片结果 |
| `per_clip.csv`、`failures.csv`、`groups.csv` | 逐动作、失败清单、按名称分组统计 |
| `repeatability.json` | 固定子集重复运行差异 |
| `reference_comparison.json/csv` | 成功获取第三方基准后的同名动作对照 |
| `cases/selected.csv`、`cases/drift.json`、`cases/rollouts.npz` | 自动选取的案例、漂移诊断、实际与参考轨迹 |
| `cases/videos/*.mp4` | 渲染成功后，每个选中案例一个 25 FPS 视频 |
| `push.json`、`push.csv` | 成功完成扰动测试后的恢复指标 |
| `environment/`、`run_config.json`、`data_audit.json`、`lists/` | 环境、权重配置、数据审计和固定动作清单 |
| `status.json`、`run.log`、`logs/` | 阶段状态、完整日志、子脚本日志 |

视频中实体为策略执行结果，半透明为参考动作。渲染需要 ffmpeg 和可用的 OpenGL 后端，默认 `MUJOCO_GL=egl`，可通过环境变量改用已配置的后端。渲染失败会保留数值结果和轨迹，在报告中标记警告；不会生成占位视频。

相同命令重跑会复用已经检查通过的阶段和分片。更换权重、数据、评测参数或代码版本时使用新的 `--out`。同一代码版本下，可以对原来 `--no-render` 的运行去掉该选项或增加 `--render`，只补视频。核心阶段失败退出并保存部分报告；可选阶段失败时标记 `completed_with_warnings`。

若测评已经结束，只是视频渲染失败（包括旧版误用位置参数导致缺少 `--traj` / `--out-dir`），更新代码后执行：

```bash
git pull --ff-only origin main
bash scripts/recap_inference_eval.sh --render-only --gpu 1 --out logs/repro_e8_auto
```

此模式沿用原 `run_config.json` 的评测配置，只读取已录制的 `cases/rollouts.npz` 渲染视频，更新渲染状态和报告链接；不会重新解压、转换或推理，也不会改写原权重/代码指纹。其他阶段仍有警告时，流程状态继续保留 `completed_with_warnings`。原输出目录必须已完成测评并保留轨迹、案例清单和报告所需的结果文件。

论文表 A.8 的主参照为 25/129,514（单 H100 训练 3 h）及 0/129,514（双 H100 训练 9.8 h）。当前是第三方权重，数据过滤集合存在差异，且仓库回报按实际执行步数归一化；不能据此宣称训练效率、完整论文回报口径或真机结果已经复现。案例分组和稳定判据也需注明仓库口径。

CPU 流程测试（使用模拟评测输出，不验证 GPU 推理或实际视频渲染）：

```bash
python -m unittest discover -s tests -p test_recap_inference_eval.py -v
python -m unittest discover -s tests -p test_recap_parallel_eval.py -v
```
