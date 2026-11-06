**预训练权重自动测评**

在已经配置好 uv 环境的 H100 服务器上，从仓库根目录执行：

```bash
bash scripts/recap_inference_eval.sh --gpu 1
```

默认读取 `data/checkpoints/e8/e8_boneseed_ckpt_0100000.pt` 和 `data/raw/bones_seed/g1.tar.gz.part`。流程包括 CSV 解压、120→50 FPS 转换、过滤、单动作、2,000 条固定子集重复 3 次、全量分片、案例诊断与录制、扰动恢复、发布结果对照和 MP4 渲染。不进行训练。

| 选项 | 用途 |
|---|---|
| `--skip-prep` | 已有完整 `data/npz/bones_seed/` 和 `manifest.csv` 时跳过解压、转换 |
| `--skip-full --out logs/repro_e8_subset` | 只做子集，使用独立结果目录 |
| `--no-render` | 跳过默认开启的视频渲染 |
| `--render-only` | 为已完成的测评补做视频并更新报告；只需原输出目录及轨迹，不重新推理 |
| `--skip-push` / `--skip-reference` | 跳过扰动 / 第三方结果下载对照 |
| `--ckpt` / `--archive` / `--npz-dir` / `--out` | 改用指定路径；相对路径按仓库根目录解释 |
| `--help` | 查看全部配置选项 |

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
```
