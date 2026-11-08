**40 GiB 显存预算：从零训练并自动测评**

在已经跑通推理的服务器环境中执行。脚本复用现有 50 FPS NPZ 数据，从随机初始化训练世界模型、策略和 Critic；训练完成后，自动测评自己训练的权重并生成案例 MP4。

```bash
cd /home/jovyan/canyon/SuperMimic-Recap
git pull --ff-only origin main
mkdir -p logs
nohup bash scripts/recap_training_eval.sh --gpu 1 --vram-gb 40 \
  > logs/recap_training_launcher.log 2>&1 < /dev/null &
tail -f logs/recap_training_launcher.log
```

`nohup` 让训练在 SSH 断开后继续。上面的 `tail -f` 仅查看日志，按 Ctrl+C 退出查看不会停止后台训练。不要同时启动两份命令写入同一结果目录。

**自动执行顺序**

1. 检查 GPU 1 的可用显存，记录环境、代码版本和参数。每次启动试训或正式训练前，都要求至少有 40 GiB 空闲。
2. 校验 `data/npz/bones_seed/manifest.csv` 和 50 FPS 数据。优先复用 `logs/repro_e8_auto/manifest_filtered.csv`；不存在时尝试 NPZ 目录中的过滤清单，仍不存在则自动过滤。数据不会重新解压或转换。
3. 生成 FP16、71 维紧凑动作库 `data/library/bones_seed_train_40g.pt`，减少 GPU 上的动作库占用。训练与测评共用同一份过滤清单。
4. 用正式训练的环境数和批量试训 100 次迭代，验证编译、参数更新、权重保存和显存。通过后重新随机初始化，开始正式训练。
5. 正式训练 320,000 次迭代，每 4,000 次保存检查点，每 50 次写日志和显存采样。
6. 训练结束后，对第 0、40k、80k、100k、120k、160k、200k、240k、280k 次迭代的权重做同一组 2,000 动作子集测评，生成性能随迭代变化的数据。
7. 对最终权重执行原有推理流程：单动作、固定 2,000 动作子集重复 3 次、全部过滤动作分片测评、失败诊断、扰动测试、第三方结果对照及案例视频。最终子集结果也加入检查点曲线。

检查点测评和最终测评均在训练退出后顺序执行。不会同时把训练与测评放在 GPU 上。视频默认开启，文件数由案例筛选结果决定，不固定为上次的 9 个。

**默认训练配置**

| 项目 | 默认值 |
|---|---|
| GPU | 物理 GPU 1；进程内映射为 `cuda:0` |
| 显存预算 | 40 GiB |
| 并行仿真环境 | 4,096 |
| 策略批量 / 世界模型批量 | 2,048 / 2,048 |
| 回放容量 | 每个环境 1,000 步 |
| 想象长度 / 世界模型训练窗口 | H=32 / k=8 |
| UTD | 0.4 |
| 正式训练 / 独立试训 | 320,000 / 100 次迭代 |
| 随机种子 | 0 |
| 测评并行环境 / 全量分片大小 | 256 / 2,000 动作 |

一次训练迭代包含所有并行环境的一次控制步，不能把 32 万次迭代理解为仅采集 32 万条转移。模型结构、噪声、随机化、Critic 和学习率调度沿用仓库配置；完整配置保存在 `train/config.json`。

仓库 [EXPERIMENTS.md](EXPERIMENTS.md) 中相近的紧凑库配置在 RTX 5090 上约占 20～22 GB。这是选取默认批量的依据，H100 上的实际占用以本次试训为准。PyTorch 缓存分配器额度设为 36 GiB，另留 4 GiB 给 Warp 和 CUDA 上下文；日志点检测到本训练进程超过 40 GiB 时停止。该额度不是 GPU 硬隔离，也不限制其他用户的进程；容器 PID 不可见时，只能记录 PyTorch 峰值，报告会说明进程采样缺失。[分配器额度的官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.set_per_process_memory_fraction.html)

首次打包动作库会占用主机 RAM 和磁盘；构建器会同时保留各动作数组及拼接结果。紧凑库约数 GB，32 万步会保存约 80 份中间权重，整个训练输出可能达到几十 GB。首次 `torch.compile` 编译耗时不代表稳定训练速度。

**输出：`logs/repro_train_40g/`**

| 文件或目录 | 内容 |
|---|---|
| `report.md`、`training_summary.json` | 总结、实测失败率/回报、耗时、显存与论文对比边界 |
| `train/ckpt_latest.pt`、`train/ckpt_*.pt` | 正式训练最终及中间权重，包含模型、优化器和采样器状态 |
| `train/config.json`、`train/checkpoint_metadata.json` | 完整训练配置、完成迭代数和更新次数 |
| `train/tb/`、`training_curves.csv`、`training_metrics.json` | TensorBoard 事件、全部标量曲线、各指标最后记录值 |
| `train/memory.csv`、`train/runtime*.json` | 显存采样、PyTorch 峰值、每次训练进程耗时和续训记录 |
| `checkpoint_eval/`、`checkpoint_eval.csv` | 中间权重子集结果，以及失败率/回报随迭代变化的数据 |
| `smoke/` | 独立试训的权重、日志和显存记录 |
| `evaluation/report.md`、`evaluation/summary.csv` | 最终权重的完整推理测评报告和汇总 |
| `evaluation/full.json`、`evaluation/per_clip.csv`、`evaluation/failures.csv` | 最终权重全量逐动作结果及失败清单 |
| `evaluation/cases/videos/*.mp4` | 案例视频；实体为策略结果，半透明为参考动作 |
| `evaluation/` 下其余文件 | 子集重复、分组、漂移、扰动、第三方结果对照等，见[推理测评说明](RECAP_INFERENCE_EVAL.md) |
| `environment/`、`run_config.json`、`training_data.json`、`library.json`、`lists/` | 环境、代码与数据指纹、动作清单 |
| `status.json`、`run.log`、`logs/` | 阶段状态、完整日志和各子进程日志 |

打包整个输出目录会包含所有权重及视频：

```bash
tar -czf logs/repro_train_40g_all_results.tar.gz -C logs repro_train_40g
```

紧凑库和原始数据放在 `data/`，不在这个结果包内；上次 e8 推理结果仍在 `logs/repro_e8_auto/`。

**中断、续跑与常用选项**

训练中断后，确认旧进程已经退出，再重跑原命令。脚本恢复本次正式训练最近的有效检查点，跳过已完成的试训和测评；回放缓冲区重新预热，不能保证与从未中断的运行逐位一致。断点之间尚未保存的进度需要重跑。已有权重不会被当成新训练覆盖。

| 参数 | 用途 |
|---|---|
| `--smoke-only` | 只完成数据准备和 100 次试训；去掉该参数后开始正式从零训练 |
| `--manifest PATH` | 明确指定本次训练和测评共用的过滤清单 |
| `--out DIR` | 新实验使用新目录；更改参数或代码时也应换目录 |
| `--library PATH` | 指定紧凑库；已有文件须有脚本生成的数据身份记录，否则换新路径 |
| `--num-envs N --policy-batch N --wm-batch N` | 明确改变训练规模；不会在显存不足时自动改参数 |
| `--iterations N` | 改训练总迭代数；同时改变余弦学习率时间尺度，报告须注明 |
| `--skip-full` | 只跑最终子集测评，不能作为全量失败数结果 |
| `--no-render` | 跳过视频；同配置重跑时去掉此选项可补视频 |
| `--help` | 查看全部选项 |

核心阶段失败会退出并保存部分报告；可选的视频、扰动或参考结果对照失败会标记 `completed_with_warnings`。同配置重跑只恢复未完成的阶段。以 `status.json` 的最新状态为准，追加日志可能含历史错误。

**报告中的对比口径**

| 本次可测指标 | 对照 |
|---|---|
| BONES 全量失败数/失败率、按动作分组失败数 | 论文表 A.8：25/129,514（单 H100 × 3 h），0/129,514（双 H100 × 9.8 h） |
| 全量平均归一化回报、步数加权回报 | 第三方 e8 发布结果及上次测评；该 e8 权重全量为 145/129,491 失败、平均约 0.932 |
| 世界模型 loss、在线 1/8 步归一化误差 | 本次日志反映世界模型学习过程；在线缓冲区误差不能冒充独立固定测试集结果 |
| 检查点失败率/回报曲线 | 相同的 2,000 动作固定子集；第 0 步是随机初始化基线 |
| 总耗时、循环耗时、显存 | 本次共享单 H100 的实测条件；累计正式训练耗时排除数据打包、独立试训和后续评测 |

这是第三方仓库的训练复现。单卡、策略批量及过滤集合与论文或第三方四卡 e8 不完全相同；基准回报按实际执行步数归一化。32 万次训练不保证达到论文数值或训练时间。BONES 测试衡量训练动作覆盖，不是未见动作泛化，也不包含真机实验。

本地可验证脚本流程、断点恢复和 CLI 参数：

```bash
env -u PYTHONPATH uv run --locked python -m unittest discover -s tests -p 'test_recap_*.py' -v
```

这些测试用 CPU 模拟 GPU 子进程输出。实际 CUDA 训练、显存和 MP4 渲染仍需在服务器运行验证。
