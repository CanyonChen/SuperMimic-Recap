**双 H100：从零训练并自动测评**

在已经跑通推理的服务器环境中执行。`--gpus 0,1` 启动两个同步训练进程，共同训练同一套世界模型、策略和 Critic；每卡独立仿真和采样，更新时平均梯度。脚本复用现有 50 FPS NPZ 数据和已完成的紧凑动作库，训练结束后自动测评并生成案例 MP4。

```bash
cd /home/jovyan/canyon/SuperMimic-Recap
git pull --ff-only origin main
mkdir -p logs
nohup bash scripts/recap_training_eval.sh --gpus 0,1 \
  > logs/recap_training_2xh100_launcher.log 2>&1 < /dev/null &
tail -f logs/recap_training_2xh100_launcher.log
```

`nohup` 让训练在 SSH 断开后继续。上面的 `tail -f` 仅查看日志，按 Ctrl+C 退出查看不会停止后台训练。不要同时启动两份命令写入同一结果目录。

双卡默认使用新目录 `logs/repro_train_2xh100/`，从随机初始化开始；原来 `logs/repro_train_40g/` 的试训、错误记录和权重保留。紧凑库仍沿用原路径 `data/library/bones_seed_train_40g.pt`，其中的数据与卡数无关。升级代码后不要把新双卡参数写到旧单卡结果目录。

双卡通过 `python -m torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0` 启动。每个进程在导入 torch、MuJoCo Warp 前绑定自己的 GPU，再通过 NCCL 同步梯度；某个进程失败时整组退出，重跑本脚本从已保存的检查点恢复。[PyTorch 启动器说明](https://docs.pytorch.org/docs/2.14/elastic/quickstart.html)

**自动执行顺序**

1. 检查 GPU 0 和 GPU 1，记录环境、代码版本和参数。每次启动试训或正式训练前，两卡各需至少有 72 GiB 空闲。
2. 校验 `data/npz/bones_seed/manifest.csv` 和 50 FPS 数据。优先复用 `logs/repro_e8_auto/manifest_filtered.csv`；不存在时尝试 NPZ 目录中的过滤清单，仍不存在则自动过滤。数据不会重新解压或转换。
3. 复用已经校验通过的 FP16、71 维紧凑动作库 `data/library/bones_seed_train_40g.pt`；文件不存在时才构建。每卡加载一份。训练与测评共用同一份过滤清单。
4. 用正式训练的环境数和批量试训 100 次迭代，验证编译、参数更新、权重保存和显存。通过后重新随机初始化，开始正式训练。
5. 正式训练 320,000 次迭代，每 4,000 次保存检查点，每 50 次写日志和显存采样。
6. 训练结束后，对第 0、40k、80k、100k、120k、160k、200k、240k、280k 次迭代的权重做同一组 2,000 动作子集测评，生成性能随迭代变化的数据。
7. 对最终权重执行原有推理流程：单动作、固定 2,000 动作子集重复 3 次、全部过滤动作分片测评、失败诊断、扰动测试、第三方结果对照及案例视频。最终子集结果也加入检查点曲线。

检查点测评和最终测评均在双卡训练退出后，使用第一张选定 GPU 顺序执行。视频默认开启，文件数由案例筛选结果决定，不固定为上次的 9 个。

数据审计现在会先打印目录，再每检查 10,000 个文件输出一次进度；子进程启动时会打印 PID。worker 会分别打印 `[rank 0/2]`、`[rank 1/2]` 及对应的物理 GPU，随后进入初始化和训练。

**默认训练配置**

| 项目 | 双卡 `--gpus 0,1` | 单卡 `--gpu 1` |
|---|---|---|
| GPU | 物理 GPU 0、1；各进程内均为 `cuda:0` | 物理 GPU 1；进程内为 `cuda:0` |
| 每卡显存预算 | 72 GiB | 40 GiB |
| 每卡并行仿真环境 / 总环境 | 4,096 / 8,192 | 4,096 / 4,096 |
| 每卡策略批量 / 总策略批量 | 4,096 / 8,192 | 2,048 / 2,048 |
| 每卡世界模型批量 | 2,048，10 批/次更新 | 相同 |
| 学习率缩放 | 仓库单卡基值 × sqrt(2) | × 1 |
| 回放容量 | 每个环境 1,000 步 | 相同 |
| 想象长度 / 世界模型训练窗口 | H=32 / k=8 | 相同 |
| UTD | 0.4 | 相同 |
| 正式训练 / 独立试训 | 320,000 / 100 次迭代 | 相同 |
| 随机种子 | rank 0 为 0，rank 1 为 1000 | 0 |
| 测评并行环境 / 全量分片大小 | 256 / 2,000 动作 | 相同 |
| 默认结果目录 | `logs/repro_train_2xh100/` | `logs/repro_train_40g/` |

一次双卡训练迭代包含总共 8,192 个环境的一次控制步。模型结构、噪声、随机化、Critic 和学习率调度沿用仓库配置；完整配置保存在 `train/config.json`。双卡增大每次迭代的采样和更新批量，不承诺相同迭代数的墙钟时间减半。稳定训练速度以日志 `it/s` 为准，总仿真转移吞吐约为 `it/s × 8192`，还需结合回报曲线评价。

双卡模式中，每卡 PyTorch 缓存分配器额度为 68 GiB，另留 4 GiB 给 Warp、CUDA/NCCL 上下文，余下显存留作余量。单卡模式仍为 36+4 GiB。额度是上限，实际分配量取决于环境和批量；每个 rank 都在日志点采样显存，发现超过预算就停止。容器 PID 不可见时，只能记录 PyTorch 峰值，报告会说明进程采样缺失。实际 H100 占用以双卡试训为准。[分配器额度的官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.set_per_process_memory_fraction.html)

首次打包动作库会占用主机 RAM 和磁盘；构建器会同时保留各动作数组及拼接结果。紧凑库约数 GB，32 万步会保存约 80 份中间权重，整个训练输出可能达到几十 GB。首次 `torch.compile` 编译耗时不代表稳定训练速度。

**双卡输出：`logs/repro_train_2xh100/`**

| 文件或目录 | 内容 |
|---|---|
| `report.md`、`training_summary.json` | 总结、实测失败率/回报、耗时、显存与论文对比边界 |
| `train/ckpt_latest.pt`、`train/ckpt_*.pt` | 正式训练最终及中间权重，包含模型、优化器和采样器状态 |
| `train/config.json`、`train/checkpoint_metadata.json` | 完整训练配置、完成迭代数和更新次数 |
| `train/tb/`、`training_curves.csv`、`training_metrics.json` | TensorBoard 事件、全部标量曲线、各指标最后记录值 |
| `train/memory.csv`、`train/runtime*.json` | rank 0 的显存采样、PyTorch 峰值、每次训练进程耗时和续训记录 |
| `train/rank_1/memory.csv`、`train/rank_1/runtime*.json` | rank 1 对应记录；两卡均完成才算训练阶段通过 |
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
tar -czf logs/repro_train_2xh100_all_results.tar.gz -C logs repro_train_2xh100
```

紧凑库和原始数据放在 `data/`，不在这个结果包内；上次 e8 推理结果仍在 `logs/repro_e8_auto/`。

**中断、续跑与常用选项**

训练中断后，确认旧进程已经退出，再重跑原命令。两卡加载本次正式训练最近的有效检查点，跳过已完成的试训和测评；模型和优化器保持同步，采样器恢复 rank 0 保存的状态，各卡回放缓冲区重新预热。不能保证与从未中断的运行逐位一致，断点之间尚未保存的进度需要重跑。单卡与双卡权重的卡数检查不匹配时拒绝续训。

| 参数 | 用途 |
|---|---|
| `--gpus 0,1` | 双卡同步训练，自动切换每卡 72 GiB、策略批量 4096 和独立输出目录 |
| `--vram-gb N` | 每张卡各自的预算，并在启动阶段检查两卡可用显存 |
| `--smoke-only` | 只完成数据准备和 100 次试训；去掉该参数后开始正式从零训练 |
| `--manifest PATH` | 明确指定本次训练和测评共用的过滤清单 |
| `--out DIR` | 新实验使用新目录；更改参数或代码时也应换目录 |
| `--library PATH` | 指定紧凑库；已有文件须有脚本生成的数据身份记录，否则换新路径 |
| `--num-envs N --policy-batch N --wm-batch N` | 改变每张卡的训练规模；不会在显存不足时自动改参数 |
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
| 总耗时、循环耗时、各卡显存、GPU·h | 记录实际单/双 H100 条件；累计正式训练耗时排除数据打包、独立试训和后续评测。墙钟指标取 rank 0，GPU·h 为各 rank 进程时长之和，含初始化和等待 |

这是第三方仓库的训练复现。双卡环境数和策略批量按论文规模配置，过滤集合、实现及第三方四卡 e8 的训练条件仍有差异；基准回报按实际执行步数归一化。TensorBoard 的世界模型误差等局部指标来自 rank 0。32 万次训练不保证达到论文数值或训练时间。BONES 测试衡量训练动作覆盖，不是未见动作泛化，也不包含真机实验。

本地可验证脚本流程、断点恢复和 CLI 参数：

```bash
env -u PYTHONPATH uv run --locked python -m unittest discover -s tests -p 'test_recap_*.py' -v
```

这些测试用 CPU 模拟 GPU 子进程输出，并核对实际分布式初始化函数的卡绑定、NCCL 参数和每卡日志。实际双 H100/NCCL 训练、加速比、显存和 MP4 渲染仍需在服务器运行验证。
