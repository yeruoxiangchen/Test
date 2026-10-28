# 动作次数与节奏可控的视频生成：初步对照实验

这是独立的论文选题实验。先测清楚同一视频基座在计数、分时段文字提示、时间路由下的行为，再决定适配训练方案。
**目前不含训练后的新方法；Prompt Relay 是已有方法的移植对照。生成完成不等于实验指标已经完成。**

## 固定实验

- 基座：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`，首次下载锁定仓库 revision 并校验 SHA-256。
- 动作：拍手、开合跳、深蹲；每种动作各 1/2/3 次，另有 3 次且第 2 次后暂停。
- 共 12 个条件 × 2 个种子（42、123）× 4 个方法 = **96 段视频**。
- 正式 pilot：832×480、81 帧、16 fps、50 步 UniPC、CFG 6、flow shift 8。
- smoke：416×240、81 帧、4 步、一个条件、四个方法，**只验证运行链路，不能评价生成能力**。
- 同条件、同种子下初始噪声相同，记录噪声 SHA-256。方法运行顺序固定打乱。
- 不动态筛掉困难样本；失败、不可辨认、歧义分别记录。

| 方法 | 内容 |
|---|---|
| `count_prompt` | 直接说明完整动作定义及要求次数，暂停条件也写入提示 |
| `timed_prompt` | 同一要求，展开为带秒数的动作时间段 |
| `relay_unrouted` | 使用 Relay 完整提示，关闭时间路由 |
| `relay` | 与上一组文字完全相同，在条件分支的交叉注意力中施加时间惩罚 |

关键因果对照是 `relay` 对 `relay_unrouted`，不能把更长文字提示的收益都归于路由。
普通计数提示的时间成功率只作为诊断；它没有接收精确秒数，不能用时间指标作为公平主比较。

## 已有方法与移植边界

参考 [Prompt Relay](https://arxiv.org/abs/2604.10030)，核对了作者公开实现：
[Wan2.2 commit 4823394](https://github.com/GordonChen19/Wan2.2/tree/482339441717aee0b1cb97a314c1e733557d1536)。
采用其 `relu(abs(t-midpoint)-window)^2/(2*sigma^2)` 惩罚，`window=segment_length//2-2`、`sigma=0.1448`。
全局文本不受时间惩罚。沿用一个完整文本的 T5 编码；局部片段按 fast tokenizer 的字符 offset 定位，避免重复动作文本被错误映射到第一次出现。
只在 conditional CFG 分支路由，unconditional 分支保持原样。按 query 分块执行 PyTorch SDPA，避免构造所有头的完整注意力图。

这是 **Diffusers 0.35.1 / Wan2.1-1.3B 移植版本**，不是作者 Wan2.2-A14B 的原配置复现。短动作段的控制可能仍弱，这是要观察的现象。

[TempoControl 官方 README](https://github.com/Shira-Schiber/TempoControl)注明约 110GB 单卡显存。
本轮没有运行或声称复现 TempoControl。之后若继续课题，需要补充合适的强基线与更新基座。

## 文件位置

- 代码及小型隔离环境：`/home/zjr/Tracker/video_count_probe/`。
- 权重：`/data/zjr/models/Wan2.1-T2V-1.3B-Diffusers/`。
- 所有长实验输出、文本缓存、日志：`/data/zjr/video_count_probe_20261025/`。
- `artifacts/plan_preview/` 只有可审查的实验计划，不包含权重或实验视频。
- `.venv` 只读继承已有 `hunyuan3d2mv` 环境中的 torch，额外依赖安装到本项目，不修改原 Conda 环境。

## 执行

在 `/home/zjr/Tracker` 下运行。已有环境时无需重复 setup。

```bash
bash video_count_probe/run.sh setup
bash video_count_probe/run.sh test
bash video_count_probe/run.sh prepare
bash video_count_probe/run.sh download
bash video_count_probe/run.sh check
```

`download` 会下载大模型并校验，耗时取决于网络和磁盘。中断后重新执行可续传；已有 revision 不会自动漂移。生成阶段完全离线，不会隐式下载。

下载按文件串行续传，每个文件遇到连接中断、超时或临时服务错误会最多尝试 5 次；读取超时默认 60 秒。
已有完成文件和 `.incomplete` 分片会保留，不使用 `force_download=True`。
下载/环境检查也会保存日志到输出根目录的 `logs/`，完整异常会写入日志；
模型目录的 `download_status.json` 记录当前文件、尝试次数和最近错误。
再次中断时，把 `download_status.json` 或最新 `download_*.log` 的末尾提供给我即可。

本机已锁定 revision `0fad780a534b6463e45facd96134c9f345acfa5b`，完整文件约 **28.93GB**。
2026-10-25 已完成约 21MB 的配置/分词器下载和 36 条提示的实际分词验证（最长 306/512 tokens）；大权重尚需执行上面的 `download`。

先选择一张空闲 GPU，再运行短检查。下面的 `1` 需要根据实际占用修改：

```bash
CUDA_VISIBLE_DEVICES=1 bash video_count_probe/run.sh smoke
CUDA_VISIBLE_DEVICES=1 bash video_count_probe/run.sh pilot
```

默认要求选定 GPU 至少有 16000MiB 空闲，且无明显占用。显式 `--allow-busy` 可允许共享，但仍检查空闲显存。
脚本不停止其他进程、不自行等待空卡、不自动启动多个 worker。不同输出目录也有每 GPU 锁，避免 smoke/pilot 重叠。
单卡运行 96 段视频属于长任务；可以 `--limit 4` 先生成四个配对方法，之后原命令续跑。
smoke 的四步视频可能质量很差，不能据此否定研究方向。

运行中会显示 log 路径。中断后重跑复用校验通过的完整视频；如留下错误记录，检查错误后加 `--retry-failures`。
改变代码、模型或依赖版本后，需要使用新 `--root`，避免结果混合。

## 核验及指标

生成结束自动输出 `pilot/review.html`。将该文件和同级 `videos/` 一起复制到可打开浏览器的电脑；也可自行通过本地文件服务打开。
页面打乱顺序并隐藏方法、目标次数和提示，仅显示动作计数定义。逐段记录完整动作的完成时刻、可辨认性、动作是否出现、计数歧义，以及第 2/3 次之间是否明显停顿。
**标注按钮中的“完成一次”是完整周期结束，不是下蹲最低点、开合跳张开时刻或手掌刚开始靠近。**

人工核验后导出 `annotations.json`，放回实验目录：

```bash
bash video_count_probe/run.sh report \
  --annotations /data/zjr/video_count_probe_20261025/pilot/annotations.json
```

主要看：

- 可辨认且完整次数正确的比例，分母包含坏视频；避免通过剔除坏视频虚增成功率。
- 可辨认性、动作出现率、歧义率；在可核验视频上的计数 MAE。
- 完整动作结束时刻是否落入要求区间（容差 0.25s），以及暂停是否被遵守。
- 按动作、目标次数、暂停条件分组；同条件同种子的成对胜/负/平。

`metrics_ready=True` 需要 pilot 的 96 段视频和对应完整人工标注全部存在。
缺视频、运行报错、缺标注分别统计，不会混为模型质量失败。部分数据也可查看描述性结果，但不能冒充完整实验。
这里不设置“提升低于某个数就结束课题”的自动门槛，不从小 pilot 给出显著性或可发表承诺。

## 本地测试的范围

测试覆盖计划配对、时间和文本 token 映射、重复进程锁、结果身份、标注校验，以及真实 Diffusers Wan 层与采样循环的 conditional/unconditional 隔离、零惩罚等价和异常后恢复。
这些测试使用小型随机初始化模型，不下载大权重、不占用 GPU，**不提供真实生成质量结果**。

## 2026-10-27：无次数要求的动作能力对照

新入口为 `bash video_count_probe/action_calibration/run.sh`，默认输出位于
`/data/zjr/video_count_action_calibration_20261027`。沿用原模型、负面提示、50 步采样、分辨率和种子；
不修改原生成源码指纹，不重跑原有 96 段。

正式运行只新增 12 段：3 动作 × 2 种子 × 2 提示。
`short_action` 使用极简动作提示；`uncounted_matched` 只删除原 `count_prompt` 中的 `exactly N`。
后者与原目标 1/2/3 的提示均形成相同的无次数提示，因此只有 6 个动作/种子分析单位，
不能把一个新视频与三个旧目标的比较当作三个独立样本。极简提示同时改变措辞和描述细节，
其收益不能单独归因于长度。

```bash
bash video_count_probe/action_calibration/run.sh prepare
bash video_count_probe/action_calibration/run.sh test
bash video_count_probe/action_calibration/run.sh check
CUDA_VISIBLE_DEVICES=1 bash video_count_probe/action_calibration/run.sh smoke
CUDA_VISIBLE_DEVICES=1 bash video_count_probe/action_calibration/run.sh pilot
```

优先选择空闲 GPU。确认资源余量后，可给 smoke/pilot 加 `--allow-busy` 允许共用，
仍要求启动时至少有 16000 MiB 空闲；每次生成记录共用状态、生成阶段峰值和包含文本编码的进程峰值。
正式生成约 2 小时（按上一轮单段中位用时估计）；共用 GPU 时会更慢。
已完成视频校验后跳过；`--limit 2` 可限制此次新增数量，原命令可续跑。
发生错误后先检查日志，再用 `--retry-failures` 重试。输出与 GPU 均有独占锁，不需要手动删除锁文件。
运行中可执行 `bash video_count_probe/action_calibration/run.sh status` 查看生成进度。

本轮主要判断“是否至少出现一个可辨认的完整动作周期”，不要求次数恰好正确。
精确次数有歧义但明确存在完整动作时，可以肯定动作能力；半次动作不能当作完整动作。
生成结束自动导出 `pilot/review_frames/blind_manifest.json` 和帧图，Codex 可继续完成 AI 评审，
不要求用户先人工标完。`review.html` 保留可选人工复核入口。AI 标注来源与人工标注必须明确区分。
旧的 18 段直接次数提示作为描述性参考，沿用明确标为 AI 的旧标注。
`smoke` 只有一段低分辨率四步视频，仅检查执行链路，不参与动作质量结论。
