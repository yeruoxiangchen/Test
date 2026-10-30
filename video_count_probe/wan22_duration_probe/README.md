# Wan2.2 两次深蹲：固定时刻与视频时长对照

这一轮新增 **6 段正式视频**，复用原实验的 3 段 A 组结果。只做推理诊断。

| 组别 | 帧数 / 时间轴 | 提示 | 来源 |
|---|---|---|---|
| A (`a5`) | 121 帧 / 0–5 秒 | 前 4 秒完成两次，末秒保持直立 | 原 `wan22_squat_count` 的 `squat_n2 / phase_prompt`，3 段 |
| B (`b5`) | 121 帧 / 0–5 秒 | 完整做两次后停止，取消固定秒数 | 新生成 3 段 |
| C (`c8`) | 193 帧 / 0–8 秒 | 与 B 完全相同的文本 | 新生成 3 段 |

种子固定为 2026、3407、7919。相邻帧间隔 1/24 秒；容器播放总时长可能比最后一帧时间戳多 1/24 秒。
沿用 Wan2.2-TI2V-5B、1280×704、24 fps、50 步、CFG=5、flow shift=5，模型与环境必须匹配旧实验。
B/C 共享同一份正向文本嵌入。保留原来两次完整下蹲/起身的描述，只去掉固定时钟指令。

A/B 在相同形状、相同种子下，**逐种子检查初始噪声 SHA-256 完全相同**。
B/C 的噪声形状分别为 `[1,48,31,44,80]`、`[1,48,49,44,80]`，只称相同种子，不能称同一噪声。
六段按种子交替 B/C 顺序；3 个种子无法完全平衡先后，时间和显存只作资源记录。

输出、缓存、CPU 测试临时文件和日志都在：

`/data/zjr/video_count_wan22_duration_20261029`

复用已有权重 `/data/zjr/models/Wan2.2-TI2V-5B-Diffusers`，无需下载。
依赖旧实验 `/data/zjr/video_count_wan22_squat_count_20261028/pilot` 的计划、3 个视频、记录与标注。
引用文件被校验并冻结；保留旧目录。新代码不会修改旧实验的源码、计划、视频或标注。

## 已有环境中的准备与运行

在 `/home/zjr/Tracker` 下执行。`run.sh` 默认使用 `video_count_probe/.venv/bin/python`；
如需显式指定解释器，设置 `VIDEO_COUNT_PYTHON`。不修改现有 Conda 环境。

```bash
bash video_count_probe/wan22_duration_probe/run.sh prepare &&
bash video_count_probe/wan22_duration_probe/run.sh test &&
bash video_count_probe/wan22_duration_probe/run.sh check
```

这些检查只在 CPU 上执行。`check` 会直接重建真实形状的噪声并验证 A/B 配对。
新正向提示需要编码；下面只在 CPU 加载 T5，并缓存正向/负向共 2 份文本：

```bash
bash video_count_probe/wan22_duration_probe/run.sh encode
```

优先选一张空闲的 3090。默认启动检查要求至少 22500 MiB 空闲。
下面的 GPU 7 仅为示例，需要替换为当时空闲的卡：

```bash
CUDA_VISIBLE_DEVICES=7 bash video_count_probe/wan22_duration_probe/run.sh smoke &&
CUDA_VISIBLE_DEVICES=7 bash video_count_probe/wan22_duration_probe/run.sh pilot
```

`smoke` 各生成一段 121 帧和 193 帧的视频，均为全分辨率、2 步，用于实际推理、VAE 解码和显存检查。
`pilot` 必须有对应长度、相同代码/模型/环境的成功 smoke 记录，才会运行该组的 50 步生成。
两步峰值仍不能保证五十步绝对不会 OOM；成功 smoke 不计入质量评价。

**8 秒的 GPU 峰值尚未实测。** 不能沿用旧实验 17 GiB 的经验直接承诺可共享运行。
显式共享时，smoke 和 pilot 都传相同预算，例如：

```bash
CUDA_VISIBLE_DEVICES=7 bash video_count_probe/wan22_duration_probe/run.sh smoke \
    --shared-gpu-memory-mib 17408 &&
CUDA_VISIBLE_DEVICES=7 bash video_count_probe/wan22_duration_probe/run.sh pilot \
    --shared-gpu-memory-mib 17408
```

这只是设置预算的示例，不是 8 秒一定能在 17 GiB 内完成的保证。启动和每段生成前要求预算之外至少
2048 MiB 空间；PyTorch 预算不包含全部 CUDA/库分配。若预算低于已完成 smoke 的 reserved 峰值，pilot 会拒绝。
运行器不改变分辨率、步数或帧数来规避 OOM，也不终止其他任务。

如果 C 组 smoke 因显存失败，换到空闲卡后可只补 C，保留已成功的 B：

```bash
CUDA_VISIBLE_DEVICES=7 bash video_count_probe/wan22_duration_probe/run.sh smoke \
    --condition c8 --retry-failures
```

也可以先单独完成 B 的正式结果：

```bash
CUDA_VISIBLE_DEVICES=7 bash video_count_probe/wan22_duration_probe/run.sh pilot --condition b5
```

每个输出目录只有一个 writer，不能同时启动两份 pilot，即使选择了不同组/不同 GPU。
重新运行原命令会校验并跳过已完成视频；SIGINT/SIGTERM 的中断自动允许重试。
OOM 等实际错误需处理后添加 `--retry-failures`。`--limit 1` 只生成下一条待完成记录。
计划、生成代码、模型或环境变化会拒绝混合结果，此时使用新 `--root`。
输出根目录可用 `WAN22_DURATION_ROOT` 或 `--root` 指定，必须位于 `/data/zjr`。

```bash
bash video_count_probe/wan22_duration_probe/run.sh status
bash video_count_probe/wan22_duration_probe/run.sh status --mode smoke
```

`generated_new=6/6 reused_A=3` 表示正式视频齐全。`assessment_complete=false` 在未标注时是预期状态。
生成速度取决于卡上负载；旧 5 秒样本约 19–30 分钟/段，8 秒耗时待 smoke/首段实测，不能承诺线性增长。

## 复核与评分

生成后自动生成 `pilot/review.html`、`review_frames/blind_manifest.json`、`annotations_template.json`。
页面只展示新增 B/C，A 的原标注单独保留来源。条件/种子名被隐藏，但时长可见，不能宣称对时长完全盲评。

复制模板为 `pilot/annotations.json`，填写 `rater`、`rater_type=ai/human`。每段填写：

- `reviewed`、`count_judgeable`。
- `complete_cycles`：完整动作列表，每项为 `start_frame`、`bottom_frame`、`end_frame`。
  确认零次用 `[]`，无法判断用 `null`；起始处已经蹲下不补算完整一次。
- `partial_attempts`：残缺动作列表，每项为 `start_frame`、`end_frame`；没有用 `[]`，不确定用 `null`。
  与完整周期不得重叠，低位停顿/小幅起伏不重复算一次。
- `initial_upright`、`final_second_upright_still`：布尔值或 `null`。
- `notes`：记录可见帧证据，不确定性也写明。

B 的帧范围为 0..120，末秒为 96..120；C 为 0..192，末秒为 168..192。
所有帧号除以 24 得到视频时间。

```bash
bash video_count_probe/wan22_duration_probe/run.sh report
# 可单独选择另一位评审，不合并不同评审者：
bash video_count_probe/wan22_duration_probe/run.sh report --annotations /data/zjr/<project>/reviewer2.json
```

主指标为 `exact_count`（完整两次）和 `count_and_stop`（完整两次，且达到目标后没有追加完整/残缺动作）。
`last_second_rest`、是否初始直立、残缺动作、目标完成时间分别报告。
**B/C 不要求固定时刻完成，因此不会因最后一秒仍在完成第二次动作而把次数主指标判失败。**
本轮 A/B/C 统一按以上指标重新汇总，原实验的严格“4秒前完成+末秒静止”指标保留在原报告。

报告区分新生成、新标注、复用参考、失败、主动中断和未知标签。旧参考的 3 条标注不会被计为新视频已评估。
AI 标注只作为初步诊断，保留来源与不确定性，不据 3 个种子估计稳定总体提升。
若时长增加后出现两次，仅说明该配置能生成更多完整周期，还需要验证它是否真正响应次数指令。
