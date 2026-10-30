# Wan2.2 深蹲次数与停止控制：提示对照

上一轮六段动作筛查的两段深蹲都可确认完整周期。本轮只测试深蹲：完整做 1 次或 2 次，
随后停止，并在最后 1 秒保持站立。这是配对推理实验，不含训练，也不是新控制算法的实现。

## 固定配置

- 2 个目标次数 × 3 个新种子（2026、3407、7919）× 2 种提示，共 **12 段 / 6 对**。
- `count_prompt`：直接要求完整次数、做完停止。
- `phase_prompt`：把下降、起身、重复、停止写成动作阶段；仍是整段视频的一条文本。
  没有 Prompt Relay、时间 attention mask、逐帧条件或 LoRA。
- 两组共享人物、全身取景、初始站立和最后一秒停止指令。每对使用相同的初始 latent；
  生成前检查已完成另一侧的噪声哈希，报告再次检查。六对的提示先后顺序各占一半。
- 沿用 Wan2.2-TI2V-5B、1280×704、121 帧、24 fps、50 步、CFG=5、flow shift=5。
  模型、VAE 分块修复、BF16 transformer、FP32 VAE、CPU offload 都复用上一轮。
- `smoke` 为同尺寸、两种提示各一段、每段 2 步，仅验证运行与显存，不计入正式结果。

数据、缓存、日志和输出：`/data/zjr/video_count_wan22_squat_count_20261028`。
复用权重：`/data/zjr/models/Wan2.2-TI2V-5B-Diffusers`，无需再次下载。
原能力筛查的代码、计划、视频和标注保持不变；新 runner 将依赖源码纳入独立身份校验。

## 运行

在 `/home/zjr/Tracker` 下执行准备和短检查，无需占用 GPU：

```bash
bash video_count_probe/wan22_squat_count/run.sh prepare &&
bash video_count_probe/wan22_squat_count/run.sh test &&
bash video_count_probe/wan22_squat_count/run.sh check
```

新提示需单独编码，共 5 份文本（4 条正向提示 + 共用负向提示），只在 CPU 加载 T5：

```bash
bash video_count_probe/wan22_squat_count/run.sh encode
```

选当前有足够空闲显存的一张卡；下面的 7 是示例，不是实时空闲卡承诺：

```bash
CUDA_VISIBLE_DEVICES=7 bash video_count_probe/wan22_squat_count/run.sh smoke \
    --shared-gpu-memory-mib 17408 &&
CUDA_VISIBLE_DEVICES=7 bash video_count_probe/wan22_squat_count/run.sh pilot \
    --shared-gpu-memory-mib 17408
```

共享预算仍为本进程 PyTorch allocator 的 17 GiB，启动要求至少 19 GiB 空闲，并在加载和每段前复查。
CUDA context / 库分配额外占用显存。沿用上一轮实际 allocated 约 14.56 GiB、reserved 约 16.97 GiB
作为参考，新提示的真实峰值由 smoke 确认。若独占空卡，可省略共享参数，默认要求基本空闲的 3090。

上一轮正式视频每段约 28–31 分钟；按相近共享负载估算 12 段约 **6 小时**，另加编码、加载和 smoke。
这不是保证耗时。不要同时对同一输出目录启动两份生成。

查看进度：

```bash
bash video_count_probe/wan22_squat_count/run.sh status
```

中断后重跑原生成命令，完成的视频校验后跳过；SIGINT/SIGTERM 打断的条目允许直接重试。
OOM 等真正错误需先排查，再加 `--retry-failures`。`--limit 2` 可只处理接下来两段，不改变计划；
数量上限按剩余视频计算，不保证在存在部分完成结果时恰好是一整对。
模型、环境、生成依赖代码或已冻结计划发生改变时会拒绝混用结果，需新 `--root`。

## 复核与评分

生成后自动保存 `review.html`、`review_frames/blind_manifest.json` 和 `annotations_template.json`。
复核页面隐藏次数、方法和种子，先按实际运动标出周期，再解盲计算。标注者若已看过映射，应披露，
不能宣称独立盲评。可以由 Codex 复核，并设置 `rater_type=ai`；后续人工复核保存另一份文件。

复制模板为 `pilot/annotations.json`，填写 `rater` 和 `rater_type`。每段：

- `reviewed`：是否已查看；`count_judgeable`：能否判断完整次数。
- `complete_cycles`：所有完整周期的列表，每个包含 `start_frame`、`bottom_frame`、`end_frame`。
  例如 `[{"start_frame":0,"bottom_frame":24,"end_frame":48}]`；明确为零次用 `[]`，看不清用 `null`。
- `partial_attempts`：未完成动作列表，每项含 `start_frame`、`end_frame`，没有用 `[]`，不确定用 `null`。
  不把同一完整深蹲内的局部起伏重复标成额外动作。完整周期和未完成动作区间不能重叠。
- `initial_upright`、`final_second_upright_still`：布尔值或 `null`；最后一秒对应帧 96..120。
  正常呼吸、轻微手部动作不算停止失败，新的下蹲、跳起、迈步算失败。
- `notes`：填写帧号或时间证据；所有帧索引在 0..120，视频帧率是 24 fps。

```bash
bash video_count_probe/wan22_squat_count/run.sh report
# 或显式选择另一份独立标注，不合并不同评审者
bash video_count_probe/wan22_squat_count/run.sh report --annotations /data/zjr/<project>/reviewer2.json
```

分别报告完整次数、达到目标后停止、最后一秒站立，以及三者兼顾且第 96 帧前完成的联合成功。
“恰好一次完整深蹲，随后又蹲了半次”可通过次数项，但不通过停止/联合项。
开头的未完成动作、初始姿态另行报告，不事后加进主指标。

`review_complete` 表示所有视频看过；`assessment_complete` 表示主要判断也都有确定值。
保留无法判断的 `null`，不当作失败或成功。成功上下界只表示未决标注/缺失结果的范围，不是置信区间。
报告包含方法及次数分组、配对胜负/平局、显存峰值、标注来源和哈希。
六对结果只用于选择下一步，不据此宣称稳定提升或论文创新成立。
