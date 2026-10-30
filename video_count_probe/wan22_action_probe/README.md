# Wan2.2 六段动作能力筛查

目的：先确认更强的视频基座能否生成完整动作，再决定是否继续研究动作次数和停止控制。
这是选题的可行性检查，不是论文方法、训练实验或最终效果结论。

固定拍手、开合跳、深蹲三类动作，各 seed=42/123，共 6 段正式视频。
沿用上一轮 `short_action` 的三条文本，不要求次数，不改写提示词，不提供首帧。
采用 Wan2.2-TI2V-5B-Diffusers 的 704×1280、121 帧、24 fps、50 步、CFG=5、flow shift=5。
smoke 是同样尺寸的一段拍手视频，但仅采样 2 步；只检查执行与显存，不能评价质量。

新旧实验的 VAE、分辨率、帧率、CFG、negative prompt 等同时改变。这轮只做基座能力筛选；
同 seed 不代表相同初始噪声，差异不能全部归因于参数量。

## 已完成的准备

- 使用现有 `video_count_probe/.venv`，Diffusers 0.35.1，不升级或修改旧环境。
- 官方模型版本已固定为 `b8fff7315c768468a5333511427288870b2e9635`，配置与分词器已下载校验。
- 完整下载清单约 34.20 GB。2026-10-28 用户已完成权重下载校验及四份 CPU 文本缓存；真实 5B 推理待运行。
- CPU 测试覆盖实验配对、哈希损坏、互斥锁、续跑、标注来源、24 fps 抽帧、共享显存预算与余量变化、
  随机小型 Wan 的实际 CFG/扩展时间步和 residual patch VAE 解码。
- 新建局部 `vae_compat.py`：修复 0.35.1 分块解码遗漏的首帧处理和 unpatchify，
  并保持其普通解码的 clamp 语义。单块结果与本机普通解码一致，多块尺寸和帧数通过检查。
  这项修复不改 site-packages；真实模型效果和峰值仍待 smoke 检验。

## 目录

- 数据、文本缓存、临时文件、日志、结果：`/data/zjr/video_count_wan22_action_20261028`
- 模型：`/data/zjr/models/Wan2.2-TI2V-5B-Diffusers`
- 参考清单：`/data/zjr/video_count_action_calibration_20261027/pilot/plan.json`

所有新缓存变量均由 CLI 设置到本项目的 `/data/zjr` 目录。旧实验代码、计划和结果不修改。

## 用户运行的长任务

在 `/home/zjr/Tracker` 执行。前两步无需 GPU；CPU 文本编码可能较慢。

```bash
bash video_count_probe/wan22_action_probe/run.sh download &&
bash video_count_probe/wan22_action_probe/run.sh encode
```

下载按已固定 revision 续传，并校验权重 SHA-256。重新运行 `download` 不会更新模型版本。
网络中断时重跑原命令即可；不要同时启动两个下载器。

选一张空闲 3090，把下方 `1` 改为该卡编号，再运行：

```bash
export CUDA_VISIBLE_DEVICES=1
bash video_count_probe/wan22_action_probe/run.sh check &&
bash video_count_probe/wan22_action_probe/run.sh smoke &&
bash video_count_probe/wan22_action_probe/run.sh pilot
```

启动前要求至少 22500 MiB 空闲、占用不超过 1000 MiB、利用率不超过 10%。
这个门槛用于避开其他任务，不代表实测显存峰值。使用 BF16 transformer、FP32 VAE 分块解码和
model CPU offload；不量化。T5 在单独的 CPU 进程中预先编码，生成时不加载 T5。

如果需要与现有任务共用一张卡，可以显式给出本进程的 PyTorch 显存预算，例如 17408 MiB（17 GiB）：

```bash
CUDA_VISIBLE_DEVICES=6 bash video_count_probe/wan22_action_probe/run.sh smoke --shared-gpu-memory-mib 17408
```

此时允许 GPU 已在使用，但启动时要求至少 `17408 + 2048 = 19456 MiB` 空闲，加载模型和每段视频前再查余量。
上限通过本进程的 PyTorch caching allocator 实施；CUDA context、库分配不包含在这个上限里。
它不会预留整卡或限制其他进程，也不能保证其他任务未来显存增长时仍有足够空间。
共享会争用算力，耗时不能用于独占 GPU 的速度比较。是否能在 17 GiB 预算内跑完仍需 smoke 实测。
先单独运行 smoke，成功并查看显存记录后再用相同参数运行 pilot；不要为了绕过 OOM 直接取消上限。
GPU 编号仅是示例，应按运行时的占用选择。

日志打印每段生成时间、PyTorch peak allocated/reserved（不等于整卡 nvidia-smi 峰值）。
如果 smoke OOM 或解码失败，`&&` 会停止，pilot 不会开始；保留错误文件供排查，勿盲目重试。

不改代码、配置和模型的情况下可重跑原命令继续未完成视频；完成的视频先校验后跳过。
失败任务需要排查后加 `--retry-failures`。如果生成实现发生变化，应使用新 `--root`，
不要把不同版本结果混在一个实验目录里。

## 完成之后

`smoke`/`pilot` 自动生成 `review.html`、`review_frames/blind_manifest.json` 和抽帧图。
每隔 3 帧抽一帧，即 8 fps，时间标记按真实 24 fps 计算。
可以交给 Codex 看抽帧，接触动作不明确时仍需复查原视频；AI 标注须保留来源，不能称作人工真值。
生成成功并不等于动作完整。

```bash
bash video_count_probe/wan22_action_probe/run.sh status
```

`status` 只读报告内容，不覆盖已有标注；进度日志仍会写入 `logs/`。
将 `annotations_template.json` 填写后另存为 `pilot/annotations.json`，指定 `rater`、`rater_type`，
每段填 `complete_cycle=true/false` 和视觉证据 `notes`；无法判断则保持 `null`。
`report` 自动读取该文件，显式 `--annotations` 可改用另一个来源，来源不会合并：

```bash
bash video_count_probe/wan22_action_probe/run.sh report
```

先按三类动作分别看是否有清楚、完整的周期；六段只足以指导下一轮实验选择。
是否增加次数控制实验由复核结果决定，脚本不会自动扩展到 18 段或启动训练。

官方模型与原生配置：
[模型仓库](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)、
[Wan2.2 TI2V-5B 配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)。
