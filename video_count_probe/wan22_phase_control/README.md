# Wan2.2：两次深蹲后停止的时间路由对照

本实验检验：把动作阶段文本路由到相应视频时间，是否比相同详细文本、不加路由更容易控制次数和停止。它是固定参数的小规模可行性实验，不是训练，也不构成方法创新或稳定提升的证据。

- 模型：已有 Wan2.2-TI2V-5B-Diffusers；无新增下载。
- 正式实验：2 组 × 3 种子（2026、3407、7919），共 6 条；193 帧、24fps、1280×704、50 步、CFG=5、flow shift=5。
- 两组共用完全相同的正负文本缓存和初始噪声。初始噪声还与此前 C8 记录校验 SHA；旧视频不进入本次两组效果比较。
- `text_only` 和 `routed` 均使用相同的分块 SDPA 条件 cross-attention，前者偏置为零。无条件 CFG、self-attention、权重和采样器不变。
- 时间表：0–0.5s 站立；0.5–1.75s 下蹲；1.75–3s 起身；3–4.25s 第二次下蹲；4.25–5.5s 起身；5.5–8s 站立停止。
- 路由：阶段区间内加性偏置为 0，区间外偏置为 `-min(distance_seconds² / (2×0.25²), 8)`；全局、特殊及 padding token 偏置为 0。49 个 latent 时间锚点按 `index×4/24` 秒映射，880 个空间 token 共用该时间偏置。每次只构造最多 1024×512 的 mask，避免展开完整多头注意力矩阵。
- 灵感来自现有 Prompt Relay 适配器；短阶段的秒数偏置是固定工程选择。T5 特征带上下文，VAE 时间感受野重叠，因此并不实现完全独立的阶段语义。

## 运行与监视

所有输出、缓存、测试临时文件在 `/data/zjr/video_count_wan22_phase_control_20261030/`。Python 环境仍在 home，模型读取 `/data/zjr/models/`。

```bash
cd /home/zjr/Tracker
bash video_count_probe/wan22_phase_control/run.sh prepare
bash video_count_probe/wan22_phase_control/run.sh test
bash video_count_probe/wan22_phase_control/run.sh check
# 如后台服务已经启动，不要再次执行下面的启动命令。
CUDA_VISIBLE_DEVICES=5 video_count_probe/.venv/bin/python -B -u \
  video_count_probe/wan22_phase_control/operations/launch.py
```

启动器按顺序执行 CPU 文本编码、两组各一条完整分辨率的两步 smoke、六条正式 pilot。smoke 不评估动作效果。两组 smoke 都通过且条件/无条件调用数正确，pilot 才能启动。失败即停止；不自动更换参数、降低分辨率或扩大实验。

启动前需至少 22500MiB 可用、已用不超过 1000MiB 且利用率不超过 10%。启动后的显存竞争仅记录告警，不再根据固定容量阈值发送 SIGTERM；真实推理错误仍会停止序列。该检查不是 GPU 独占预约，其他作业仍可能进入并引起 OOM。

2026-10-30 修复：第一条 smoke 已保存，第二条完成采样后在 VAE 解码中被旧监控终止。当时剩余 7318MiB，外部进程占用 1922MiB；旧策略因估算容量 22205MiB 低于固定阈值 22500MiB 而误停，并非实际 OOM。修改仅涉及 `operations/launch.py`，推理源码哈希与实验参数保持不变，成功视频可校验复用；启动器单独记录 SHA。

```bash
tail -f /data/zjr/video_count_wan22_phase_control_20261030/logs/sequence.log
systemctl --user status wan22-phase-control-20261030-retry2 --no-pager
bash video_count_probe/wan22_phase_control/run.sh status
```

## 判读

正式输出在 `pilot/videos/`。`pilot/review.html` 和 `pilot/review_frames/blind_manifest.json` 隐藏组别与种子用于视觉检查。逐帧记录完整周期、半次动作、开始姿态、结束静止，以及冻结、变形和突变；视频不足以判断时保留 unknown。`annotations_template.json` 需要填写并另存为 `annotations.json`；模板不会覆盖已有正式标注。

```bash
bash video_count_probe/wan22_phase_control/run.sh report
```

主要比较完整次数是否恰好两次、完成后是否继续动作；另报告逐次时间、初始站立、最后一秒静止。固定阶段时间表是控制目标，计数主指标不额外要求严格卡在 5.5 秒结束。每种子列出两组结果，不对三个样本作稳定提升或显著性声明。若数字正确但出现画面冻结/突变，不能作为可用的视频控制成功。

CPU 测试覆盖实际 Wan attention 与独立 dense 公式对齐、零偏置与原生输出一致、条件/无条件区分、异常后恢复，以及实际 Wan2.2 expanded-timestep pipeline。GPU smoke 另检查真实权重、完整视频尺寸、解码与显存。
