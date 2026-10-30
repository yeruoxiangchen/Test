# 固定8秒：目标次数1/2/3响应诊断

Wan2.2-TI2V-5B，193帧、24fps、1280×704、50步、CFG=5、flow shift=5。
三个种子2026、3407、7919。新增目标1次与3次各三条，复用上一轮C8/目标2次的三条视频及AI标注。
每条新视频的初始噪声SHA256必须与同seed的旧C8视频一致。新旧使用相同模型、软件环境、推理及VAE解码函数。

目标2次提示词逐字保留；目标1/3使用同一场景、动作定义、停止条件，并相应减少/增加阶段描述。
这是次数指令及对应阶段描述的响应诊断，并非只改变一个数字词的严格消融。只有一种动作、三个种子，
且2次结果已经看过，因此结果只作探索性依据，不宣称独立验证、稳定效果或已经形成新方法。

数据、日志、缓存、测试临时目录：`/data/zjr/video_count_wan22_count_response_20261029`。
不下载权重、不修改旧实验、不自动训练或扩展实验。

`run_sequence.sh`与`operations/guarded_retry.py`还将CUDA驱动缓存、Triton缓存指向本实验的`cache/`目录。
解释器及依赖仍来自home上的`video_count_probe/.venv`和`anaconda3/envs/hunyuan3d2mv`，
因此启动/导入库仍可能读取home；数据和权重在data不代表运行环境也已迁移。CPU offload使用系统内存，不是将权重写回home。

`operations/guarded_retry.py`用于失败后续跑。启动检查与原推理脚本一致：空闲显存至少22500MiB、
已用显存不超过1000MiB、利用率不超过10%，连续三次满足后启动；允许少量后台CUDA上下文。
运行期间不再因“出现其他进程”直接终止，而是估算留给本任务的总容量（包含它已经占用的显存）。
只有容量连续三次低于22500MiB才中止本任务，退出75；不终止其他任务。
监测不构成GPU独占预订，也不保证避免瞬间增长引起的OOM。最长等待30分钟，原推理参数及代码哈希保持不变。

```bash
cd /home/zjr/Tracker
bash video_count_probe/wan22_count_response/run.sh prepare
bash video_count_probe/wan22_count_response/run.sh test
bash video_count_probe/wan22_count_response/run.sh check
CUDA_VISIBLE_DEVICES=4 bash video_count_probe/wan22_count_response/run_sequence.sh
```

顺序执行CPU文本编码、最长提示词的2步全分辨率smoke、六条正式生成。
默认要求空闲卡和至少22500MiB空闲显存；上一轮8秒实测allocated约18000MiB，reserved约21902MiB。
上一轮约34分钟/条，六条约3.5小时，不含编码/smoke/加载。显存和速度仍以新运行实测为准。
不要在同一输出目录重复启动；已有进程用status检查。中断后按相同命令续跑会校验并跳过完成结果。
明确失败（例如OOM）排查后用`run.sh smoke|pilot --retry-failures`；程序不会自动降分辨率或改步数。

```bash
bash video_count_probe/wan22_count_response/run.sh status
bash video_count_probe/wan22_count_response/run.sh status --mode smoke
```

完成会生成`pilot/review.html`、`review_frames/blind_manifest.json`和`annotations_template.json`。
新增视频对评审隐藏目标/种子，先记录完整周期及残缺动作，再揭示目标计分。原目标2次标注单独保留来源。
保存标注到`pilot/annotations.json`后执行`run.sh report`。AI标注需填写`rater_type=ai`，不得当作独立人工评测。

主指标：完整次数是否准确、达到目标后是否停止（不能有额外完整或残缺动作）。无固定完成时刻要求。
诊断指标：逐seed的目标1/2/3实际计数、是否单调响应、是否三种目标都生成相同次数、初始是否站立、末秒是否静止。
动作周期沿用“直立→明显屈膝和髋部下降→直立”，没有预设蹲深阈值，浅蹲应明确记录并做敏感性复核。
`assessment_complete=false`在生成结束但未审阅时正常；`generated_new=6/6`只表示六条新视频生成完成。
