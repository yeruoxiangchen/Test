# HSSD16 嵌套残差实验

实现与原实验隔离。共同父模型为每域已有的 **no_layer500 raw**，保留并冻结其
完整外观路径；新增初始为零的残差。三组分别读取外观、GT几何、预测几何。
其中GT组仅用于定位几何上界，不作为正常推理方法。

详细结构、训练预算、坐标和评测规则见 [PROTOCOL.md](PROTOCOL.md)。
结果写到 `pose_aligned_layered_reconstruction/outputs/hssd_residual_20261021/`。

## 启动与监视

在 `/home/zjr/Tracker` 下执行，`HSSD_RESIDUAL_GPU` 为物理卡号，可自行替换。
预检完成后启动单卡训练；该命令在后台运行，关闭当前SSH终端不影响子进程。

```bash
HSSD_RESIDUAL_GPU=3 bash pose_aligned_layered_reconstruction/experiments/hssd_residual_20261021/run.sh start-train
```

监视（不占用GPU，也不会启动训练）：

```bash
bash pose_aligned_layered_reconstruction/experiments/hssd_residual_20261021/run.sh watch
```

单次查询：

```bash
bash pose_aligned_layered_reconstruction/experiments/hssd_residual_20261021/run.sh status
```

训练为 clean/occluded × appearance_control/oracle/predicted，共6组，每组200步。
结束状态为 `training_complete_pending_evaluation`，此时尚无新mesh结论。
随后单独启动完整采样、mesh指标、图库和参数匹配消融：

```bash
HSSD_RESIDUAL_GPU=3 bash pose_aligned_layered_reconstruction/experiments/hssd_residual_20261021/run.sh start-eval
```

评估生成192个新mesh，核验后复用原no_layer的64个基线mesh。结束状态为
`evaluation_complete_review_required`，不是自动宣布拟合通过。查看
`comparison.json`、各 `evaluations/<domain>_<arm>/summary.json` 以及
`evaluations/<domain>_<arm>/<index>/<seed>_comparison.jpg`。
必须同时判断 predicted 相对冻结基线和额外 appearance_control 的收益。

中断或运行异常后，使用相同 `start-train` / `start-eval` 命令继续；只恢复相同
代码、数据、基线及预算。队列锁阻止重复并发。训练按last.pt精确恢复，评估核验
后跳过已有病例，不因失败删对象。源码或协议修改后会拒绝混用旧结果，应另建版本。
`status.json` 是最近任务状态，详细异常看其 `log` 字段所指文件。

## 工程预检

重新验证当前代码时执行：

```bash
HSSD_RESIDUAL_GPU=3 bash pose_aligned_layered_reconstruction/experiments/hssd_residual_20261021/run.sh preflight
```

预检只做同一对象的三组各两步更新、零初始化/梯度/保存恢复检查及25步SS采样
和SS解码，不做完整SLat/mesh。成功写入 `smoke.json`；正式训练要求该记录与
当前源码相符。预检临时权重不用于正式实验，正式残差重新从相同零初始化开始。

CPU测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/zjr/anaconda3/envs/reconviagen/bin/python -m unittest \
  pose_aligned_layered_reconstruction.experiments.hssd_residual_20261021.test_model
```

本轮不重新编码latent、不训练SLat、不将oracle结果当作预测模型成绩，也没有
自动扩量步骤。原遮挡SLat失败保留为独立待解决问题。
