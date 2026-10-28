# HSSD CellQuery16，2026-10-23

在相同外观基线上检验SS格点查询与27点格内平均。详见[固定协议](PROTOCOL.md)。
所有命令可从任意目录执行，GPU使用物理编号；脚本不会自动抢占或更换GPU。

第一轮工程预检通过后，启动十组各200步训练并接SS评估：

```bash
HSSD_CELL_GPU=1 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_cell_query_20261023/run.sh start-run
```

监视（Ctrl-C只退出监视）：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_cell_query_20261023/run.sh watch
```

run只执行训练和SS，完成后ss_complete_review_required表示等待结果审阅。
重新执行start-run会在身份核验后跳过完成阶段、恢复未完成训练/采样。
也可分开使用start-train、start-ss；不能同时启动多个队列。

完整mesh是单独的耗时任务；审阅SS后再由用户启动：

```bash
HSSD_CELL_GPU=1 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_cell_query_20261023/run.sh start-mesh
```

只读状态用status。preflight执行五组各2步与单个25步SS解码，start-preflight后台运行。
diagnose/start-diagnose为32样本CPU几何汇聚诊断，不采样生成模型。

验证集清单/几何坐标检查（CPU）：

```bash
cd /home/zjr/Tracker
/home/zjr/anaconda3/envs/reconviagen/bin/python -m pose_aligned_layered_reconstruction.experiments.hssd_cell_query_20261023.validation select
/home/zjr/anaconda3/envs/reconviagen/bin/python -m pose_aligned_layered_reconstruction.experiments.hssd_cell_query_20261023.validation audit
```

验证32对象来自两个独立场景，与训练场景所有资产家族隔离。清单不包含训练输入
或latent；还需场景图像/标签/可见率及latent资格检查后才能用于独立验证。
本轮SS训练与评估仅使用原16对象，不读取验证数据。

输出：`pose_aligned_layered_reconstruction/outputs/hssd_cell_query_20261023/`。
`geometry_summary.json`为接口诊断；`ss_summary.json`为完整SS比较；
`mesh_summary_{clean,occluded}.json`为完整mesh比较；`validation/selection.json`
为冻结验证清单。阶段完成不代表论文假设成立。
