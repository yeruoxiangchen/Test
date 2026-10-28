# LaRI Causal4，2026-10-23

固定模型的分项诊断；[协议](PROTOCOL.md)定义所有范围与判读限制。
新输出位于`pose_aligned_layered_reconstruction/outputs/hssd_lari_causal_20261023`。

在预检通过后，单卡串行运行192条SS干预＋72条固定案例mesh：

```bash
HSSD_CAUSAL_GPU=0 bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_lari_causal_20261023/run.sh start-run
```

监视（Ctrl-C只退出监视）：

```bash
bash /home/zjr/Tracker/pose_aligned_layered_reconstruction/experiments/hssd_lari_causal_20261023/run.sh watch
```

GPU为物理编号；换卡只修改HSSD_CAUSAL_GPU。命令可在任意目录执行。
脚本不抢占其他任务。重启同一命令会核验身份并跳过已有记录。
可以分别使用start-ss或start-mesh，不能同时启动两个队列。
完成状态为diagnostics_complete_review_required；单阶段分别是ss_complete_review_required、mesh_complete_review_required。

本轮没有训练。原LaRI的拟合审计为audit_summary.json；因果替换结果为ss_summary.json；
固定案例mesh结果为mesh_summary.json，四视角并排图在figures/。所有病例与seed均保留。
GT替换和事后选出的4对象只用于诊断，不是可部署方法或泛化成绩。

短时检查入口：audit是32条样本CPU审计；preflight是两个原生25步SS回放及16次干预前向，
没有完整mesh解码或正式长队列。首次运行由代理完成，用户无需重复运行。
