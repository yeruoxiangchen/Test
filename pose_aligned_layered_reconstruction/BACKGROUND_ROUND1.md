# 第一轮后台运行与监视

2026-09-09：按用户要求转为独立后台服务，不再由对话前台持续等待。

服务：`tracker-palr-round1-20260909.service`。使用 GPU 1–6，不占用 GPU 0/7。
已为用户 `zjr` 启用 linger；关闭对话、终端或退出登录不依赖前台会话维持训练。
这是临时 systemd 服务，**不承诺机器重启后的自动恢复**。

## 查看状态

推荐使用中文进度面板，直接显示各阶段分母与当前训练步数：

```bash
cd /home/zjr/Tracker
/home/zjr/anaconda3/envs/reconviagen/bin/python -m pose_aligned_layered_reconstruction.monitor_round1 --watch 5
```

面板每5秒刷新，直接读取训练及校验文件；按 Ctrl+C 只退出监视，不会停止训练。
不加 `--watch 5` 则只显示一次。看最上方“总状态”，只有“整轮完成”才表示本轮结束。
LaRI分母是4,000 updates，SS分母是3,000 updates，Dev64评估分母是256条。
启动时会先显示六个进程合计校验了多少个缓存，`0 updates` 在这时不表示卡住。

不需要切换目录：

```bash
systemctl --user status tracker-palr-round1-20260909.service
journalctl --user -u tracker-palr-round1-20260909.service -f
```

查看结构化进度及完整文件日志：

```bash
cd /home/zjr/Tracker
jq '{state,stage,updated_utc,progress,preparation_errors,error}' pose_aligned_layered_reconstruction/outputs/round1_20260909/background_status.json
tail -f pose_aligned_layered_reconstruction/outputs/round1_20260909/background.log
```

`background_status.json` 每约30秒更新一次，含当前阶段、训练step、已准备对象数、评估记录数、任务PID及日志位置。
`jobs/` 保留所有历史尝试，因此旧任务的 `failed` 不等于当前后台失败；以总状态、`preparation_errors` 和当前任务为准。
单步loss不是 occupancy 成绩；最终通过完整扩散采样后计算 IoU/Precision/Recall/F1。

## 固定流程与预算

1. 原32对象SS训练到2,000步已完成；后台做最终32对象×2 seeds的occupancy复核。
2. 最终过拟合门通过且2,000对象缓存准备完整后，才允许正式训练。
3. 正式LaRI头随机初始化，6卡每卡1对象，4,000 updates。
4. 正式SS重新从SS30K EMA+零初始化新增模块出发，冻结正式LaRI头，6卡，3,000 updates。
5. 固定最终checkpoint评估 reserved Dev64；25步、CFG5、seeds42/43、父模型/候选配对，共256条occupancy记录。

预算和判据见 [EXPERIMENT_ROUND1.md](EXPERIMENT_ROUND1.md)。
任一检查失败会阻止后续正式训练；不换掉困难对象，不按Dev64选checkpoint，不悄悄延长训练。
已有数据准备可按hash验证后续做；正式训练**未实现完整优化器自动断点续训**，后台不会把重复从头训练冒充续训。

## 文件位置

所有相对路径均在 `outputs/round1_20260909/` 下：

- `background_status.json`：总状态；`running` / `blocked` / `stopped` / `complete`。
- `background.log`：调度、心跳、错误；`jobs/<任务名>.log`：每项任务的详细输出。
- `background_plan.json`：固定预算及交接时生产代码SHA；后续启动阶段发现代码被改会暂停。
- `overfit32_gate_final.json`：最终过拟合门及配对occupancy结果。
- `formal2000_lari_v2/checkpoints/step_004000.pt`：启动超时修复后的正式LaRI最终checkpoint计划位置。
- `formal2000_ss_v1/checkpoints/step_003000.pt`：计划的正式SS新增条件最终checkpoint。
- `formal_dev64_eval/aggregate.json`、`round1_report.json`：计划的最终配对结果。

正式checkpoint与最终报告只有相应阶段成功后才会生成。`complete` 表示实验完成，**不表示一定优于父模型**。
本轮终点仅为SS occupancy，不是最终Mesh精度或遮挡鲁棒性证明。

## 2026-09-09 启动超时恢复

原 `formal2000_lari_v1` 在任何训练更新之前失败：rank0独自校验全部2,000对象，
其余rank等待时超过NCCL的600秒默认同步超时。旧目录只含run_config，旧日志完整保留。
修复为六rank按不重叠索引分片做完整SHA校验，交换并核对全部校验回执后才初始化模型；
同步超时限额改为30分钟，并写入逐rank校验进度，不跳过任何数据完整性检查。

用户要求继续后，仅恢复尚未开始的正式训练，使用新目录 `formal2000_lari_v2`；
已完成的过拟合、2,000对象数据和Dev64数据不重做，4,000/3,000步预算不变。
此次恢复记录为 `background_recovery2.json`，不是优化器断点续训。
`background_status.json` 的 `formal_runs` 指定当前实际运行目录，监视面板会自动跟随。

## 停止

```bash
systemctl --user stop tracker-palr-round1-20260909.service
```

停止后台及其本轮进程；已保存结果与checkpoint保留。不要直接 `restart` 以为能够续训，
中断后的正式训练需检查最后完整checkpoint再决定恢复方案。

无需服务管理权限的停止请求也可在本轮目录创建 `STOP` 文件；后台下一个轮询时会停止。
