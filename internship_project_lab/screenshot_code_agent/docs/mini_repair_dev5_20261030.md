# 2026-10-30：固定五题的失败反馈与重试控制

## 后续改版：mini_recovery（2026-10-30）

入口`scripts/run_mini_recovery_dev5.sh`，配置`configs/mini-recovery-dev5.json`，开关mini_recovery要求mini_repair_control。旧输出及冻结代码不修改。最终本轮输出独立放`/data/zjr/screenshot_code_agent/mini_recovery_dev5_20261030_v2`，仍为同5道已知开发题、原Qwen3-Coder BF16、TP4、两题并发、96次调用/1800秒、原最终评分器，没有截图输入。

改动：

- 新增根目录repro/reproduce/debug/temp等明确调试产物只从导出补丁排除，工作区文件保留；原仓库已有文件不受此规则影响。
- 将已有可见开发契约拆为独立检查组，输出具名失败及traceback；不因第一组断言失败而跳过后续组。语法/导入正常时，即使行为检查失败也进行干净副本的公开回归比较，合并反馈。没有追加标准参考补丁或使用隐藏评分选择候选。
- 保存部分候选必须同时满足：原公开测试无新增回归；通过的具名行为集合严格包含原代码的通过集合，并且更新时严格包含上一部分候选的通过集合。最终优先完全通过候选，否则可选择部分候选，并明确verified=false、partial=true。部分检查通过不等于最终任务成功。
- 所有工具后的源码状态检查采用可信基线Git目录、文件stat缓存及Python AST。注释/文档字符串不改变进展状态；bash修改与Git撤回均可识别。非Python源码使用内容摘要。调试文件、测试及缓存不计为实现进展，未改变文件不反复读取内容。连续12个动作没有新的有效源码状态且存在未解决检查失败，下一次模型请求前停止；读代码和重写自测也计数，回到已经出现过的源码状态不重置。
- 自动公开检查间隔从12个动作改为6个动作。普通命令30秒、可见行为10秒、公开pytest90秒及原总预算保留。新状态只说明源码发生变化，不保证变化方向正确；不断产生新但错误的源码仍可能耗尽总预算。

回放发现pvlib旧候选还将“interp缺少参数”的原ValueError变成TypeError，原公开测试会拒绝。初次新增23项回放中22通过，1项失败是测试错误预期它应保存为部分候选；核对原测试及新日志后，将期望修正为必须拦截该回归，未放宽运行时判断。定向9项复核通过。其余回放确认PyVista中途补丁在撤回后被保留，marshmallow功能错误不再被空调试文件遮挡，已成功的astroid/pydicom继续通过。完整冻结预检结果另行补充。

第一份冻结快照mini_recovery_dev5_20261030的完整113项预检通过，未启动GPU。随后复查发现旧持久反馈只保留最后3000字符，可能丢失开头的失败摘要；已改为优先保留开头6000字符并增加长日志回归检查。为不修改冻结文件，最终版本改用_v2独立输出与预检；旧快照保留作为验证记录，不用于模型评测。

最终_v2冻结预检113通过、0失败、0跳过，222.70秒；session_preflight.json确认passed=true、checks=113、gpu_started=false。Ruff及Shell检查通过。初始服务在3卡显存不足时等待，尚未加载模型；已停止并更换为sca-mini-recovery-dev5-20261030-gpu5.service，实际使用2、5、6、7卡（命令尾部--gpus覆盖默认卡号）。仍保留显存门槛与自动等待，冻结代码不变，模型最终成绩尚未产生。

启动更新：四卡初检通过后，其他负载在模型初始化期间增占显存；vLLM报Free memory 1.61/23.56GiB，小于所需17.43GiB，启动失败，尚未进入题目评测。原服务已restart，控制器PID3858044，active/running，重新等待显存。此时2卡余1918MiB、6卡余18676MiB，均低于19723MiB门槛；没有降低门槛、停止其他任务或更改冻结代码。共享显存检查不等于预留资源，无法阻止其他作业后续增长。

监视入口：

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_mini_recovery_dev5.sh status'
journalctl --user -fu sca-mini-recovery-dev5-20261030-gpu5.service -o cat
```

源码、预检收据和实际候选均保存在该轮输出；监视新服务时不要混用上一轮mini-repair服务名。最终模型成功率必须等本轮report.json。

## 完成结果（后续更新，优先于下文启动快照）

服务成功退出，report.complete=true，5/5进入最终评分并有可解析测试结果。2026-10-30 00:32:50—01:58:31 UTC，控制器总时长1小时25分41秒；首题开始至末题结束约1小时16秒。成功2/5，仍是astroid-984、pydicom-793，与mini_verified相同，没有新增成功题。这是固定已知开发题、无截图输入的回归。

| 任务 | 上轮目标测试 | 本轮目标测试 | 本轮原有回归 | 结果 |
|---|---:|---:|---:|---|
| marshmallow-2123 | 21/23 | 22/23 | 145/145 | 失败，repr默认仍漏absolute=True |
| astroid-984 | 2/2 | 2/2 | 42/42 | 成功，本轮额外路径检查也通过 |
| PyVista-4329 | 1/3 | 0/3 | 195/195 | 失败，末尾撤回源码修改，只剩空调试文件 |
| pvlib-1832 | 4/5 | 4/5 | 435/435 | 失败，physical的n_ar仍未传递 |
| pydicom-793 | 13/13 | 13/13 | 72/72 | 成功 |

模型调用297→259（减少12.8%），输入token 4,737,072→3,926,153（减少17.1%）；输出75,686。实际题目评测窗口仍约60分钟，上轮总时长71分54秒，本轮85分41秒，包含等待和加载，不能称整体加速。报告模型启动1045秒。astroid约1801→594秒，调用55→35，补丁增加根目录退出条件，公开额外行为及最终评分均通过。三道失败题均耗尽1800秒级预算；其模型请求分别约1762/1710/1695秒（marshmallow/PyVista/pvlib），不能把主要耗时归因于执行测试。

### 已证实的剩余问题

1. **交付检查短路遮挡功能失败。** 三道失败题都保留根目录空reproduce_issue.py。mini_delivery在发现此类文件时不运行visible_development_case，报告该字段为null；提交反馈持续要求迁移调试文件。模型曾手工运行可见脚本而看到功能错误，因此不能说从未收到功能反馈，但持久反馈被较低优先级清理问题占据。新增更严格产物检查带来了实际流程副作用。
2. **防循环规则覆盖窄，且未触发。** 三道失败题最高stalls分别5/2/1，没有达到6；exact-timeout retry阻止次数为0。模型改用读取、自己重写自测等动作绕过计数，注释编辑也会重置计数。没有no_progress退出，不能将调用减少归因于成功触发防循环。
3. **PyVista撤回候选后丢失功能改动。** 第65步再次写入错误的单参数SetCells，第66步已明确TypeError；第71步git checkout HEAD -- pyvista/core/pointset.py，第73步自测显示faces仍可写。最终patch仅含空reproduce_issue.py。此前没有候选完全通过公开检查，候选保留机制未能提供可回退的部分进展；全部任务used_earlier_candidate=false。报告zero_patch=0仅说明补丁字节非空，不代表存在有效源码修改。
4. **局部契约仍未落实。** marshmallow修正了异常文案，但只在absolute=False时显示repr参数；pvlib仍只补interp可选参数，没有修physical n_ar。两者原有回归通过不能替代目标契约通过。

三题末次请求出现APITimeoutError，结合退出原因budget_exceeded和约1800秒耗时，应解释为预算结束时的请求超时，不能笼统称所有API调用无错误；报告runtime_failures/recoverable_tool_errors/agent_errors均为0。五题独立评分完成，没有评分基础设施失败证据。

下一步应先修检查链：聚合清理与功能失败、避免空调试文件遮挡行为验证；以源码行为变化及失败集合识别无进展，不能只统计工具名或任意文本编辑；保留并比较部分有效候选，避免清空源码后仅靠空文件构成非空补丁。先做本轮真实轨迹CPU回放验证，再决定小规模模型重测；本次仅分析与归档，没有再次启动推理、重评分或改动历史补丁。

证据：本输出目录report.json、status.json，以及mini/runs各题result.json、events.jsonl、patch.diff、delivery_check.json。分析基于原文件离线读取。

承接mini_verified的2/5结果。当前修改本地mini适配及开发验收，不修改旧输出或标准评分协议，不把已知题回归当作未见泛化。

## 修改

- `mini_repair_control`显式开关，依赖mini_verified。未解决的公开失败在下一次模型请求末尾再次提供；即使较早历史被压缩，当前失败仍可见。
- 新增submit工具，仍走相同提交验收，避免模型用普通echo不断宣称完成而未调用提交协议。
- 普通bash命令上限30秒，包含pytest的命令沿用原上限；可见开发断言上限10秒。时间不足仍受原任务总预算限制。
- 同一命令曾超时且未发生成功edit_file修改时，不再执行，返回诊断提示。成功编辑解锁重试，失败编辑不解锁。
- 已知失败后累计6次check/submit/状态echo而没有成功源码编辑，会以no_progress结束该次Agent尝试，保留候选进行原最终评分。源码读取不计入状态重试；成功验收清除此计数。此机制限制无效消耗，不保证提升成功率。
- 新开发提示覆盖父目录遍历终止、VTK setter参数、physical可选参数；URL的repr及异常文案作为已知开发契约补充。新增条件来自上一轮分析，须披露为开发监督。
- 提交检查补充根目录reproduce_调试产物识别。没有直接改写模型的历史补丁。

注意：重试解锁依据成功的edit_file工具；推荐通过此工具修改。bash仍保留，但直接通过bash改源码不会自动解锁此前被阻止的完全相同超时命令。这是明确的控制边界，不应描述为任意shell编辑均能识别。

## 验证

复用既有85项环境/流程检查，并增加重试控制测试：重复失败停止、源码编辑重置、超时重复被阻止、正常读取不阻止、短超时与提交协议、真实MiniModel下一请求包含失败事实。

新预检重放上一轮mini_verified补丁：预期四份仍有缺陷的补丁被拒绝，pydicom通过；另以v4已成功的astroid/pydicom模型补丁作正向对照。仅读模型历史补丁，不使用参考标准补丁。所有临时数据位于/data。

实际冻结环境预检：90通过、0失败、0错误、0跳过，153.304秒；预检收据passed=true，gpu_started=false。Ruff和Shell语法检查通过。这验证执行控制与已知补丁回放，不代表新模型已完成修复。

## 运行和交付

入口：`scripts/run_mini_repair_dev5.sh`；配置：`configs/mini-repair-dev5.json`。原Qwen3-Coder BF16、TP4、同样5题、两题并发，96次模型调用和1800秒任务总预算不变。

输出：`/data/zjr/screenshot_code_agent/mini_repair_dev5_20261030/`。
已启动后台服务：`sca-mini-repair-dev5-20261030.service`，启动核验active/running，控制器PID3383486。实际启动指定2、5、6、7卡；脚本默认卡号为2、4、5、7，命令尾部参数覆盖默认值。

启动后日志阶段为waiting_for_gpu_memory，尚未加载模型：2卡随后被占用5520MiB，剩余18608MiB，低于本轮19723MiB门槛；5、6、7卡满足要求。服务自动等待显存达标，不需重复启动。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_mini_repair_dev5.sh status'
journalctl --user -fu sca-mini-repair-dev5-20261030.service -o cat
```

预检收据为输出目录`session_preflight.json`，模型实际成绩须以完成后的report.json为准。代码冻结后，工作区后续改动不会改变该次运行。
