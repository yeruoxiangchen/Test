# Full 环境预检修复与恢复（2026-10-25 UTC）

## 发现的问题

旧服务 `sca-scale4gpu-dev-20261024-retry1.service` 仍在推进，并非模型推理卡死。环境安装、权重下载和四卡工具调用检查已经完成，但预检出现大量环境错误，且旧流程要求开发／测试共500个候选全部完成后才开始开发集评测。

01:27 UTC检查时，开发125个候选有37个通过、34个环境失败、17个异常、37个未完成；测试375个尚未处理。失败发生在模型答题之前，不是模型修复成绩。随后停止旧队列进行修复，原始日志、镜像缓存和成功任务保留；失败重试归档到原数据目录的 `preflight_attempts/`。

## 修复内容

- Docker Hub访问凭证自动刷新，包括manifest和并发blob下载；TLS错误有限重试。fresh token无匿名pull权限与过期token分别处理。
- 历史依赖以独立只读覆盖层固定：pvlib使用NumPy1.23.5／Pandas1.5.3／SciPy1.10.1，较新Python的pydicom使用pytest7.4.4，PyVista使用VTK9.2.6。Python3.6／3.7的pydicom保留原pytest，避免不兼容升级。图形库仅补充四个已记录SHA的共享库。
- 使用同一固定官方pytest-options parser规范化期望测试ID，保留原始与规范化映射，拒绝冲突，不删除测试或降低通过条件。
- 原镜像不可匿名取得时，可复用同仓库的已固定运行环境，导出目标题的准确base_commit；逐文件Git blob、路径和模式校验。记录真实环境digest和适配来源，不使用donor的当前源码，不将其称为原始官方镜像。
- SQLFluff的原始内部目录链接按白名单保留；目标源码重新生成egg-info，避免沿用donor的旧版本和插件入口。生成前后核对源码不变。
- 开发集与独立测试集分别预检、分别封存。开发125个候选完成且达到100个有效任务后即可进行开发集对照，测试375个候选不再阻塞开发评测。100／300目标、固定候选和模型失败分母保持不变。

兼容层只改变运行依赖。每道题仍须验证原始错误存在、原参考补丁通过全部要求的测试。协议仍是明确记录的OCI适配评测，不是官方Docker排行榜提交。

## 已验证的实际恢复

| 样例 | 原始代码的目标测试 | 参考修复的目标测试 | 验证方式 |
|---|---:|---:|---|
| pvlib-python-1852 | 0/2 | 2/2 | 正式audit入口；300项保留测试均通过 |
| pydicom-995 | 0/256 | 256/256 | 正式audit入口，修正44个测试ID的路径格式 |
| sqlfluff-5074 | 0/7 | 7/7 | 正式audit入口，缺失镜像适配及目标源码元数据 |
| pydicom-1720 | 0/345 | 345/345 | 隔离依赖对照，pytest7.4.4 |
| pyvista-4406 | 0/2 | 2/2 | 隔离依赖对照，VTK9.2.6，6项保留测试通过 |

完整工程测试173项通过（49.57秒），Ruff通过。诊断脚本及逐题证据在 `docs/artifacts/scale_repair_20261025/`，正式重试审计在原数据目录。已通过样例不代表其余所有环境已通过，批量恢复结果以新服务进度为准。

## 恢复任务与监视

新服务：`sca-scale4gpu-repair-20261025.service`。

新输出：`outputs/scale_4gpu_repair_20261025/`。旧输出及旧冻结runtime保留。

恢复动作复用已完成的环境安装与模型下载，只重试失败／未完成的开发候选。随后比较原8B基线与Coder；基线使用2号卡，Coder使用2、5、6、7四卡，每卡显存预算80%，另需1536 MiB余量，容量不足时等待。模型服务在实际评测前仍执行工具协议检查。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_scale_4gpu.sh status --output /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/outputs/scale_4gpu_repair_20261025'
```

```bash
journalctl --user -fu sca-scale4gpu-repair-20261025.service -o cat
```

开发报告将写入 `outputs/scale_4gpu_repair_20261025/dev/comparison.md`。独立测试尚未运行，后续需显式准备测试环境并启动测试，不能将开发结果当独立测试成绩。

02:36 UTC复查：新服务仍在运行，已完成33个重试工作进程；开发集有效题数从最初37增加到67，当前汇总为67通过、14环境失败、5异常、39未出结果（其中包含正在重试／尚未重试的旧失败）。正式模型评测尚未开始。

## 03:58 UTC：基线模型服务退出

本次预检已经完成：125个开发候选中110个通过、12个环境失败、3个异常，按固定规则选出100题。退出发生在基线Qwen3-VL-8B评测期间，尚未进入四卡Coder评测。服务PID1612625于03:53通过健康检查及工具调用往返检查，最后一条成功响应在03:56:08，控制器在03:58:13确认退出。

服务日志没有Python异常、CUDA OOM或正常关闭记录；原控制器未保存模型子进程退出码，当前账号也无内核日志读取权限。因此具体终止原因仍未确定，不能将它直接归因为显存不足。主机可用内存检查期间从约10GiB上升至约50GiB，仅说明同期负载有变化，不能证明此次退出原因。

三个已启动任务中，`marshmallow-code__marshmallow-2123`因模型输出格式错误正常结束，结果保留；`pvlib__pvlib-python-1616`和`pydicom__pydicom-816`均在同一时刻出现`APIConnectionError`，属于此次服务中断影响的任务。旧恢复逻辑只识别`cancelled`，漏掉了后者的`unresolved/model_error`；恢复需要将这两次中断原样归档，不能将它们计入已完成任务，也不能重跑真实的模型失败以择优计分。

原始状态、服务日志、固定计划及逐题结果摘要保存在 [退出诊断快照](artifacts/scale_repair_20261025/server_exit_0358/diagnosis.json)。冻结的Agent源码校验未变，开发集仍为原100题。

### 04:12 UTC恢复

控制器已统一识别取消及明确的连接中断，状态统计不再将这些结果计作完成，最终报告也拒绝含中断任务的结果。服务退出时在清理之前保存退出码、信号、日志位置、受影响任务及可用内存；模型服务开启Python故障堆栈记录。未修改冻结的Agent、模型权重、推理预算或固定题目。

验证：聚焦检查34项通过；完整测试182项通过（49.34秒），Ruff及语法检查通过。旧输出按新逻辑正确统计为完成1题、中断2题。此次只改控制器恢复与监控，沿用原输出目录，不混入新的模型或Agent策略。

恢复服务`sca-scale4gpu-eval-retry1-20261025.service`于04:12:02 UTC启动，控制器PID1638109，执行`resume --resume-interrupted`，跳过已完成预检及下载，继续原两组评测。GPU候选仍为2、5、6、7。

原来的`watch ... status --output outputs/scale_4gpu_repair_20261025`命令继续有效；实时日志改为：

```bash
journalctl --user -fu sca-scale4gpu-eval-retry1-20261025.service -o cat
```

以上记录证明恢复已启动，不代表100题对照评测已经完成，也不代表原始服务终止原因已查明。

## 05:36 UTC：启动自检请求超时

05:26:32 UTC重启同一服务后，固定输入校验和GPU容量检查完成，模型权重也加载成功。05:33:23的`GET /health`返回200，但随后的第一条`echo_probe`工具调用请求在180秒内未返回；05:36:24控制器抛出`TimeoutError`，进入`start_service`异常清理并向模型进程发送SIGTERM。这里的`returncode=-15`是超时后的主动清理结果，与04:34评测期间观察到的独立SIGTERM事件不同，不能把两次原因混为一谈。

本次没有启动新的正式评测题，原先完成的3题及中断记录保留。相同冻结模型服务以前约8秒、19秒通过了同一工具自检；这次日志显示请求进入生成代码，但没有响应完成、CUDA OOM或Python异常记录。检查时主机I/O等待较高、8GiB交换空间已用满，提示可能存在资源争用，但不足以确认具体阻塞位置。原始状态、日志及主机资源快照见 [启动超时诊断](artifacts/scale_repair_20261025/startup_timeout_0536/diagnosis.json)。

启动控制器现改为每条自检请求最多600秒，单个在途请求，不在超时后叠加重试；主线程每秒检查模型进程并更新`protocol_smoke`阶段、`tool_call`／`tool_response`步骤和等待秒数，每15秒采样进程内存／读盘及主机资源。通过完整工具往返检查才进入正式评测。失败时在主动清理之前保存`*_startup_failure_*.json`，分别记录清理前是否存活、原始退出码及控制器是否请求终止。

这里只调整启动等待与诊断，不改变冻结模型服务、Agent、100题清单、正式评测调用上限或token预算。代码修复不等于已证明资源争用原因消除；真实模型自检需下一次恢复运行确认。

验证结果：42项控制器聚焦测试通过（pytest计时51.35秒，解释器／导入启动另有明显读盘等待），Ruff和语法检查通过；冻结源码指纹未变。未额外启动GPU任务。用户可执行`systemctl --user restart sca-scale4gpu-eval-retry1-20261025.service`恢复；原status和journal监视命令继续有效。

## 06:04 UTC：正式评测期间再次收到SIGTERM

05:46:53的恢复已在06:01:14通过完整工具协议自检，并进入基线评测。06:04:06.181两道在途任务同时记录连接断开，06:04:07.205控制器确认模型PID1772662已因SIGTERM退出，然后才执行清理。这次与05:36自检超时后的主动清理不同。

内部审计未发现误杀路径：astroid任务仅执行读取文件和glob，pyvista任务尚无模型工具执行；冻结沙箱使用独立PID命名空间及进程组，沙箱超时清理使用SIGKILL。用户服务未配置运行时限或MemoryMax限制，journal中也没有先于模型退出的整项服务停止记录。已有日志无法确认SIGTERM发送者；内存读盘压力不能单独解释信号来源。

增设可选`--trace-server-signals`，仅对模型服务进行信号跟踪，记录SIGTERM发送方PID/UID，不忽略或阻止终止信号。使用`strace -DDD -f --seccomp-bpf -e trace=none`，保留模型作为控制器直接子进程及原退出语义，日志不记录请求体或环境变量。短小自有CPU进程实测已确认PID保持、退出码-15保持、发送者正确记录、tracer自行退出。诊断模式的耗时不能当作无跟踪的模型速度基准。

原状态、模型日志、独立退出记录与CPU探针证据见 [SIGTERM诊断](artifacts/scale_repair_20261025/sigterm_0604/diagnosis.json)。真实模型尚未开启该诊断模式，未声称已经找到信号发送者或根治退出原因。

验证：51项控制器聚焦测试通过（2.99秒），Ruff及语法检查通过；信号解析优先匹配模型主PID，并标记其他线程／子进程的信号，避免把子进程清理误当作主模型退出原因。

下一次手动恢复应启用诊断开关；普通重启旧unit不会自动添加该参数：

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
systemd-run --user --unit=sca-scale4gpu-signaldiag-20261025 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/run_scale_4gpu.sh resume \
  --output outputs/scale_4gpu_repair_20261025 \
  --gpus 2,5,6,7 --allow-shared-gpus \
  --gpu-memory-utilization 0.80 --gpu-headroom-mib 1536 --wait-for-gpus \
  --resume-interrupted --trace-server-signals
```

监视新unit：`journalctl --user -fu sca-scale4gpu-signaldiag-20261025.service -o cat`。原输出目录的status命令仍有效。未执行该长任务启动命令。

## 06:31 UTC：确认终止发送者为earlyoom

用户启动的诊断服务于06:18:20开始，06:31:06.820016跟踪日志捕获到模型主进程PID1810479收到`SIGTERM {si_code=SI_USER, si_pid=3993715, si_uid=61876}`，随后服务失败。宿主`ps`确认PID3993715是`/usr/bin/earlyoom`；`getent`确认UID61876是earlyoom动态用户；`systemctl show earlyoom.service`的MainPID也为3993715。因此本次终止来源已经直接确认，不再只是根据内存水平推测。

实际`/etc/default/earlyoom`及运行参数为`-m 8 -s 20`，在可用主机内存和剩余swap同时低于相应阈值时发送SIGTERM；swap已用满。另配置`--prefer '^python[0-9.]*$'`优先选择Python进程，`--avoid`包含node、SSH及VS Code相关名称，降低这些进程被选择的优先级。`--avoid`不是绝对豁免。模型退出后的MemAvailable快照不能当作发送信号瞬间的内存值。

VS Code常驻主进程仍占约19GiB；C/C++索引进程再次启动后几分钟内从约2GiB增至约6GiB。合理处理顺序是限制／暂时禁用当前工作区C/C++索引，保存工作并处理终端前台任务后重启远程编辑器服务，释放非必要内存；确保运行期间保有RAM余量后再续跑。不能把GPU空闲当作主机RAM充足，也不能通过忽略SIGTERM或无条件重启来代替内存治理。

已将信号原文、退出记录、服务配置与PID/UID核验结果保存至 [earlyoom确认记录](artifacts/scale_repair_20261025/earlyoom_confirmed_0631/diagnosis.json)。当前有效完成基线4/100题（0成功），两题因本次断连中断，Coder为0/100；这些少量结果不能作为最终模型结论。此次只读核验及归档，未修改系统保护服务、未停止其他进程、未再启动评测。此前无信号跟踪的SIGTERM不能逐次认定发送者相同。

## 20:31 UTC完成、23:21 UTC离线轨迹审计

用户恢复的控制器于06:53:47 UTC启动，20:31:59 UTC完成，实际历时13小时38分。两臂各100题、无遗留服务中断：baseline为11/100，Coder为17/100，配对为保留9、额外修复8、退步2、共同失败81。原始分母与分数不变。baseline另有4条`status=error`记录，仍计入失败；100条结果落盘不等于100条运行过程都无错误。详见[原始对照报告](../outputs/scale_4gpu_repair_20261025/dev/comparison.md)。本轮只有Full固定100题开发集的原始文本代码侧评测，没有300题独立测试成绩，也没有新增截图端到端成绩。

本次离线审计读取两臂全部200条`result.json/events.jsonl`，逐条配对工具调用和返回、核对最终usage与原报告，人工复核Coder全部29个零补丁失败及5个对照案例。新增[审计脚本](../scripts/audit_scale_trajectories.py)、[逐题CSV](artifacts/scale_repair_20261025/trajectory_audit_20261025/verified/tasks.csv)、[机器可复核统计与输入指纹](artifacts/scale_repair_20261025/trajectory_audit_20261025/verified/analysis.json)、[pydicom/pvlib逐题证据](artifacts/scale_repair_20261025/trajectory_audit_20261025/manual_medical.json)、[其他仓库及对照证据](artifacts/scale_repair_20261025/trajectory_audit_20261025/manual_other.json)。事件行号均指原始任务目录`events.jsonl`，不使用压缩后的conversation作为完整轨迹。

### 1. 首要发现：15个任务的文件工具存在确定的Python兼容缺陷

`OCISandbox.filesystem`使用任务自己的Python运行`/sca_worker.py`，但worker使用了目标环境不支持的语法/API：

| 实际错误 | 任务数（每臂） | baseline失败工具调用 | Coder失败工具调用 |
|---|---:|---:|---:|
| Python3.6：`from __future__ import annotations`解析失败 | 11 | 357 | 98 |
| Python3.8：`PosixPath.is_relative_to`不存在 | 4 | 90 | 33 |
| 合计 | 15 | 447 | 131 |

15题均来自pydicom，两臂在这些题均为0成功。以上是实际工具返回错误，不只是根据Python版本推测。例：Coder `pydicom__pydicom-816`事件8为首次glob的worker SyntaxError；`pydicom__pydicom-1090`事件7/8返回pathlib AttributeError。`pydicom-839`的事件68→70与91→93是实际源码编辑被worker挡住的直接证据。返回码没有被吞掉；错误以普通工具反馈继续交给模型，因而反复消耗调用。Bash仍可运行，不能把修复这些工具后的潜在收益直接算成15道必对。

预检只执行简单的`import os`探针和原始/gold补丁测试，没有运行glob/read/grep/edit/write的契约自检；因此gold指定测试通过与Agent工具不可用同时发生。原报告是**带有已知工具缺陷的系统表现**，不能据此认定Coder模型能力上限。不得静默删除15题并改写成功率。

### 2. Coder的调用花费与29个零补丁失败

Coder总计3431次模型调用，其中3240次决策、191次摘要；3226次工具调用分布如下。重复请求不自动等于无效请求，尤其修改后重跑测试可能必要。

| 工具用途 | 次数 |
|---|---:|
| read_file / grep / glob | 1346 / 397 / 173 |
| bash | 962 |
| edit_file / write_file | 118 / 126 |
| run_tests | 104 |

87题最终耗尽共享36次模型调用，全部报`Shared model call limit reached (decisions + summaries)`；其中83失败、4个已有补丁独立验证成功。它们没有触发时间或工具次数上限。`separate_summaries`分开记录决策/摘要，但摘要仍计入共享36次，这是现有预算政策及测试规定，不是计数bug。不能把87次预算停止直接称为87次修复失败。

Coder29个零补丁失败的互斥“观察到的主要模式”如下；这是事后轨迹分类，不是反事实因果实验。

| 模式 | 题数 | 观察 |
|---|---:|---|
| 工具worker不兼容，随后无效重试/低效shell恢复 | 10 | 8个Python3.6、2个Python3.8；Bash仍工作 |
| 一直读取、搜索、诊断，没有源码修改尝试 | 17 | 合计578次工具请求；人工复核48条bash均未尝试修改生产源码 |
| 编辑片段不唯一，反复匹配失败 | 1 | pydicom-1334六次编辑均match_count=2，文件哈希未变 |
| 唯一编辑指向受保护测试，被正确拒绝 | 1 | pvlib-1048修改测试导入触发只读错误，未转向源码实现 |

17个仅探索案例包含8个pvlib、2个astroid、1个pyvista、6个sqlfluff；其中360次read_file、124次grep、37次glob、48次bash、9次run_tests。pvlib-1469已在事件194指出重复传入albedo的问题，但最后六次调用仍只读取；pvlib-1296重复分页读取同一文件；这些是定位后未进入实现的明确例子。不能据此宣称把调用数翻倍一定能修好。

### 3. 摘要会误写执行状态；有补丁也不代表做了有效修复

Coder100题发生887次`context_compacted`事件，不代表887次模型摘要；模型摘要只有191次，其余可能是裁剪或本地提取。存在可核验的摘要失真：

- sqlfluff-3109事件61/104声称“Successful edit”，但整条轨迹没有任何修改请求。
- pvlib-1157事件37声称有最近测试失败，但没有run_tests；随后多次回读相同源码。任务共32次context_summary，其中30次为本地提取，不能重复计算为模型调用。
- astroid-927事件67/128把方法归属到错误的类，事件194才纠正；说明压缩后的状态也需要可验证来源。

这些例子证明摘要内容存在错误，不能证明所有回读都由摘要导致。下一版应从工具事实生成“已读/已改/已测”状态，模型摘要只保留假设与计划，不能独立声明修改或测试成功。

对照案例进一步区分“有diff”和“有效源码修复”：astroid-1262的449字节补丁只有诊断脚本；sqlfluff-4764没有edit/write/bash调用，1471字节diff来自测试期间生成或改动的文件，不能视为主动修复；pyvista-4311确实改了源码，却引入不存在的VTK方法并使6项原本通过的指定测试失败。成功例astroid-1903在事件54→56完成目标编辑，事件69通过指定F2P/P2P；sqlfluff-2386事件149已通过，后续继续尝试直到预算耗尽，最终仍判成功。

### 4. 避免把正常基线失败误判成环境错误

pyvista-432缺少helpers、astroid-2015缺少module_in_path、sqlfluff-3109缺少OutputStreamFormatter导致的测试收集ImportError，与预检baseline一致；gold补丁明确新增/重命名这些API，故这些是待修复的问题，不计入worker故障。模型应修改源码，而非保护测试。

预检评分只检查指定F2P/P2P：pyvista-432 gold指定13/13 F2P通过，但整条命令仍有3项np.int兼容失败；astroid-2015指定100/100 F2P通过，但另有1项弃用警告失败；两者P2P为空。因此应写“指定评测测试通过”，不可写“仓库所有测试全绿”。

### 5. 后续改进顺序（尚未实施或重新评测）

1. **P0 工具兼容与预检补齐**：在新冻结版本中让worker兼容任务实际Python3.6/3.8/3.9，保留原任务解释器和测试语义；在可丢弃副本中实际验证五类文件工具、唯一匹配、符号链接越界和只读保护。工具启动错误应在模型调用前阻止该任务，不能靠模型猜测恢复。独立验证指定gold测试，不借此放宽分母或验收标准。
2. **P1 可验证状态与有效行动**：保留确定性的修改/测试记录，防止摘要把计划写成完成；对不变内容的重复读取提示已有证据和下一项待验证假设；已定位后要求形成最小补丁假设或明确缺失证据，不能强迫无依据修改。测试已满足指定标准时引导提交。
3. **P2 才做预算对照**：先在相同36次预算下对比工具修复版本；再对固定开发诊断子集做36/72等预算消融，将摘要预算政策单列。不把所有修复、上下文及预算变化混为一次实验，也不提前用300题保留测试集调参。
4. RL/OPD不作为当前第一步：先修复已知错误反馈与工具契约，再判断剩余失败是否值得训练。源码/模型能力、工具健康、策略效率分开评估。

本次只新增离线审计器和分析记录，没有修改Agent运行逻辑，没有请求模型或启动任务测试/长评测。审计器5项针对性测试通过，Ruff通过；103项冻结源码指纹未变化，见[验证记录](artifacts/scale_repair_20261025/trajectory_audit_20261025/validation.json)。复算命令如下，输出目录必须不存在：

```bash
python scripts/audit_scale_trajectories.py \
  --run outputs/scale_4gpu_repair_20261025 \
  --output /tmp/sca_trajectory_audit_new
```

## 23:42 UTC起：工具修复与v7状态实现，等待用户启动长评测

上述后续改进中的P0、P1已实现，尚无新的模型成功率。历史100题的11/100、17/100及其103项冻结源码保持不变。新增两套运行目录，按以下顺序比较，继续使用同一100题开发集；没有扩大调用预算，也不使用300题保留测试集调参。

| 新运行目录 | 策略 | 用途 |
|---|---|---|
| `outputs/scale_toolsfix_dev_20261025` | `--agent-mode v6` | 修复工具、增加真实预检，保留原v6策略和模型摘要预算；重新比较baseline/Coder |
| `outputs/scale_v7_dev_20261025` | `--agent-mode v7 --arms coder` | 在相同工具修复上，单独检查状态记录、确定性压缩及行动提示对Coder的作用 |

### 已完成的实现

- worker不再依赖Python3.7的future annotations或Python3.9的`Path.is_relative_to`，沿用每道任务原来的解释器。读取反馈增加文件SHA256；路径、符号链接越界、`.git`及保护测试限制保持有效。
- `sandbox.probe()`实际运行glob/grep/read/write/edit及v2反馈，使用任务环境中的独立小型临时仓库，完成后清理。正式控制器在启动评测模型前检查各不同任务运行环境；每题执行前仍做即时检查。旧gold预检通过不能代替新工具检查。检查失败即停止，不静默删题。
- v7新增`repair_state`：成功修改必须有实际`changed:true`和有效文件哈希；测试结果绑定变更阶段，后续编辑或shell操作使旧结果失效。独立最终验证结果不会反向写入Agent已经看到的状态。
- v7用确定性的工具事实和标明来源的工具片段压缩上下文，不请求模型生成状态摘要；相同参数及相同内容的重复读取只给提示，仍正常执行。长期未确认修改时提示形成修复假设或指出缺少的证据，指定测试通过时提示提交，不强迫修改源码。
- 输入完整性检查增加`validation_completed/validation_total/validation_task`进度；`tool_preflight`显示环境检查进度。原有仓库完整性校验仍保留。

两版都保留36次总模型调用、36轮、4096输出token及8192上下文上限。v7不调用模型摘要，因而可将原先用于摘要的调用用于决策；这是v7整体策略变化的一部分，不能声称仅隔离了“记忆准确性”一个变量。本轮没有模型权重训练、RL或OPD。

### 短验证

真实OCI环境Python3.6.13、3.8.20、3.9.21均通过19项工具检查，最终版本合计约3.9秒；调用仓库、任务元数据保持不变。见[最终真实环境回执](artifacts/v7_runtime_compat_20261025_final/report.json)。第一次受执行沙箱的NETLINK权限限制而失败的回执单独保留在`artifacts/v7_runtime_compat_20261025`，随后在宿主完成验证，没有把沙箱失败记录覆盖为成功。

23:55 UTC最终全套测试264项通过（pytest报告67.91秒），修改涉及的Python文件Ruff通过。两套新源码已冻结，历史103项源码指纹保持不变；见[整体验证与版本记录](artifacts/scale_repair_20261025/tool_state_fix_20261025/validation.json)。未启动GPU长评测，因此目前没有修复后的成功率。

### 先启动工具修复后的100题对照

以下为长任务，由用户启动。baseline使用所选第一张卡，Coder使用四卡张量并行，两组顺序执行。选择的卡不满足空闲显存要求时等待，不停止其他任务。必须使用新的输出目录，不能把新代码续入旧11%/17%实验。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
systemd-run --user --unit=sca-toolsfix-dev-20261025 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/run_scale_4gpu.sh resume \
  --output outputs/scale_toolsfix_dev_20261025 \
  --data datasets/generated/swebench_full_20261024 \
  --agent-mode v6 --arms baseline,coder \
  --gpus 2,5,6,7 --allow-shared-gpus --wait-for-gpus \
  --gpu-memory-utilization 0.8 --gpu-headroom-mib 1536 \
  --resume-interrupted --trace-server-signals
```

```bash
journalctl --user -fu sca-toolsfix-dev-20261025.service -o cat
```

另一个终端查看计数：

```bash
watch -n 15 'bash scripts/run_scale_4gpu.sh status --output outputs/scale_toolsfix_dev_20261025'
```

启动后依次为`checking_frozen_inputs`（100题输入完整性）、`tool_preflight`（实际文件工具）、模型加载和协议检查、`evaluating`。前两个阶段不调用模型。最终对照报告为`outputs/scale_toolsfix_dev_20261025/dev/comparison.md`。

### P0结束后，再单独测试v7的Coder

复用相同数据与数值预算，但新输出目录和新模式避免混合结果。不要与上一条任务同时启动。

```bash
systemd-run --user --unit=sca-v7-dev-20261025 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/run_scale_4gpu.sh resume \
  --output outputs/scale_v7_dev_20261025 \
  --data datasets/generated/swebench_full_20261024 \
  --agent-mode v7 --arms coder \
  --gpus 2,5,6,7 --allow-shared-gpus --wait-for-gpus \
  --gpu-memory-utilization 0.8 --gpu-headroom-mib 1536 \
  --resume-interrupted --trace-server-signals
```

```bash
journalctl --user -fu sca-v7-dev-20261025.service -o cat
watch -n 15 'bash scripts/run_scale_4gpu.sh status --output outputs/scale_v7_dev_20261025'
```

v7只运行Coder，因此报告在`outputs/scale_v7_dev_20261025/dev/coder/report.md`，不会生成baseline/Coder双模型`comparison.md`。结束后应与P0的Coder逐题配对，统计新增修复、退步、无补丁比例、调用数及测试情况；不能用单一总成功率推断是哪项机制生效。

## 2026-10-26 08:32 UTC：baseline结束，Coder启动等待超时

`sca-toolsfix-dev-20261025.service`于10月25日23:57:25启动，10月26日08:32:33以退出码1结束。82个不同任务环境的工具检查已经覆盖100题。baseline报告为100题结束、10题修复成功、90题unresolved；Coder尚未开始正式任务，双模型对照尚未完成。既有baseline结果保留，可在相同输出目录恢复，不应重跑已完成题目。

直接失败点是Coder服务未在1200秒内响应健康检查。失败回执记录其清理前仍存活，状态为`D (disk sleep)`、RSS约677MiB、物理读取约536MiB，主机I/O PSI `full avg10=83.20%`。Coder服务日志只有attempt标记，尚无vLLM启动信息；这支持启动期间主机I/O阻塞，不能据此确定已经进入权重加载或NCCL初始化。

信号记录明确SIGTERM发送者为控制器PID3084217，时间1793003546.509，是超时后的主动清理；此次不是已观察到的earlyoom终止。证据见[启动失败回执](../outputs/scale_toolsfix_dev_20261025/logs/dev_coder_startup_failure_1793003546397692882.json)与[信号记录](../outputs/scale_toolsfix_dev_20261025/logs/dev_coder_signals_1792994983814322369.log)。

09:13 UTC短采样显示项目所在`/dev/sdb1`的磁盘利用率约99%–100%，平均读取等待110–127ms；系统I/O压力仍高。未定位到故障瞬间具体占盘进程，不能将其直接归因于VS Code或某个训练。当前状态顶层`arm=baseline`、`protocol_smoke_step`和旧`startup_progress`为上一阶段遗留字段，实际失败应看`model_arm=coder`及`model_startup_failure`中的新数据。未修改冻结Agent、模型、预算或停止其他程序，也未启动长任务；应先缓解存储争用，再从原输出恢复Coder，单纯延长等待不能保证启动成功。

## 2026-10-26：定位读盘进程及准备迁至/data

磁盘映射为`/home → sdb`（机械盘）、`/data → sda`（机械盘）、`/`及`/tmp → sdc`（SSD）。空间充足不代表I/O空闲。服务器已安装以下工具，可由有sudo权限的用户查看全机，而非只看当前账号：

```bash
# 每秒按进程和设备汇总；重点看DISK=sdb、D=R的行。
sudo biotop-bpfcc -C 1

# 常规进程I/O监视，左右方向键选DISK READ排序。
sudo iotop -oP -d 1

# 找到PID后，按单个进程查看命令与文件；12345替换为实际PID。
ps -p 12345 -o user,pid,ppid,stat,etime,args
sudo lsof -nP -p 12345 | grep '/home/'
```

`biotop`的`Kbytes`本应为该采样间隔的读写量，但随后用户在本机观察到该列全部为0：本机旧版本在完成回调读取请求剩余长度，不能用它按字节排名。下文的本地汇总脚本在请求开始时保存长度。`AVGms`是请求发出至完成的平均延迟，不包含完整的排队等待。`lsof`显示打开/映射的文件，需结合持续读盘活动判断；仅有打开记录不证明当前正在读取。内核代办I/O可能显示为内核线程，不能把单次采样当作故障时的完整归因。按PID查询可避免对整个`/home`递归扫描。

### 已准备的迁移范围

新增`scripts/migrate_coder_storage.py`，使用系统`/usr/bin/python3`运行。默认将三个目录复制至独立的`/data/zjr/sca_coder_runtime_20261026`，原逻辑路径改为软链接，原目录改名保留备份，不重装软件、不改回执或冻结Agent：

| 原目录 | 新物理目录 |
|---|---|
| `.model-env` | `/data/zjr/sca_coder_runtime_20261026/model-env` |
| `.coder-env` | `/data/zjr/sca_coder_runtime_20261026/coder-env` |
| `.cache/models/Qwen3-Coder-30B-A3B-Instruct` | `/data/zjr/sca_coder_runtime_20261026/model` |

权重回执包含22个文件、61,075,328,222字节，约56.88GiB；两个环境另外占空间。`.coder-env`的解释器指向`.model-env`，两个环境必须一起迁，保留原前缀软链接可维持pyvenv.cfg、入口脚本及冻结运行时的兼容性。VL-8B权重本来就在`/data/zjr/models`。

迁移使用可续传rsync，环境做checksum比较，模型按既有SHA256回执验证；全部复制校验通过后才切换路径，再以禁用GPU的短导入检查核对实际环境、torch/vLLM路径和版本。失败会尝试恢复原目录，保留目标副本供重试；重复执行完成的迁移不会用旧备份覆盖已使用的环境。`rollback`只恢复本次迁移的路径，不删除目标副本。

新增项目迁移独占锁及评测共享锁，现有控制器锁和宿主活跃进程检查同时参与。迁移期间不要另起旧冻结脚本或其他直接使用这两个环境的命令；新评测控制器会拒绝与迁移同时启动。

### 用户启动长复制，助手未开始真实迁移

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
/usr/bin/python3 scripts/migrate_coder_storage.py plan

systemd-run --user --unit=sca-coder-migrate-data-20261026 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/python3 scripts/migrate_coder_storage.py apply
```

查看复制百分比和检查阶段：

```bash
journalctl --user -fu sca-coder-migrate-data-20261026.service -o cat
```

另一个终端看状态：

```bash
watch -n 5 '/usr/bin/python3 scripts/migrate_coder_storage.py status'
```

`completed`表示迁移和CPU导入检查完成，不表示Coder评测完成。原始副本先保留，待正式Coder启动成功后再考虑清理。需恢复原路径时，在无评测运行的情况下执行：

```bash
/usr/bin/python3 scripts/migrate_coder_storage.py rollback
```

迁移完成后沿用`outputs/scale_toolsfix_dev_20261025`恢复，baseline已有100条结果会跳过。以下仍为长任务，迁移脚本不会自动执行：

```bash
systemd-run --user --unit=sca-toolsfix-data-retry-20261026 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/run_scale_4gpu.sh resume \
  --output outputs/scale_toolsfix_dev_20261025 \
  --agent-mode v6 --arms baseline,coder \
  --gpus 2,5,6,7 --allow-shared-gpus --wait-for-gpus \
  --gpu-memory-utilization 0.8 --gpu-headroom-mib 1536 \
  --resume-interrupted --trace-server-signals
```

这次仅迁移环境和Coder权重，项目代码、数据集、输出及其他缓存仍在原位置；这些目录后续的读写仍可能受`/home`负载影响。`/data`也是机械盘，本方案通过分散磁盘争用改善启动条件，不保证消除所有I/O延迟。

准备阶段验证：115项迁移/锁/控制器测试通过（9.55秒），Ruff通过；真实宿主只读空闲检查通过。仅运行小文件测试及计划检查，实际迁移状态为`not_started`，未启动GPU或长复制。见[验证记录](artifacts/scale_repair_20261025/storage_migration_20261026/validation.json)。CPU导入日志保存在迁移目标的`cpu_import_check.log`，失败或超时也保留输出；切换失败不会只留下不明的exit1。

用户随后启动迁移并成功完成：`sca-coder-migrate-data-20261026.service`的Result为success、退出码0；三条原路径均已指向`/data/zjr/sca_coder_runtime_20261026`。CPU检查确认环境及torch/vLLM模块实际来自`/data`，版本分别为torch2.8.0+cu128、vLLM0.10.2、transformers4.55.2。原目录备份保留；迁移没有自动启动GPU或评测。完成回执见[已验证迁移状态](artifacts/scale_repair_20261025/storage_migration_20261026/completed.json)。下一步使用上面的`sca-toolsfix-data-retry-20261026`命令续跑原100题对照，跳过已完成baseline，启动四卡Coder；长评测交由用户启动。

### 按磁盘汇总读取占比

新增`scripts/disk_io_share.py`：仅在内存中提取本机已安装biotop的探针，开始时保存请求长度，完成时使用原子计数；不改系统工具。采样结束后关闭探针，再按TGID合并线程名称并汇总指定磁盘的读取量，未知PID及未显示的行仍计入分母。另按账号汇总，并以磁盘扇区统计作近似覆盖核对。

```bash
sudo /usr/bin/python3 \
  /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/disk_io_share.py \
  --disk sdb --seconds 30
```

- `READ%`：该进程读取字节/本次观察到的sdb全部读取字节。
- `R_IO%`：该进程读请求次数/本次观察到的sdb全部读请求次数。
- `MiB/s`：窗口平均读取吞吐，`AVGms`为请求发出至完成的平均延迟。
- `trace/diskstats`：探针字节与同一窗口磁盘计数的近似比值；窗口边界、在途请求及合并/重发等会影响匹配。覆盖异常时不能声称已完整归因。

这些比例不等于磁盘忙碌时间占比，不证明因果或某进程应被终止。若有读请求但长度仍全为0，脚本报统计无效，不继续给虚假比例。8项CPU统计/计数修正测试通过，Ruff通过；助手尝试5秒实际采样时`sudo -n`提示需要密码，未加载内核探针，故实际BCC编译和采样仍需用户在终端运行确认。未修改进程或启动模型。

## 2026-10-26 21:31 UTC完成、2026-10-27离线复盘：P0工具修复对照

`sca-toolsfix-data-retry-20261026.service`成功结束，控制器记录恢复开始于2026-10-26 12:52:19 UTC，完成于21:31:12 UTC，本次恢复历时8小时38分53秒。日志中的`Consumed 1d 4h 51min`是累计CPU时间，不能用作实际等待时间。baseline已有结果被保留，Coder恢复后补齐；两臂最终各有100题结果。控制器完成并不表示每题正常退出，下面单列4个运行异常。

### 成绩及比较范围

| 模型 | 修工具前 | 本轮P0 | 变化 |
|---|---:|---:|---:|
| Qwen3-VL-8B baseline | 11/100（11%） | 10/100（10%） | -1个百分点 |
| Qwen3-Coder-30B-A3B-Instruct | 17/100（17%） | 19/100（19%） | +2个百分点 |

旧运行是`outputs/scale_4gpu_repair_20261025`，新运行是`outputs/scale_toolsfix_dev_20261025`。核对确认两次100题清单、输入哈希、公共配置、模型回执和Coder环境回执一致；旧103项、新111项冻结源码均未变化。两轮都使用v6、36次决策与模型摘要共享调用预算、最多2次模型摘要、4096输出上限和8192 Agent上下文。没有SFT、RL、OPD或模型权重更新；P0包含文件worker兼容修复、工具检查及返回信息等工程变更，反馈文本变化也可能改变后续轨迹，不能将全部差异只归因于Python兼容修复。

这是6个仓库的SWE-bench Full开发集100题，使用原始文本输入及本地官方镜像适配评分。300题保留测试集尚未运行，也没有本轮截图端到端成绩；不能作为官方排行榜成绩或未见测试泛化结论。

本轮Coder相对本轮baseline保留全部10个成功，额外修复9题，退步0题，共同失败81题。Coder相对自己的旧结果则是保留15、增加4、退步2、共同失败79；单次净增2题不足以证明稳定收益。

| Coder仓库 | 修工具前 | 本轮P0 | 成功题数变化 |
|---|---:|---:|---:|
| marshmallow | 5/9 | 3/9 | -2 |
| pvlib | 4/18 | 4/18 | 0 |
| pydicom | 0/17 | 3/17 | +3 |
| astroid | 5/22 | 5/22 | 0 |
| pyvista | 1/13 | 1/13 | 0 |
| sqlfluff | 2/21 | 3/21 | +1 |

新增修复为`pydicom-1050`、`pydicom-1192`、`pydicom-1228`、`sqlfluff-905`；退步为`marshmallow-1252`和`marshmallow-1524`。完整任务ID及逐题配对见下方分析JSON。

### 工具修复确实生效，但不能直接转化为同等数量的成功

本轮200条轨迹的运行时文件工具探针全部通过19项检查；完整事件审计中，旧的Python3.6 future annotations语法错误和Python3.8 `Path.is_relative_to`错误均为0。旧轮两臂各有同样15个pydicom任务受这两类错误影响，分别累计447/131次失败工具调用。

这15题修复工具后baseline成功1题、Coder成功3题，其他题仍未解决。Coder的3个pydicom新增成功都来自这个集合，说明修复消除了真实阻碍；剩余任务还需要正确定位、修改和测试，不能把15个环境受影响任务等同于15个可直接恢复的成功。

| Coder轨迹指标 | 修工具前 | 本轮P0 |
|---|---:|---:|
| 零补丁失败题数 | 29 | 22 |
| 成功的显式edit_file/write_file调用数 | 179 | 229 |
| 共享调用预算耗尽题数 | 87 | 82 |
| 预算耗尽但最终独立验证成功 | 4 | 5 |

显式修改调用数按工具成功反馈计数，不代表正确修改次数。82次预算耗尽中有5题修复成功、77题未获成功，其中1题随后补丁导出超时；不能把82次预算耗尽全部当成失败。22个零补丁失败中，19题预算耗尽、2题API超时、1题提前提交空补丁；21题没有显式edit_file/write_file请求，另1题编辑匹配失败。没有显式编辑请求本身不排除通过bash修改，零补丁结论以最终diff为准。

Coder本轮3191次工具调用中，read_file/grep/glob共2135次，占66.9%；这不表示全部阅读都无效，但下面的具体轨迹证明存在耗尽预算而未转入实现的问题。本轮3209次决策调用、196次模型摘要调用，共3405次；不能把上下文压缩事件数当成模型摘要调用数。

### 人工复核的失败与退步案例

| 任务 | 实际轨迹与最终结果 | 对下一步的启示 |
|---|---|---|
| `pydicom-816` | 文件工具恢复正常；23次read_file、9次grep、2次glob，用完34次决策及2次摘要，没有修改和测试，补丁为0 | 需要让已有定位证据推动下一步行动，减少纯阅读循环 |
| `pvlib-1469` | 25次read_file、6次grep、3次glob，连续翻阅modelchain.py；预算耗尽，无修改及补丁 | 扩大调用预算之前先检查定位与行动的衔接 |
| `marshmallow-1252` | 旧轮成功，本轮只新增3个复现/诊断脚本，没有修改生产源码；目标失败测试0/1通过，原通过测试37/37保留 | 非空diff和诊断脚本跑通不能替代实际修复 |
| `marshmallow-1524` | 旧轮成功，本轮修改Email校验，但遗漏URL相关路径；目标失败测试2/5通过，剩余3个URL换行测试仍失败，原通过测试127/127保留 | 修复覆盖不足，需要跟踪完整问题要求和测试证据 |

两题退步都有具体轨迹原因，不能简单解释成随机波动，也不能由单次轨迹断言工具修复必然伤害这些任务。成功案例中`pydicom-1228`、`sqlfluff-905`最终验证通过但仍用尽预算，说明停止/提交策略也值得检查。

### 4个运行异常保留在100题分母

| Coder任务 | 异常 | 原结果归类 |
|---|---|---|
| `marshmallow-2123` | 模型请求APITimeoutError | unresolved / model_error |
| `pydicom-901` | 模型请求APITimeoutError | unresolved / model_error |
| `pvlib-1296` | 模型请求APITimeoutError | unresolved / model_error |
| `sqlfluff-3435` | 补丁导出时git add超过30秒，未进入最终独立验证 | error / patch_export |

Coder合计19条verified、80条unresolved、1条error；其中3条unresolved是API超时。当前恢复逻辑识别取消或APIConnectionError，APITimeoutError仍计为已结束，故控制器能报告completed。这里保留原始19/100，不删除异常分母、不选择性重跑后覆盖原成绩。仅凭当前结果不能确定3次API超时的根因；git add超时也不能单独证明是磁盘故障。

### 后续优先级与本次审计范围

1. 在新运行前处理API超时的分类、有限重试/恢复和补丁导出超时，保留原尝试与原因，避免服务或宿主异常混入正常模型失败。此项是后续建议，本次没有修改控制器或重跑题目。
2. 接着在相同100题、同为36次总模型调用下验证已经准备的v7，使用独立输出目录。其主要改动是基于工具事实的状态记录、确定性上下文整理和重复读取提示，正好对应本轮观察到的部分问题。v7不占用模型摘要调用，也会改变可用于决策的预算份额，应把结果解释为整套策略对比，不能只归因于记忆模块。
3. v7结果出来后逐题比较新增修复、退步、零补丁、预算耗尽及运行异常，再决定是否扩充预算或训练。当前证据不足以声称模型达到硬件条件下的能力上限；也不支持直接承诺RL/OPD能解决这些失败。300题保留集继续留作方案固定后的确认。

截至本次复盘，`outputs/scale_v7_dev_20261025`只有已准备的运行材料，没有status.json或新评测结果。本次只离线审计200条结果和完整事件，重算配对与失败分类，核对输入哈希及冻结源码；没有启动模型、GPU评测、任务测试或训练，也没有改动原始成绩。

证据入口：

- [本轮原始对照报告](../outputs/scale_toolsfix_dev_20261025/dev/comparison.md)及[机器可读报告](../outputs/scale_toolsfix_dev_20261025/dev/comparison.json)。
- [新旧逐题配对、4个异常和人工案例测试计数](artifacts/scale_repair_20261025/toolsfix_analysis_20261027/comparison_analysis.json)。
- [完整事件审计与输入指纹](artifacts/scale_repair_20261025/toolsfix_analysis_20261027/traces/analysis.json)及[逐题CSV](artifacts/scale_repair_20261025/toolsfix_analysis_20261027/traces/tasks.csv)。

## 2026-10-27：架构复核与实验路线调整

用户指出数天等待后仅净增2题，要求结合相关工作判断该修实现还是加策略。本次只读审查核心执行代码、配置及已有小型结果文件，检索一手研究资料；没有启动模型或修改冻结运行。当前runner/context/tools/models/config/progress/worker与P0冻结版本一致。结论是：局部实现缺陷仍需修，但尚未发现能解释81题失败的单个大bug。下一轮应优先引入成熟Agent框架对照、校准上下文与预算，单独继续跑v7不足以回答模型和框架分别限制了多少能力。

### 对当前19/100评测口径的补充说明

此前的“独立验证”指在干净副本重放补丁，不代表测试对Agent隐藏。`scale_data.py:244`在起始仓库安装官方`test_patch`，257–258只读保护这些文件；`runner.py:129–137`的`run_tests`会返回官方FAIL_TO_PASS/PASS_TO_PASS失败ID和resolved，最终仍用同套测试验证（201–213）。没有发现参考生产代码补丁进入Agent仓库，但参考测试可读、可执行，且提供目标测试判分反馈。

因此原19/100是**已知目标测试反馈条件下的本地开发修复成绩**。原结果有其意义并原样保留，但此前仅标注“官方镜像适配、非排行榜”还不够具体；不能将其等同于隐藏参考测试的标准SWE-bench rollout。后续标准协议应让Agent仅访问原始仓库已有测试及自己生成的复现测试，将参考test_patch和判分留给最终评分副本。协议变化需要独立目录、所有比较臂重新建基线，新分数不能直接减去19%宣称策略增益。此项改善可比性，不承诺提高成功率。

### 已确认的实现缺陷与待验证的设计限制

| 层面 | 代码证据 | 处理原则 |
|---|---|---|
| 搜索结果不完整却未提示 | `worker.py:112,117,132–135`先枚举整个搜索目录，再排除.git等；最多200结果、每行300字符，无截断或剩余提示 | 提前剪枝、明确截断/缩小范围/分页；先用CPU接口测试验证，成功率影响未知 |
| 路径契约不一致 | OCI cwd为/testbed（`runtime.py:111`），文件worker仅接受/workspace绝对路径；系统与bash描述却声称cwd=/workspace | 统一路径语义；v7仅改系统字符串，尚未完整解决。不能解释大部分失败 |
| 异常恢复缺口 | 3次APITimeoutError未按连接中断处理；`repository.py:20`统一30秒git超时造成1次导出失败 | 分类并保留尝试，限定恢复；用阶段独立超时，避免重跑整个修复轨迹 |
| 有效代码上下文过小 | `configs/scale-4gpu.json:13–20`为8k估算token/36总调用；`context.py:64–94`在70%约5.7k开始压缩，旧工具内容截1400/1000字符 | 这是配置和信息保留策略，不是预算计数bug。优先测经过容量校验的32k服务及合理Agent输入余量 |
| 单轨迹缺乏明确阶段产物 | 只有自由工具循环；最终仅导出当前workspace；没有候选补丁选择、完整需求验收清单 | 需通过阶段指标选择策略，不宜直接堆多Agent或大规模采样 |

36次共享预算和最多2次摘要按设计正确执行；抽查失败ID及resolved位于测试反馈前部，不能笼统说模型没有看到测试结果。没有证据表明当前qwen3_coder parser是主因，固定vLLM0.10.2确有相应实现，不应拿最新版命名反推旧版错误。

v7主要是可靠工具记录和重复提示，仍继承8k压缩（`context.py:98–117`）；重复检查只识别相同参数/相同证据（`progress.py:59–76`），不能识别连续换offset翻页。所有edit/write成功都算进展（86–92），写诊断脚本也会重置提醒；记忆主要保留路径、区间、哈希和测试计数，尚不能记录完整根因假设和未满足需求。因此不把v7当作已解决定位、实现与验证衔接的方案。

### 数天等待与实际推理时间分开核算

100份Coder最终result记录的任务跨度为2026-10-26 16:13:42至21:29:43 UTC，约5小时16分；单题平均373.8秒、中位349.8秒，最长795.2秒。单题时间合计10.384小时，除以墙钟约1.97，说明两个任务槽位基本填满。恢复控制器到首题相隔3小时21分，工具探针阶段约5分半；模型服务真正attempt约16:02，不能把整个等待归为模型加载或工具预检。历次故障、迁移、准备和资源等待共同拉长了用户经历的数天周期。

`run_scale_4gpu.py:74–76`将单个Coder模型TP=4，max-num-seqs=2；960只允许1或2个任务worker。四卡并不等于四个独立修复Agent。后续可独立测2/4/8任务并发的吞吐和显存，先验证再选择；不能承诺并发翻倍就线性提速。优先复用模型服务、有效预检回执，并按AGENTS将新数据/输出/缓存放/data。现有81个rootfs及testbed、100份prepared repo、20份fallback_source均在本地，无需为新协议重新下载整套模型与环境。

### 相关工作如何指导下一轮

- [mini-SWE-agent官方](https://mini-swe-agent.com/latest/)强调简单bash执行及线性历史，适合作为独立成熟基线。固定实际版本；v2已采用原生工具调用，不沿用旧版“无tool calling”表述。采用它并不预设必然胜过当前框架。
- [Qwen3-Coder-30B-A3B官方模型卡](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)给出原生262144上下文及专用函数调用格式，说明8k是本项目主动限制，不是模型能力上限。实际服务长度和并发仍需按本机显存验证，不直接启用256k。
- [SWE-agent原论文](https://arxiv.org/html/2405.15793v3)通过实验研究搜索、编辑反馈和历史处理，并观察到逐项穷举搜索耗尽预算；这支持检查接口设计，不能把其旧模型收益迁移成我们的预期分数。
- [Agentless论文](https://arxiv.org/html/2407.01489v2)及[实现](https://github.com/OpenAutoCoder/Agentless)将文件→函数→编辑位置定位、补丁生成及测试筛选分开。值得借鉴的是可检查的阶段产物和先在原代码复现的验证流程，不立即复制大量候选采样。

建议下一轮做一次能区分责任的比较：先收敛必要接口问题和测试可见性，固定模型、100题清单、初始仓库、可见信息及评分器，比较当前框架与固定版本mini-SWE-agent。两者采用相同经过容量验证的上下文、输入/输出token及墙钟预算，并报告实际消耗；调用次数上限可同设，但一次工具动作的信息量不同，不能仅凭同36步声称成本公平。少量跨仓库烟测只用于工程验收，正式结论仍用完整100题，保留300题独立测试。

若成熟框架明显更好，优先替换通用执行核心，保留本项目截图证据读取、隔离运行、日志和独立评测资产；若接近，再做上下文/预算敏感性对照、定位与修复策略实验。当前19%仅作为旧协议历史记录。后续增加定位命中（仅离线对照参考修改位置）、首次生产源码编辑、只写诊断脚本、复现有效性、局部修复/回归失败等指标，选择真正阻碍成功的阶段。RL/OPD应在执行及反馈协议稳定、有可靠轨迹后再推进。

## 2026-10-27：mini-SWE-agent接入与新协议对照入口

已实现两套执行框架的独立对照。每题仅运行一个Agent，没有互相委派、协作或传递答案。CoreCoder沿用v6执行框架；mini使用[上游v2.4.6固定提交](https://github.com/SWE-agent/mini-swe-agent/tree/a83fcae82d2a08f0ee0c688f9d137b3566c097f8)的真实DefaultAgent，源码及MIT许可证保留在vendor，文件哈希记录于`MINI_UPSTREAM.json`。通过Model/Environment适配器连接现有Qwen API与隔离沙箱；没有使用上游会在宿主执行命令的LocalEnvironment。

主要文件是`src/screenshot_agent/harness.py`、`harness_runner.py`、`hidden_eval.py`、`scripts/run_harness_compare.py`及`configs/harness-compare.json`。mini保留上游bash工具、线性历史、格式错误反馈及推荐工作流；适配提交指令，由外部评测器统一导出workspace diff，而非接收Agent自报的patch.txt。提示中告知无网络、依赖预装。CoreCoder保留自己的文件工具和RepairContext，因此比较的是两套框架的整体行为，不能把差异仅归因于某一个工具。

### 固定比较范围和资源

| 项目 | 两臂设置 |
|---|---|
| 数据 | 原P0同一批100题SWE-bench Full开发集，6个仓库，原始问题文本 |
| 模型 | 同一Qwen3-Coder-30B-A3B-Instruct权重、vLLM0.10.2、temperature=0 |
| 模型服务 | TP=4，共用一次启动；默认2题并发，按题交错投放两臂 |
| 上下文 | 服务32768 token；完整输入含工具schema最多24576；单次最多输出4096 |
| 调用 | 每题最多80次模型请求（含摘要和重试）、200次工具；CoreCoder最多2次摘要 |
| 总token | 每题输入150万、输出10万；每次请求先预留输出额度，失败请求按上限记账 |
| 时间 | 每题2400秒；单命令最多600秒；单API请求最多300秒；最终评分独立计时 |
| 可见信息 | 相同原始问题、干净仓库、已有公共测试；参考测试和判分仅在最终评分时可见 |
| 输出 | 200条轨迹、补丁、退出原因；固定100题分母、配对差异、实际调用/token/时间 |

这不是新旧19%之间的直接增益实验。新协议隐藏参考测试，且资源预算从旧8k/36次提高；需要在新协议内比较CoreCoder与mini。300题保留测试集继续不动，也没有SFT/RL/OPD或新的截图端到端成绩。

API连接/请求超时最多额外尝试一次，两次都占共享调用/token预算；已结束的运行异常不会自动挑选重跑。显式恢复只处理被取消或未产生最终结果的中断尝试，原目录归档至`interrupted_attempts`。搜索工具提前排除.git等目录，返回200项上限/行文本截断提示；/testbed通过真实同目录校验兼容文件工具。git操作默认超时由30秒放宽到180秒。旧冻结运行和成绩保留。

### 隐藏参考测试与数据位置

prepare阶段复制旧已预检仓库，移除`.sca_eval.sh`并反向撤销参考test_patch；随后重新应用两者，必须与旧预检的完整仓库哈希一致，再次撤销后也必须回到干净哈希。借此继承旧的buggy失败/gold通过证据，同时验证复制和拆分，没有逐题重复长时间环境测试。每次正式运行前检查新输入和配置指纹。

参考patch、判分脚本和benchmark元数据在repo之外，Agent命名空间中不可访问；Agent的run_tests仅运行公共测试。任务结束后先导出补丁，在独立副本重放，再安装参考test_patch并评分。公共测试目录只读保护按公开文件名确定，不从秘密测试位置生成提示。评分仍是本地官方OCI镜像适配协议，不能作为官方排行榜结果。

新运行根目录固定为：

```text
/data/zjr/screenshot_code_agent/harness_compare_20261027/
  data/tasks/           干净仓库与沙箱外评分材料
  cache/task_runtimes/  仍在home的OCI rootfs/兼容依赖副本
  cache/、tmp/          模型运行缓存与临时文件
  runtime/             冻结执行源码
  corecoder/、mini/     各臂配置、轨迹、补丁、结果
  logs/                复制、模型服务和任务日志
  report.json、report.md
```

代码和小文档在Tracker，权重继续复用`/data/zjr/sca_coder_runtime_20261026/model`。prepare只复制本次所需的旧/home运行环境到/data，用低I/O优先级rsync、完成回执和已完成目录的硬链接去重；源目录保留，传输中断可接着复制。该复制可能耗时，状态`copying_runtime_to_data`表示正在准备，不表示模型已启动；status会显示当前源/目标与复制进度。不会重新下载整套模型或镜像，也不会迁移其他项目。

### 用户启动和监视

依赖与固定上游源码已经准备好。需要在新checkout重装时使用`bash scripts/setup_harness_compare.sh`。下方all命令顺序完成数据准备、冻结、输入校验、GPU等待、模型工具协议检查、并发长输入检查及两臂各100题评测。默认卡号2、5、6、7，按启动时空闲情况调整；程序检查显存，不会停止别人的进程。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
systemd-run --user --unit=sca-harness-compare-20261027 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/run_harness_compare.sh all \
  --gpus 2,5,6,7 --workers 2 \
  --allow-shared-gpus --wait-for-gpus --resume-interrupted
```

```bash
journalctl --user -fu sca-harness-compare-20261027.service -o cat
```

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_harness_compare.sh status'
```

如想先只准备数据，把all换成prepare；准备完成再用新unit名执行run，其余参数相同。若被中断，同一输出目录使用新unit名重新执行all或run，保留`--resume-interrupted`。不要更改已封存运行的workers、协议或预算；容量检查失败需要改配置时使用新的`--output /data/zjr/screenshot_code_agent/...`，保持比较记录独立。

`--workers 4`或8是可选吞吐试验，并未实测，首次建议保持2。四卡指单个Qwen模型TP=4；workers是并发任务数，与多Agent协作无关。长输入检查只验证并发prefill与短输出能够完成，不保证最大输出情况下的速度或显存。正式结果生成后查看/data运行目录下的report.md；其中单题时间合计不是并行墙钟时长。

### 本轮验收与未运行部分

15项新增短测试通过，包括两框架使用脚本响应在真实沙箱中完成修复、参考信息仅在最终评分挂载、格式错误恢复、传输有限重试计费、固定分母及服务参数；另146项相关回归通过。Ruff与shell语法检查通过。真实marshmallow-2123输入的拆分往返校验通过；工具消息token计数为302，与服务环境Transformers按vLLM格式归一化后的结果相同。

这些验证没有调用真实模型。尚未执行批量运行环境复制、32k配置GPU容量检查、正式100题×2对照或长训练；没有新的成功率或提速结论。短测试数据在/data；[小型验收记录](artifacts/scale_repair_20261025/harness_compare_20261027/validation.json)记录上游版本、源码指纹和验证边界。

## 2026-10-27：CoreCoder工程修复流程

用户将当前目标明确为可展示、可使用的简历开发项目，优先复用工程能力。新增独立入口 `scripts/run_repair_workflow.sh`，默认输出 `/data/zjr/screenshot_code_agent/repair_workflow_20261027`。现有 CoreCoder、mini 对照、旧冻结源码和成绩保留。

实际功能：

- 输入仓库、问题、截图和测试命令；截图复用现有 CPU OCR 或配置的 VLM。OCR 默认不额外占 GPU；VLM 需要提供其模型、接口和凭据。Oracle 仅供有已知转录的演示/测试使用。
- 在隔离工作副本建立本地代码索引，返回相关文件、符号和行号。采用词法/标识符检索及文件、符号加权，无外部 embedding 服务；工具执行期间只查询内存索引，阶段结束后检查文件树并增量刷新。
- 单个 CoreCoder Agent 按定位、复现、修改推进。定位阶段只给读工具；步骤记录实际执行，不将自然语言“完成”当作测试成功。
- 多段 SEARCH/REPLACE 补丁先校验所有原始匹配位置，再逆序替换；拒绝歧义和重叠，保持 Python3.6 文件 worker 兼容以及测试路径只读保护。
- 初次修复后在独立副本重放补丁并跑用户指定公共测试。失败或空补丁最多反馈纠错两轮；分析解析失败回退到真实测试日志，不自动判通过。最终再从干净基线重放验证。隐藏参考测试只用于最后评分，不进入纠错反馈。
- 持久化工作区、阶段、使用额度和进行中的请求预留。恢复保留原修改，使用新对话重新观察当前 diff；已结束失败不会选择性重跑。任务、配置、源码或基线变化时拒绝覆盖原结果。最终参考评分开始后，恢复只能继续验证，不能回到 Agent。
- 每题输出 `report.html`、`patch.diff`、`delivery.zip`、`result.json`、调用轨迹；报告展示截图、识别内容、计划、diff、公共测试及失败分析，执行中每15秒刷新。交付包只含面向用户的报告、补丁、结果与截图；不打包完整仓库、私有评分输入或缓存。批量页面可筛选任务并跳转逐题报告。

借鉴来源和实际改写范围记录于 [WORKFLOW_UPSTREAM.json](../WORKFLOW_UPSTREAM.json)：CoreCoder-Agent 的计划/反馈结构、AnyCoder 的补丁操作、Aider 的仓库地图设计。AnyCoder 的 MIT 声明保存在 `third_party/AnyCoder.LICENSE`，同时进入冻结源码。没有训练模型，也没有声称实现 Aider 的图排序或语义向量检索。

### 先查看演示

已生成的固定演示：`/data/zjr/screenshot_code_agent/repair_workflow_20261027/demo/run/report.html`。下载 `delivery.zip` 解压后打开同级 `report.html` 可以连同截图和 diff 查看。输入是一段折扣计算报错，执行真实沙箱、补丁及测试；模型响应和截图转录预先给定，页面明确标为演示，不能计为真实模型修复成绩。

如需重新生成一个独立演示目录：

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
bash scripts/run_repair_workflow.sh demo
```

### 自己的仓库和截图

下面命令显式启动本地四卡 Qwen3-Coder 服务，CPU OCR 读取截图。替换仓库、图片、问题和测试命令；初版默认沙箱提供系统 Python，带额外依赖的仓库通过 `--task /data/.../task.json` 配置已有 runtime。所有修改在输出工作区进行，原仓库保留。

```bash
bash scripts/run_repair_workflow.sh repair --local \
  --output /data/zjr/screenshot_code_agent/my_repair \
  --repo /data/zjr/my_project --name bug_001 \
  --problem '根据截图定位并修复这个错误' \
  --image /data/zjr/inputs/error.png --reader ocr \
  --test-command 'python3 -m unittest discover -s tests -v' \
  --gpus 2,5,6,7 --allow-shared-gpus --wait-for-gpus
```

结果页在 `my_repair/requests/bug_001/report.html`，原始记录保留于同目录。已有兼容模型服务时去掉 `--local`，通过 `--config` 指向接口、模型和本地 tokenizer 的配置，并设置配置所指定的 API key 环境变量。Coder 本身是文本模型；截图使用 `--reader vlm` 时需另行配置 `vision`，不能把图片直接发给 Coder。

中断后用相同配置和端口恢复（已正常结束的失败不能覆盖重跑）：

```bash
bash scripts/run_repair_workflow.sh resume --local \
  --output /data/zjr/screenshot_code_agent/my_repair \
  --run-dir /data/zjr/screenshot_code_agent/my_repair/requests/bug_001 \
  --gpus 2,5,6,7 --allow-shared-gpus --wait-for-gpus
```

单题文件进度：`bash scripts/run_repair_workflow.sh status --run-dir /data/zjr/screenshot_code_agent/my_repair/requests/bug_001`。文件时间不能单独证明进程存活，后台任务同时看 journal。

### 四卡100题验收，由用户启动

复用原100题清单和现有模型。如果 mini 对照的 `prepared.json` 已准备好，验证后直接复用其只读输入；否则本入口按同一隐藏参考测试协议准备输入，尚在 home 的任务运行环境低优先级复制至 data，源目录保留。四卡是一个模型 TP=4，默认两题并发；本轮只跑新流程这一组100题，不自动接着跑两框架对照、截图实验或训练。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
systemd-run --user --unit=sca-workflow-dev-20261027 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/run_repair_workflow.sh all \
  --gpus 2,5,6,7 --workers 2 \
  --allow-shared-gpus --wait-for-gpus --resume-interrupted
```

```bash
journalctl --user -fu sca-workflow-dev-20261027.service -o cat
```

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_repair_workflow.sh status'
```

如中断，用新的 unit 名重启同一命令和输出目录，保留 `--resume-interrupted`。`prepare` 仅准备；`evaluate` 仅从已准备输入开始运行。新端口18856，模型窗口32k、完整输入24576 token、单次输出4096、每题共享80次模型调用/200次工具调用/2400秒，最终评分另计时间。计划和失败分析也计入调用额度，阶段额度与全局额度取先达到的限制。

完整结果为运行根目录的 `report.html` / `report.json` / `report.md`，固定100题分母，包含运行异常和零补丁。此100题验收使用原始问题文本，不测截图端到端成功率。与旧参考测试可见的19/100协议不同，不能直接宣称提升了多少点。新流程尚未进行真实模型验收，代码和短测试通过不代表模型成功率已经提高。

短验证：最终15项新增测试通过，另170项相关回归通过；Ruff和shell语法检查通过。新测试覆盖实际沙箱执行、两轮反馈接口、原始跨度补丁、受保护测试、引用隔离、用量/工作区恢复、评分开始后仅恢复验证、剩余截图续读、HTML转义、固定分母及只启动一个批量模型服务。测试和演示都使用预设模型响应，未启动GPU；记录见 [validation.json](artifacts/scale_repair_20261025/workflow_20261027/validation.json)。

### 自动等待任意四张空闲卡

2026-10-27 新增 `scripts/wait_and_run_workflow.sh`，等价于 `run_repair_workflow.sh all --gpus auto --wait-for-gpus --resume-interrupted`。运行后先完成 CPU 数据准备/校验，再进入选卡循环，满足条件后直接加载模型并执行一次100题验收。等待时不创建 CUDA 上下文或加载模型。已有冻结 Agent 源码保留，自动选卡在外层启动器完成。

默认扫描物理卡0–7，每10秒检查一次。每张候选卡必须没有计算进程、已用显存不超过512MiB、GPU利用率不超过5%，且满足原80%模型显存预算与1536MiB余量。至少四张同型号、同显存容量的卡各自连续两次满足条件才入选；不是只检查当前利用率是否为0。选中后获取本项目的 GPU UUID 文件锁，再读一次实际状态；如果又被占用，释放锁并继续挑选。锁仅协调本项目使用该入口的队列，不是系统级 GPU 独占预约，其他用户的程序仍可能随后占卡。

后台启动：

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
systemd-run --user --unit=sca-workflow-wait4-20261027 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/wait_and_run_workflow.sh
```

该入口自动选卡，不要再加 `--allow-shared-gpus`。可选参数如 `--gpu-candidates 0,1,2,3,4,5,6,7`、`--gpu-poll-seconds 10`、`--gpu-stable-samples 2`、`--workers 2`、`--output /data/zjr/screenshot_code_agent/...` 会直接传递给工作流。仍保持原来的32k模型服务窗口及正式预算，不做低显存降配。

```bash
journalctl --user -fu sca-workflow-wait4-20261027.service -o cat
```

```bash
watch -n 10 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_repair_workflow.sh status'
```

`waiting_for_idle_gpus` 表示正常排队，状态中的 `gpu_queue.devices` 包含各卡被排除的原因、连续空闲次数，`selected_gpus` 显示最终选中的物理卡号。NVIDIA 查询失败也会继续等待，并记录错误，不能把未知状态当成空闲。选卡记录写入 `/data/zjr/screenshot_code_agent/repair_workflow_20261027/gpu_allocations.jsonl`，协调锁位于 `/data/zjr/screenshot_code_agent/gpu_queue_locks`。相同输出目录只能运行一个控制器；结束后不会自动再次跑同一批任务。

取消排队或停止本次任务：

```bash
systemctl --user stop sca-workflow-wait4-20261027.service
```

需要恢复时使用新的 unit 名执行相同入口；保留已结束结果和已中断工作区。查询确认本服务器 `Linger=yes`，用户服务可在 SSH 断开后继续运行。本轮只执行短测试和单次只读 GPU 查询，没有启动等卡队列或模型。

验证：自动选卡、调用器和已有工作流共108项测试通过（23.52秒），覆盖临时空闲、占用反弹、查询失败、队列锁竞争、取消、实际物理卡号传递和旧启动器回归；全部模型启动均模拟。宿主机单次真实状态读取成功，八张卡均识别为有计算任务，符合当前实际。Ruff与shell语法检查通过。

## 2026-10-28：隐藏评分挂载修复与补丁重评

当前冻结工作流的隐藏评分因只读根文件系统无法创建 `/grader/eval.sh` 而退出；旧报告中的0通过并非有效成功率。源码已修复：只在独立评分时提供临时 `/grader` 挂载目录，评分文件仍只读；启动失败会明确标为环境异常。运行中任务的冻结代码保持原样，因此其旧报告仍需通过下面的独立重评纠正。

现在即可提交后台重评队列，默认每60秒检查原100题是否全部结束；满足条件后顺序执行 CPU 重评，不启动模型，也不改原始补丁或成绩。新结果及所有临时文件位于/data，CPU/I/O优先级已降低。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
systemd-run --user --unit=sca-workflow-regrade-20261028 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/regrade_workflow.sh run --wait-for-source
```

```bash
journalctl --user -fu sca-workflow-regrade-20261028.service -o cat
```

```bash
bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/regrade_workflow.sh status
```

`waiting_for_source` 表示等待原修复任务结束；`regrading` 表示重放原补丁并运行隐藏测试；`completed` 表示重评完成。正式新报告为 `/data/zjr/screenshot_code_agent/workflow_regrade_20261028/report.json` 与 `report.md`，只有 `valid_report=true` 才能作为完整且无环境异常的报告。若失败，先检查该题 `attempt_*/test.log`，不要反复重启模型。已重评题的源文件必须保持一致才能跳过复用；代码修复导致实现变化时应另选 `--output /data/zjr/screenshot_code_agent/...`。

本次仅运行38项短测试和一题真实补丁重评；单题成功启动并解析168条测试状态，补丁本身未通过。全量重评交由用户运行。原始运行不会被上述队列停止，也不向 Agent 提供参考测试反馈。

用户随后要求先评已结束题目。新增 `--completed-only`：启动时固定已结束且释放工作区锁的任务，后续新完成题不会混入分母；重启时仍复用原有题目清单。该模式与 `--wait-for-source` 互斥，应另选输出目录。选题冻结及原重评功能的5项测试通过。

已按用户要求启动 `sca-workflow-regrade-partial-20261028.service`，选中当时完成的44题，输出为 `/data/zjr/screenshot_code_agent/workflow_regrade_partial_20261028`。这是100题中的阶段性结果，不能当作100题最终成功率。

```bash
journalctl --user -fu sca-workflow-regrade-partial-20261028.service -o cat
bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/regrade_workflow.sh status \
  --output /data/zjr/screenshot_code_agent/workflow_regrade_partial_20261028
```

## 2026-10-28：工作流v2完整请求计数与可执行反馈

本版修复完整请求token压缩，缓存修改前的公共测试基线，并用固定断言脚本独立核验原问题。注册工具为`register_reproduction(script_path, expected_failure)`；脚本在原代码上应出现预期AssertionError，在补丁代码上应正常退出。回归结果和复现结果分别记录，阶段额度耗尽标为未完成，隐藏评分保持隔离。86项短回归通过；原pvlib-1026轨迹离线完整请求从26,106压缩至21,376 token。尚未进行新的模型评测。

长评测由用户启动。下列命令复用已有100题准备数据和模型权重，在新目录创建v2源码快照，等待连续两次满足条件的任意四张空闲卡，再启动TP=4、两题并发。新端口为18866，与原任务18856分开；所有输出、验证副本和缓存均位于/data。不要在旧v1输出目录恢复为v2，否则会触发版本不匹配检查。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
systemd-run --user --unit=sca-workflow-v2-20261028 \
  --working-directory=/home/zjr/Tracker/internship_project_lab/screenshot_code_agent \
  /usr/bin/bash scripts/run_repair_workflow.sh all \
  --output /data/zjr/screenshot_code_agent/repair_workflow_v2_20261028 \
  --gpus auto --wait-for-gpus --port 18866
```

```bash
journalctl --user -fu sca-workflow-v2-20261028.service -o cat
```

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_repair_workflow.sh status --output /data/zjr/screenshot_code_agent/repair_workflow_v2_20261028'
```

原任务不受上述源码修改影响；未执行这条新启动命令。新结果写入`/data/zjr/screenshot_code_agent/repair_workflow_v2_20261028/report.json`和`report.md`。比较时使用相同隐藏测试协议，不能直接拿旧可见参考测试的19/100当策略基线。

后续状态更新：用户已启动上述服务，2026-10-28约07:14 UTC确认处于`evaluating`，GPU为2、3、4、5，两题运行、98题待处理。启动使用原冻结规则自动选中空闲卡，模型及评分器已加载；本次没有停止或重启服务。

按用户要求，源码新增自动选卡的小占用共享模式。后续新任务可在`--gpus auto`后加`--allow-shared-gpus`；最大既有占用2048MiB，RTX3090使用默认0.8预算时仍需至少21,197MiB空闲（含1536MiB余量），1786MiB既有占用可以满足要求。共享模式允许其他计算进程及GPU利用率，默认模式仍要求完全空闲；两次稳定采样、最终复查和本项目队列锁继续保留，并优先选四张真正空闲的卡。当前运行的冻结配置不改动；其他任务后续增加显存仍可能产生竞争。
