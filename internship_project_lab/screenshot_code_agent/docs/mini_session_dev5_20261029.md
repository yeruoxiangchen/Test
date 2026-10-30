# 2026-10-29：接入链路复查、修复与固定5题回归

用户要求重新检查接入代码，并将下一次测试减为5题。本次先修已复现的环境及交付问题，再做基础回归和5个真实任务环境的强制检查，通过后启动新的模型测试。原10题结果保持原样。

**完成更新：** 本轮已于14:02:53 UTC结束，0/5，374次调用。下方启动前检查和13:31状态保留当时记录；最终分析见[5题结果与失败原因](mini_session_dev5_results_20261029.md)。实际轨迹确认旧的/tmp和Git故障未复发，但暴露空字节参数恢复、提交行为检查和退出候选保留缺口。

## 为什么之前仍然出错

此前的验证主要覆盖单次读写、编辑、命令返回和最终评分，缺少跨多条命令的真实工作流检查。直接复用CoreCoder时代的隔离环境后，没有核对它与mini常用shell工作方式的差异：每次执行重建/tmp、只读.git，以及控制器从整个工作区导出补丁。简短脚本化测试没有覆盖模型常用的临时脚本、Git回滚和备份文件。这是接入设计及检查覆盖的缺口，不能归因于mini上游，也不能用“测试数量多”替代行为覆盖。

本次按“输入→模型请求→工具执行→文件状态→回滚→提交反馈→可信补丁→独立评分→状态报告”检查。并未宣称仓库所有可选模式都完全正确；新的5题使用明确开启的`mini_session`模式，历史冻结运行不变。

## 已修复的内容

1. **每题持久临时目录。** 在/data该题目录下建立独立session_tmp，挂载为/tmp；HOME位于其中。跨命令文件保留，不共享主机/tmp。每次仍是新shell，cd和shell变量不持久；后台服务与客户端需放在同一条命令中，提示已说明这一边界。
2. **可用的Git回滚。** Agent工作区的.git可写，git checkout/restore、diff、commit可以使用；原始基线及受保护测试保持独立。没有开放主机仓库或参考评分文件。
3. **补丁导出不信任Agent的HEAD/index。** 控制器使用独立基线的Git元数据和临时索引，对当前工作区文件生成差异。模型commit、reset甚至移动自己的.git后，仍能提取相对原始基线的修改。新源码和删除操作保留，新增备份扩展名文件从交付排除，原工作区证据保留。
4. **提交时给出真实失败反馈。** 在任务自己的Python版本中检查修改文件语法、可识别顶层包导入、是否存在实现变化，以及已有公开开发断言；空改动、仅注释、明显根目录调试文件和这些检查的失败会返回给同一mini对话继续处理。这里没有运行或读取参考测试。
5. **5题与预检绑定。** 新入口支持显式固定5题；正式模型服务启动前要求真实环境检查通过。预检收据绑定冻结源码、配置、清单和JUnit文件，匹配后复用；失败、跳过或收据不匹配均不会启动模型。准备数量与评测完成数量分开记录。

mini上游DefaultAgent代码仍未修改，但环境行为和提交反馈已改变，应将本轮称为“修复后的mini接入版本”，不能包装成完全原样的上游成绩或只改一个变量的消融。

## 验证范围与结果

| 环节 | 实际验证 | 结果/边界 |
| --- | --- | --- |
| 文件生命周期 | 写/tmp和HOME，下一条读取；shell工作目录恢复默认 | 通过，5种OCI环境及宿主Bubblewrap均检查 |
| 源码恢复 | 写入坏语法，git checkout恢复，再导入 | 通过 |
| 评分隔离 | 现有测试写入被拒绝，修复时无/grader/eval.sh与模型密钥 | 通过 |
| Git与导出 | Agent提交并移动.git；控制器仍导出原基线差异 | 通过，基线HEAD不变 |
| 交付内容 | 新源码保留、备份排除、根目录调试文件反馈、独立重放 | 通过 |
| 提交失败恢复 | 实际mini循环先提交坏代码，被反馈后回滚修正，再提交、评分 | 通过；这是脚本化接入测试，不是模型修复成功率 |
| 检查有效性 | 空补丁、仅注释、坏语法、缺依赖的包导入、公开断言失败 | 均能阻止该次提交 |
| 超时 | 超时返回非零及标记，后续命令仍可运行 | 通过 |
| 历史行为 | 原隐藏评分、补丁重放、软链接、工具接口及v3相关回归 | 基础套件通过 |
| 真实任务依赖 | 5题原仓库实际包导入；PyVista实际离屏渲染 | 通过 |
| 模型能力 | 能否生成语义正确、覆盖边界的修复 | 必须由本轮实际评测判断，短检查不提供保证 |

基础套件：65 passed、1 skipped，35.31秒。唯一跳过的是未配置任务清单时的真实仓库参数化用例；随后通过显式5题清单执行的强制检查为 **48 passed、0 skipped**，46.63秒。Ruff和Shell语法检查通过。

检查产物：

- `/data/zjr/screenshot_code_agent/mini_session_checks_20261029/unit.xml`
- `/data/zjr/screenshot_code_agent/mini_session_dev5_20261029/session_preflight.xml`
- `/data/zjr/screenshot_code_agent/mini_session_dev5_20261029/session_preflight.json`
- `/data/zjr/screenshot_code_agent/mini_session_dev5_20261029/logs/session_contract_preflight.log`

提交检查仍不是完整行为证明：例如AST变化不能证明逻辑修复，公开断言仍可能缺少边界，新增临时文件的识别也有明确规则而非万能分类。最终成功仍由原参考评分器在Agent退出后判定。模型预算结束仍会导出现有候选并评分，不能把提前达到限额当成验收通过。

## 固定5题

按接入故障覆盖选择，并在本次运行前冻结；不是随机未见测试集。相同5题在上一轮mini为1/5、v4为2/5，只能做开发回归对照。

| 顺序 | 任务 | 选择原因 |
| --- | --- | --- |
| 1 | marshmallow-code__marshmallow-2123 | /tmp丢失、回滚被阻、最终缩进错误 |
| 2 | pylint-dev__astroid-984 | 保留上一轮成功题，检查是否退步 |
| 3 | pyvista__pyvista-4329 | /tmp丢失、实际渲染及getter/setter问题 |
| 4 | pvlib__pvlib-python-1832 | 跨模块编辑、回滚被阻、最终语法错误 |
| 5 | pydicom__pydicom-793 | /tmp丢失以及ASCII/非ASCII边界 |

复用原准备数据和已有开发提示，不加入参考补丁，不改原分数。这批任务没有截图输入，仍评测仓库修复后端。

## 新运行

入口`bash scripts/run_mini_dev5.sh`，配置`configs/mini-session-dev5.json`。原Qwen3-Coder-30B-A3B-Instruct BF16，TP4，GPU **2、3、6、7**，两题并发；每题仍96次调用/1800秒，普通命令120秒、请求300秒，最终评分600秒，上下文32768。数据、快照、临时文件和模型缓存均在/data；控制器的既有环境仍在home引用，不声称零home读取。

后台服务`sca-mini-session-dev5-20261029.service`于13:06:57 UTC启动，控制器PID209712，初始模型进程PID212171。状态确认5/5输入校验通过、复用48项预检收据，四卡连续两次满足19723MiB空闲要求。13:31 UTC复核已进入`evaluating`，controller_alive=True，2题运行、3题等待，模型工具请求持续返回HTTP 200；尚未完成整批，不能提前报告成功率。

正式输出：`/data/zjr/screenshot_code_agent/mini_session_dev5_20261029/`。可继续修改工作区代码，当前运行使用冻结源码。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_mini_dev5.sh status'
```

```bash
journalctl --user -fu sca-mini-session-dev5-20261029.service -o cat
```

逐题结果在`mini/runs/<task>/`，新增`delivery_checks/<尝试号>/report.json`与`candidate.diff`记录提交被接受/拒绝的实际原因，最终`patch.diff`和参考测试日志仍独立保存。整批完成后查看输出根目录`report.json`。
