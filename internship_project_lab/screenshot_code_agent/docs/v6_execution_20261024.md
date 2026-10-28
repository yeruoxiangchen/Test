# v6 执行修正与开发回归（2026-10-24）

本轮落实工具反馈、失败恢复、上下文和调用预算修正，并交付新旧版本的对照入口。尚未运行 GPU 模型评测，不能报告新的修复成功率。OCR 基线改进、中等难度数据构建与新仓库泛化评测属于下一阶段。

## 已实现

- v6 文件工具解包 worker 协议后返回真实多行源码、行号和分页位置；默认最多读取120行。命令与测试反馈也使用多行文本。截图工具的JSON接口保持兼容。
- 编辑失败返回匹配数量、文件指纹、真实附近源码和操作建议；仍然只允许唯一精确匹配，不做模糊修改。同一文件内容下，相同编辑参数连续失败2次后拦截下一次请求，并提示重新读取和改写；仍重复时以 `no_progress` 结束。
- 防重复判断绑定文件内容的SHA-256。文件被其他编辑或shell改变后允许重新尝试；新补丁会返回diff和Python语法诊断。语法检查使用工具worker的解释器，并返回版本；不代替目标仓库测试。
- 保留CoreCoder原 `Agent.chat()`，v6通过替换 `agent.context` 接入有界压缩：保留原始任务和初始截图证据、保留完整工具调用/响应配对，限制摘要输出，摘要不可用时用本地摘录回退。
- 决策与摘要分别计数。默认对照仍限制总代码模型调用为18次，其中摘要最多2次、每次最多384 token；最后一次总额度留给决策。成本、时间和工具调用上限继续统一生效。
- 本地服务根据实际EOS与生成上限返回 `length`，截断输出里的工具调用不执行；分别记录 `output_truncated`、`malformed_tool_output`、`no_progress`、`no_patch_submitted` 等状态。
- 中途终止仍导出已有补丁、在干净副本独立验证；终止原因与最终测试是否通过分别保存。

原配置默认为 `agent.mode=legacy`；新配置为 [v6-local.json](../configs/v6-local.json)。上游 `vendor/CoreCoder` 未修改，v5冻结源码和原成绩保留。当前本地模型服务修正后的结束协议属于v6；对照中的v5使用自己的旧服务源码。

## 已验证的工程结果

2026-10-24：完整套件 **103 passed in 26.79s**，包括15项新增v6行为回归，真实调用Bubblewrap，不加载模型。随后补充对照预算检查，比较脚本的 **6项测试全部通过**（其中5项已在前述全套内）；累计104个不同测试用例通过。Ruff、Python编译和shell语法检查另行执行。记录见 [verification.json](../outputs/v6_implementation_20261024/verification.json)。

覆盖多行反馈与分页、真实编辑失败与文件变化后的恢复、重复失败停止、只读测试保护、语法反馈、决策/摘要共享总上限、任务上下文保留、工具消息配对、截断状态、提前停止后的补丁独立验证，以及比较报告对失败/退化的保留。

这些结果是工程验证；已知补丁回放不计为模型修复效果。

## 已准备的对照

目录：`outputs/v6_dev_20261024/`。

- 从已观察的50题中选择12题，覆盖6个仓库，包含典型失败与已知成功任务；全部标记为开发回归。原任务、原内部划分和历史成绩不变。
- 默认只用 `oracle` 准确截图文本，先排查执行过程，排除OCR变化。v5与v6各12次，共24次真实模型运行。
- 同一Qwen3-VL-8B、相同任务、18次总代码/摘要调用、50次工具调用、1200秒Agent预算、2048输出上限。实际耗时、token消耗和退出原因另列。
- v5使用 `outputs/v5_eval_source` 的冻结版本；v6使用本次准备时复制的 `runtime_v6`。执行前后检查源码指纹，准备时记录任务JSON、截图、转写和评分元数据指纹。仓库初始目录及依赖前缀复用既有预检输入。
- 两组顺序使用同一张卡，每组完成后释放模型。启动器仅管理自己创建的进程组；状态命令按PID及进程启动时间检查存活。

`prepare`已完成，不启动GPU。运行前确认选择的GPU空闲；下面的6是示例物理卡号。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
systemd-run --user --unit=sca-v6-dev-20261024 \
  --working-directory="$PWD" \
  /usr/bin/bash scripts/run_v6_compare.sh run --gpu 6
```

进度与日志：

```bash
watch -n 15 'bash scripts/run_v6_compare.sh status'
journalctl --user -fu sca-v6-dev-20261024.service -o cat
tail -n 40 outputs/v6_dev_20261024/v5.log
tail -n 40 outputs/v6_dev_20261024/v6.log
```

停止本次任务：

```bash
bash scripts/run_v6_compare.sh stop
```

完成后生成 `comparison.md` 和 `comparison.json`：修复成功数、产生补丁数量、编辑失败请求/尝试、摘要占用、实际token/耗时、退出原因，以及同题的修复、保留、退化、共同失败。失败请求数量包含重复保护拦截，不是独立任务数。

失败后保留原目录，在新目录重试，不复用不完整输出。例如：

```bash
bash scripts/run_v6_compare.sh prepare --output outputs/v6_dev_retry1 --limit 12 --readers oracle
systemd-run --user --unit=sca-v6-dev-retry1 \
  --working-directory="$PWD" \
  /usr/bin/bash scripts/run_v6_compare.sh run --output outputs/v6_dev_retry1 --gpu 6
watch -n 15 'bash scripts/run_v6_compare.sh status --output outputs/v6_dev_retry1'
```

修改源代码后应重新prepare到新目录，旧冻结runtime不会自动跟随源码变化。`--limit 0 --readers oracle,ocr,vlm`可以准备原50题的三路回归，仍然属于已观察集合，不称为新的泛化测试。

## 后续判断

先看编辑失败与无效重复是否下降、摘要是否受限、原成功任务是否保留、补丁是否更经常通过独立测试，再决定是否扩大评测或替换代码模型。当前未证明上述修改提高模型修复率，不能把工程测试通过率写成真实仓库修复率。
