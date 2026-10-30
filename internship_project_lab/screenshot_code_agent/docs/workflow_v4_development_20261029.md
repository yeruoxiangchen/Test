# 2026-10-29：针对已知失败题的 v4 开发回归

按用户要求，针对 v3 失败的 9 题修改修复流程，后续仍运行原清单全部 10 题，不把失败题拆出分母。用户最终选择继续四卡，模型保持 Qwen3-Coder-30B-A3B-Instruct BF16、TP4、两题并发，不下载或切换 FP8 权重。本轮是已知案例开发回归，现已结束：**2/10通过**，详见[最终结果与失败分析](workflow_v4_results_20261029.md)。以下保留实现、验证和启动阶段记录。

## 实际改动

- 在 `workflow_cases.py` 加入 9 题的明确接口要求、上次失败的修正提示和可执行断言；先在原代码上验证能复现失败，再用相同脚本检查候选补丁。脚本由控制器保存，Agent 不能把断言改弱。提示来自此前失败分析，因此结果会明确标记为开发回归。
- `run_tests` 返回这些断言的实际失败输出；即使公共回归抽样通过，只要原问题验收失败，就继续同一对话内的修复。没有案例脚本的任务要求注册问题复现。
- 区分实现改动与新增调试脚本；只有 debug/test/repro 文件不再满足实现改动条件。交付前将新增诊断文件移入保留的 `.sca_scratch/delivery_cleanup`，不删除诊断证据。
- 每题最多 96 次调用，主循环预留最多两轮各 16 次纠错；每题修复阶段上限 30 分钟，预留 240 秒作最后检查。普通命令上限从 600 秒降为 120 秒，最终评分仍保留原来的 600 秒上限。
- pvlib-1616 的公共回归限定为已有 IAM 测试模块，避免无关太阳位置/JIT 测试拖延。其他样本的公共测试继续按题面排序抽样。
- 修复状态计数中把输入校验进度误作为已完成题数的问题；保留 v3 报告兼容性。报告展示开发验收失败内容并标记本轮用途。

## 九题重点

| 任务 | 本轮要求 |
| --- | --- |
| marshmallow-2123 | `relative=True, absolute=False`，同时保留 URL 格式验证与既有行为 |
| pvlib-1616 | 在 `iam.physical` 接入 `n_ar`，检验数值和退化情形；原题建议 `fresnel_ar` 与既有验收接口的差异明确告知 |
| pydicom-816 | 在解码路径做通用未知编码回退、坏字节处理、警告和严格模式；不硬编码字符串特例 |
| pyvista-4329 | faces/cells 只读、setter 仍有效，检查内部原地修改的调用者 |
| sqlfluff-1733 | L039 与换行/缩进的交互，精确检查格式和幂等性，不能只留下调试脚本 |
| marshmallow-1702 | metadata 显式参数与旧关键字兼容、弃用警告、不修改调用者字典；旧关键字覆盖同名显式 metadata |
| pvlib-1832 | interp 在 IAM 参数表、Array/PVSystem、ModelChain 的完整接入与自动推断 |
| pydicom-793 | unknown VR 在值转换入口统一格式化，非字母字符触发两个字符都用小写十六进制 |
| astroid-1092 | 生成器上下文传递真正绑定的实例，断言推断为 B，不依赖对象地址 |

astroid-984 保留在这 10 题中，重新执行；没有预填上次成功补丁。

## 验证

工作流相关 53 项测试通过（46.02 秒），覆盖真实沙箱、旧流程兼容、错误反馈驱动重试、debug-only 补丁拒绝、诊断产物交付清理、时间预留。Ruff 检查通过。

额外运行 `validate_workflow_cases.py`，在独立 CPU 沙箱中检查断言能令原代码失败、参考源码修改后通过。修正了初稿中 pydicom 回退层级和 marshmallow 合并优先级两处不正确假设，最终 9/9 检查通过（54.3 秒）。这只是新增断言本身的验证，不是 Agent 的 9/9 成绩；参考源码补丁没有交给模型，也没有应用到模型修复工作区。

相关检查产物：

- `/data/zjr/screenshot_code_agent/v4_testcheck_20261029/`
- `/data/zjr/screenshot_code_agent/v4_contract_check_corrected_20261029/report.json`
- `/data/zjr/screenshot_code_agent/v4_preflight_20261029/report.json`

正式模型回归使用新目录 `/data/zjr/screenshot_code_agent/v4_dev10_20261029/`，旧 v2/v3 分数、补丁、数据和参考评分器不修改。所有新输出和运行缓存均位于 `/data`。

## 监视

```bash
journalctl --user -fu sca-v4-dev10-20261029.service -o cat
```

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_repair_workflow.sh status --output /data/zjr/screenshot_code_agent/v4_dev10_20261029'
```

10/10题最终CPU预检ready，共162个通过的公共测试状态，耗时105.39秒；源码、配置和任务指纹核对通过。服务于2026-10-29 06:35:29 UTC启动，MainPID=3602906，确认active/running并进入模型加载。因2号卡在启动前被其他任务占用约12GB，实际选择3、5、6、7；每卡显存预算74%，额外保留1536MiB，启动要求空闲19723MiB，四卡均连续两次满足条件。vLLM日志确认BF16、tensor_parallel_size=4、上下文32768、两题并发，且TP0—TP3四个进程已建立通信并加载权重。启动时尚未取得模型修复成绩。

只读核对模型进程3604059的指定环境字段：CUDA_VISIBLE_DEVICES=3,5,6,7，TMPDIR、TORCH_HOME、HF_HOME、HF_HUB_CACHE、TORCH_EXTENSIONS_DIR、MPLCONFIGDIR均在本轮/data输出目录；权重物理位置为/data/zjr/sca_coder_runtime_20261026/model。

本批输入仍未附带截图，测的是仓库修复后端。既有参考评分器仍在Agent停止后执行。一次取消的两卡方案没有启动GPU模型；对应FP8下载服务失败后未重启，新建的两卡脚本与配置已撤回。
