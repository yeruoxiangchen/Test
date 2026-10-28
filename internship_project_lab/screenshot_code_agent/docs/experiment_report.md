# 第一版验证与实验记录

日期：2026-10-21。状态：代码、合成数据、隔离回归与小规模真实模型闭环已完成。

本文保留初始小测记录。2026-10-22 的完整三组评测、公开仓库案例和新增入口见
[扩展评测与交付记录](experiment_report_v2.md)。

## 可运行性

- CoreCoder 固定在 `bd64cfe7ad2789fa0ab1b3a689ad160567b5cdb0` / 0.8.0，上游工作树未修改。
- 主环境：本项目 `.venv`，Python 3.12.7；独立依赖见 `requirements.lock.txt`。
- 隔离：本机实际可用的 Bubblewrap 用户/进程/网络命名空间；任务不可访问主机仓库之外的文件或 GPU。
- 最终检查：Ruff 和 compileall 通过，**30 项 pytest 检查通过**。
- 已验证后台查询/停止，超时后子进程终止，受保护测试不可改写，新文件/删除文件补丁可以独立重放。
- 初始复现也在单独副本执行，避免测试副作用污染私有基准快照。

## 数据

20 个自编 Python 标准库微型修复任务，每题包含错误代码、问题、参考修复和可运行测试。
每个错误版本都实际执行测试，并验证参考修改后通过。
其中 10 个采用代码截图，10 个采用实际失败日志的可见窗口截图；每题有浅色/深色两种样式，共 40 张图。
先按原任务划分 12 dev / 8 test，再生成样式。同题样式共用 group ID，规范评测清单只取一种样式。

渲染是本项目用 Pillow 编写的简化代码/终端文本渲染，借鉴 CodeOCR 的“代码/日志→截图→读取评测”流程。
未复制其实现，也不复现其论文结果。原项目参考：[CodeOCR](https://github.com/YerbaPage/CodeOCR)。

这些数据用于工程回归；没有使用 SWE-bench 任务，也没有真实用户截图。
审核记录：[dataset_build_report.json](artifacts/v1/dataset_build_report.json)。

## 无模型工程回归

20 题分别运行准确文本与普通 OCR，共 40 次已知补丁回放，全部通过补丁独立验证。
普通 OCR 实际读取图像，平均字符错误率 **21.85%**；准确文本组为 0。
平均耗时：准确文本回放 0.96 秒 / OCR 回放 5.59 秒。

回放脚本知道参考修改，**以上通过数仅证明工程链路，不是模型修复准确率**。
原始报告：[framework_validation_report.md](artifacts/v1/framework_validation_report.md)。

## 真实模型闭环

使用只读的本地 Qwen3-VL-8B-Instruct 基础权重，通过独立 `.model-env` 和 localhost API 调用。
代码 Agent 与视觉读取均使用同一基础模型；准确文本/OCR 组中的代码模型只接收文本证据。
模型环境从现有环境离线复制，并移除了复制来的 ChartQA 可编辑导入入口；原环境未修改。

条件：1 张 RTX 3090（GPU6），BF16、SDPA、贪心生成，8192 上下文，单请求最多 1024 新 token。
每题最多 12 次代码模型调用、40 次工具调用、3 次截图读取，Agent 时间预算 600 秒，单命令 30 秒。
只选择 2 个 dev 任务：代码截图 `synthetic_add`，报错截图 `synthetic_mean`。
三路各执行这两题，没有读取回放脚本或参考补丁。

| 输入读取 | 独立验证通过 | 平均端到端耗时 | 平均工具次数 | 平均读取 CER |
|---|---:|---:|---:|---:|
| 截图可见内容准确文本 | 2 / 2 | 28.17 秒 | 4.0 | 0.00% |
| RapidOCR CPU | 2 / 2 | 27.60 秒 | 5.0 | 28.20% |
| Qwen3-VL 视觉工具 | 2 / 2 | 32.02 秒 | 4.0 | 7.23% |

两题、三种读取方式共 **6 次端到端运行全部通过**。这是 **2 个独立合成任务**，不是 6 道独立题。
错误版本的目标测试失败，最终补丁均在干净副本重放后通过目标与回归测试。
实际修改分别为减号改加号、均值分母移除多余的 `+ 1`，没有修改评分测试。

VLM 读取调用每题 1 次，准确文本和 OCR 组的视觉 API 请求为 0；三组截图读取机会相同。
早期结果字段将所有截图读取计入 `vision_model_calls`，最终代码已区分截图读取与模型调用；
归档报告依据不可变请求事件恢复旧字段计数，并在记录中保留了修正说明。测试判定和原始事件没有改变。

PyTorch 记录的峰值已分配 GPU 显存为 **16.88 GiB**；驱动保留内存另计。
运行后模型服务自动停止，GPU6 恢复空闲。
本地模型 API 费用按 0 记录，不能将其理解成没有 GPU/电力/时间成本。

结果与可审阅材料：

- [真实模型完整报告](artifacts/v1/real_model_report.md)
- [逐题结果与使用量](artifacts/v1/real_model_report.json)
- [显存记录](artifacts/v1/model_stats.json)
- [代码截图案例补丁](artifacts/v1/synthetic_add_vlm.patch)
- [报错截图案例补丁](artifacts/v1/synthetic_mean_vlm.patch)
- [代码截图案例执行轨迹](artifacts/v1/synthetic_add_vlm.events.jsonl)
- [报错截图案例执行轨迹](artifacts/v1/synthetic_mean_vlm.events.jsonl)

## 失败与修正

1. **实际接口调试曾失败两轮。** 共 12 次任务尝试在本地端点报 400：`string indices must be integers, not 'str'`，
   没有生成模型 token。原因是 Qwen3-VL Processor 在扫描图片前要求所有角色的内容使用内容块，
   包括 system、assistant 和 tool 文本。修正后单独检查文本、图片和工具历史预处理，增加回归测试。
   这些调试结果保留为接口失败，不作为模型/OCR 能力比较；最终三路实验使用修复后的同一接口。
2. **代理配置也做了加固。** 本机端点显式绕过环境代理；远程模型保留原有代理行为。
   最终确认，上述 400 的根因是消息格式，不能归因于代理或模型修复能力。
3. **普通 OCR 存在代码字符错误。** 加法截图将函数头读成 `defaadd(a，b):`，也会去掉运算符两侧空格。
   报错截图出现中文标点、`failures=l` 和漏行。Agent 通过仓库源码核对后仍能修复两个简单问题。
4. **VLM 也不是逐字准确。** 加法截图的差异主要是空行/末尾换行；报错截图还存在省略/格式差异。
   当前 CER 对这些字符一视同仁，没有把“能理解报错”与“精确转写”混成一个指标。
5. **两个任务的修复率没有区别。** 本轮只能说明 VLM 的转写更接近参考，不能声称它提高了最终修复成功率。
   问题描述本身也含有较强线索，尚未通过“问题+仓库、不提供截图”对照隔离截图的增量贡献。

调试失败摘要：[integration_failures.json](artifacts/v1/integration_failures.json)。

## 结论与边界

已经验证的贡献是截图/裁剪工具、统一隔离文件与命令工具、共享预算和可追踪任务运行、
干净副本补丁验证，以及可复现数据与报告。CoreCoder 主循环、基础模型及 OCR 模型均为复用。

目前没有复杂真实仓库、第三方依赖项目和真实用户截图的修复成绩；没有训练、SFT 或 RL 结果。
真实模型仅运行两道 dev 合成题，单次贪心、顺序执行，耗时可能含预热差异，不能作为泛化或吞吐结论。
另外 8 道合成 test 任务仅完成已知补丁工程回归，没有做真实模型测试集评测。

第一版可复现命令：

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
bash scripts/validate.sh --tests-only
.venv/bin/sca run --task examples/code/task_light.json --reader ocr --model replay
# 确认 GPU6 空闲后，运行真实模型两题三路小测；结束后自动释放模型
bash scripts/run_local_smoke.sh
```

源文件指纹、环境和检查记录见 [verification.json](artifacts/v1/verification.json)。
