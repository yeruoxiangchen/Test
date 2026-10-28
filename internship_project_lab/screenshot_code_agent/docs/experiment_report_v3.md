# 官方来源任务与截图读取评测

日期：2026-10-22。

本轮接入 SWE-bench Lite 原始任务、测试补丁、官方日志解析器/评分器，以及 CodeOCR 作者发布的代码数据、渲染和重建指标。
执行使用独立 Python 3.9.20 与 Bubblewrap。Docker Hub 连接失败、公开缓存站返回 403，未运行官方 Docker 镜像；成绩属于截图派生子集。
官方测试已安装在 Agent 可见仓库中，并设为只读保护；本轮不是隐藏测试协议。参考源码补丁和官方评分元数据在仓库外。

## 数据和环境预检

预先固定 10 个任务、2 个仓库组；9 个通过错误版本复现与参考补丁评分。
任务 ID 和选择规则先于完整模型对照确定；Pytest 分为开发组、SymPy 分为测试组，再渲染两种主题。原始来源均为官方 test split。
保留原 Issue 周围描述，仅将预先选定的一个代码/输出块替换为截图；未声称每题都必须依赖图像。

| 原始任务 | 预检状态 |
|---|---|
| `pytest-dev__pytest-5103` | validated |
| `pytest-dev__pytest-5227` | validated |
| `pytest-dev__pytest-5413` | validated |
| `pytest-dev__pytest-5495` | validated |
| `pytest-dev__pytest-5692` | validated |
| `sympy__sympy-11400` | validated |
| `sympy__sympy-11870` | environment_failed |
| `sympy__sympy-11897` | validated |
| `sympy__sympy-12236` | validated |
| `sympy__sympy-12419` | validated |

## 真实模型修复

完成 27/27 次运行；通过独立重放与官方评分的运行共 2 次。
三组配置文件一致：True。模型为 Qwen3-VL-8B-Instruct；代码输出上限 1024 token，视觉转写 2048 token；18 次代码/压缩调用、50 次工具调用、1200 秒总预算、180 秒单命令。
Agent 以 4096 token 估计阈值压缩上下文，本地服务最大上下文 16384。该配置在预备运行发现原上限不足后确定，预备运行另存，不合并成绩。

| 读取方式 | 通过 / 尝试 | 平均秒数 | 平均工具调用 | 截图原始 CER |
|---|---:|---:|---:|---:|
| oracle | 1/9 | 124.92 | 7.44 | 0.00% |
| ocr | 0/9 | 128.57 | 9.67 | 21.57% |
| vlm | 1/9 | 140.26 | 10.78 | 7.50% |

截图 CER 对首轮整图读取逐字符比较，包含空白差异；与下表 CodeOCR 的归一化 CER 定义不同。两种指标不合并。
评分同时检查官方 FAIL_TO_PASS 和 PASS_TO_PASS；退出码为零、Agent 声称完成均不能单独判为成功。

| 内部分组 | 读取方式 | 通过 / 尝试 |
|---|---|---:|
| dev | oracle | 1/5 |
| dev | ocr | 0/5 |
| dev | vlm | 1/5 |
| test | oracle | 0/4 |
| test | ocr | 0/4 |
| test | vlm | 0/4 |

测试缓存剔除后，独立重放 27 份补丁，2 次通过；与原始判定全部一致：True。空补丁保留为失败。
清理不使用参考补丁、不做语义修改，也不调用模型；原始补丁、原始成绩与清理后测试日志分别保留。
本轮模型比较使用冻结的修复前工程版本。文件检索、缓存导出和 Unicode 日志解析问题的后续修复与开发组回归单独记录，不混入这 27 次成绩。

## CodeOCR 重建子集

从作者发布的 100 份代码中，按固定 ID 顺序选择 10 份不超过 80 行、2200 字符的完整文件，来自 10 个仓库。
调用上游渲染函数和 1×/2×退化函数生成 20 张图片，比较 CPU OCR 与视觉模型，共 40 次读取；没有运行这些代码。
归一化 CER 使用上游原函数，忽略空白行和行尾空格，保留缩进。另存原始 CER、WER、行相似度和 Python 语法检查。

| 读取方式 | 退化倍数 | 完成 / 尝试 | 归一化 CER | 语法通过数 |
|---|---:|---:|---:|---:|
| ocr | 1 | 10/10 | 5.39% | 0 |
| ocr | 2 | 10/10 | 8.22% | 0 |
| vlm | 1 | 10/10 | 1.05% | 8 |
| vlm | 2 | 10/10 | 1.03% | 8 |

这是代码读取测量，不能当作修复成功率。上游 API 辅助模块缺少配置常量；适配器直接加载两个未修改的纯函数 AST，渲染模块正常导入，上游源码保持原样。

初版 DejaVu Sans Mono 字体使四份代码的中文注释显示为方框；其 40 次读取仅保留作数据质量诊断。主表使用 Noto Sans Mono CJK、经字形覆盖审计的同样十份代码重新渲染，参考文本和任务 ID 没有改变。
SWE-bench 的一个尚未送入模型的截图含缺字 ℯ，已更换为支持该字符的字体。已核对 3 次实际输入与修正后的图像像素完全一致；原图及修正前后指纹保留。

## 失败与范围

- `pytest-dev__pytest-5103` / ocr：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `pytest-dev__pytest-5103` / oracle：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `pytest-dev__pytest-5103` / vlm：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `pytest-dev__pytest-5227` / ocr：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `pytest-dev__pytest-5413` / ocr：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `pytest-dev__pytest-5413` / oracle：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `pytest-dev__pytest-5413` / vlm：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `pytest-dev__pytest-5495` / ocr：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `pytest-dev__pytest-5495` / oracle：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `pytest-dev__pytest-5495` / vlm：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `pytest-dev__pytest-5692` / ocr：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `pytest-dev__pytest-5692` / oracle：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `pytest-dev__pytest-5692` / vlm：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `sympy__sympy-11400` / ocr：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `sympy__sympy-11400` / oracle：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `sympy__sympy-11400` / vlm：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `sympy__sympy-11897` / ocr：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `sympy__sympy-11897` / oracle：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `sympy__sympy-11897` / vlm：unresolved；阶段 repair_or_verification；结束原因 step_limit；
- `sympy__sympy-12236` / ocr：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `sympy__sympy-12236` / oracle：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `sympy__sympy-12236` / vlm：unresolved；阶段 repair_or_verification；结束原因 budget_exceeded；Model step limit reached (includes context summaries)
- `sympy__sympy-12419` / ocr：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `sympy__sympy-12419` / oracle：unresolved；阶段 repair_or_verification；结束原因 submitted；
- `sympy__sympy-12419` / vlm：unresolved；阶段 repair_or_verification；结束原因 submitted；

本轮包含两个仓库的小规模派生任务；全部截图合成，真实用户截图数为零。没有原始 SWE-bench 排行榜成绩、SWE-bench Multimodal 成绩或训练结果。
初始环境失败、依赖缺失、长测试超时和预备运行的上下文错误分别保留；环境预检失败不描述为模型修复失败。
本地 API 标价为零，GPU 时间、电力与计算资源有实际成本。
修复批次汇总器错误地将 Unicode NEL 当成 JSONL 记录分隔符，导致完成 27 次运行后汇总失败，最终 CUDA 峰值未采集。用独立修正的汇总器恢复报告，原始事件、补丁与逐题判定 SHA-256 均保持一致。

源码固定提交、数据指纹、逐题补丁、完整固定测试日志与评分细节见 `docs/artifacts/v3/`。

读取批次的 40 条结果与报告完整保存，但启动脚本在结束采集显存时因运行中修订发生解析错误；模型已退出。读取批次的最终 CUDA 峰值未采集，未用其他批次的值补填。

逐项原因复核见 [失败分析](failure_review_v3.md)。当前交付源码的 60 项检查和单题 OCR 开发回归见
[对照后的工程修复](post_benchmark_fixes_v3.md)；这些后续检查没有改写本报告的三路对照成绩。
