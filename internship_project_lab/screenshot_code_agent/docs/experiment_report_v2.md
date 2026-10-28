# 扩展评测与第一版交付记录

日期：2026-10-22。所有下述修复成绩均来自真实模型调用。

本轮共 23 个原始任务、21 个任务或仓库分组、69 次运行；61 次通过独立补丁重放和固定测试。
合成微型仓库与公开历史案例分别统计。三组为截图可见内容准确文本、RapidOCR CPU、Qwen3-VL 视觉工具。

## 条件与可复现性

- 本机单张 RTX 3090（GPU 7），本地 Qwen3-VL-8B-Instruct BF16 / SDPA，贪心生成。
- 三组使用相同代码模型和配置：8192 上下文，单次最多 1024 新 token；每题 12 次代码调用、40 次工具调用、3 次截图读取、600 秒，单命令 30 秒。
- 所有逐题配置文件一致：`True`；原始任务、工具、模型、费用和测试事件保留。
- 数据组成：public_historical_fix_with_synthetic_screenshot / demo: 3 题；self_authored_synthetic_micro_repository / dev: 12 题；self_authored_synthetic_micro_repository / test: 8 题。
- 所有截图为合成渲染；真实用户截图数为 0。公开案例保留完整历史版本和许可证。
- 报告中的 API 价格为 0，不代表 GPU、时间与电力没有成本。

## 按数据来源汇总

| 来源 | 读取方式 | 独立验证通过 | 平均秒数 | 平均工具调用 | 平均读取 CER |
|---|---|---:|---:|---:|---:|
| 公开历史案例 | ocr | 2/3 | 86.08 | 7.67 | 21.09% |
| 公开历史案例 | oracle | 1/3 | 65.15 | 6.67 | 0.00% |
| 公开历史案例 | vlm | 1/3 | 97.14 | 10.33 | 6.90% |
| 自编合成题 | ocr | 18/20 | 38.33 | 5.40 | 21.85% |
| 自编合成题 | oracle | 20/20 | 28.81 | 5.00 | 0.00% |
| 自编合成题 | vlm | 19/20 | 35.53 | 4.95 | 6.35% |

全部任务合计：OCR 20/23，视觉工具 20/23。
本轮两者总体固定测试通过率相同。
视觉转写 CER 更低，没有在本轮总体修复成绩中形成提升。
失败复核显示瓶颈还包括工具调用格式、重复检索/读取、精确替换及测试规格覆盖；不能把修复失败统一归为识别错误，也不据此宣称跨仓库泛化或统计显著性。

PyTorch 峰值已分配显存：18.05 GiB，驱动保留内存另计。
批次执行耗时：46.68 分钟，不含模型启动与后续报告生成。

## 按集合分开报告

| 来源 | 集合 | 读取方式 | 通过 / 尝试 | 分组数 |
|---|---|---|---:|---:|
| 公开历史案例 | demo | ocr | 2/3 | 1 |
| 公开历史案例 | demo | oracle | 1/3 | 1 |
| 公开历史案例 | demo | vlm | 1/3 | 1 |
| 自编合成题 | dev | ocr | 11/12 | 12 |
| 自编合成题 | dev | oracle | 12/12 | 12 |
| 自编合成题 | dev | vlm | 11/12 | 12 |
| 自编合成题 | test | ocr | 7/8 | 8 |
| 自编合成题 | test | oracle | 8/8 | 8 |
| 自编合成题 | test | vlm | 8/8 | 8 |

## 同题配对

| 来源 / 集合 | 比较 | 配对题数 | 都通过 | 仅左侧通过 | 仅右侧通过 | 都失败 |
|---|---|---:|---:|---:|---:|---:|
| 公开 / demo | oracle → ocr | 3 | 1 | 0 | 1 | 1 |
| 公开 / demo | oracle → vlm | 3 | 1 | 0 | 0 | 2 |
| 公开 / demo | ocr → vlm | 3 | 1 | 1 | 0 | 1 |
| 合成 / dev | oracle → ocr | 12 | 11 | 1 | 0 | 0 |
| 合成 / dev | oracle → vlm | 12 | 11 | 1 | 0 | 0 |
| 合成 / dev | ocr → vlm | 12 | 10 | 1 | 1 | 0 |
| 合成 / test | oracle → ocr | 8 | 7 | 1 | 0 | 0 |
| 合成 / test | oracle → vlm | 8 | 8 | 0 | 0 | 0 |
| 合成 / test | ocr → vlm | 8 | 7 | 0 | 1 | 0 |

CER 对空格、换行、行号和标点敏感。转写差异与最终修复结果分别记录；读取更准确本身不证明修复成功率提高。顺序固定为 oracle、OCR、VLM，耗时含预热与调度影响。

## 失败记录

- `boltons_copy_function_kwdefaults` / oracle：unresolved；阶段 `repair_or_verification`；结束原因 `submitted`；参考文件重合 `False`；详见固定测试日志
- `boltons_copy_function_kwdefaults` / vlm：unresolved；阶段 `repair_or_verification`；结束原因 `step_limit`；参考文件重合 `False`；详见固定测试日志
- `boltons_one_to_one_empty_update` / ocr：unresolved；阶段 `repair_or_verification`；结束原因 `budget_exceeded`；参考文件重合 `False`；Model step limit reached (includes context summaries)
- `boltons_one_to_one_empty_update` / oracle：unresolved；阶段 `repair_or_verification`；结束原因 `step_limit`；参考文件重合 `False`；详见固定测试日志
- `boltons_one_to_one_empty_update` / vlm：unresolved；阶段 `repair_or_verification`；结束原因 `budget_exceeded`；参考文件重合 `False`；Model step limit reached (includes context summaries)
- `synthetic_csv_fields` / ocr：unresolved；阶段 `repair_or_verification`；结束原因 `submitted`；参考文件重合 `True`；详见固定测试日志
- `synthetic_leap` / vlm：unresolved；阶段 `repair_or_verification`；结束原因 `step_limit`；参考文件重合 `False`；详见固定测试日志
- `synthetic_transpose` / ocr：unresolved；阶段 `repair_or_verification`；结束原因 `step_limit`；参考文件重合 `False`；详见固定测试日志

以上是可观察的失败阶段；没有把 CER 或参考文件重合直接当作失败根因。

已人工检查的检索循环、未闭合工具调用和 CSV 边界案例见 [失败复核](artifacts/v2/failure_review.md)。

## 事后边界检查

CSV 补充检查针对连续、首尾逗号的空字段语义，按参考补丁核对。这些测试在观察结果后加入，不改变本轮原评分和上述主指标。

| 读取方式 | 原测试通过 | 三项空字段检查通过 | 两者均通过 |
|---|---|---|---|
| oracle | True | False | False |
| ocr | False | True | False |
| vlm | True | False | False |

已知参考补丁通过全部补充检查。准确文本/视觉组通过原测试的 CSV 修改会丢失空字段；这暴露了原固定测试的覆盖不足，不能将其成绩等同于完整语义正确性。
[补充检查原始记录](artifacts/v2/posthoc_csv_audit.json)。

## 工程验证与贡献

新增可配置完整评测、公开历史案例构建、仓库/截图/描述直接输入命令、逐题后台进度、按来源与集合报告及证据归档。CoreCoder 循环、现成模型和 OCR 模型为复用。
构建公开任务时验证了 3 个错误版本与官方源码补丁；准确文本/OCR 的已知补丁回放另有 6 次独立验证。回放检查不计入这里的模型修复成绩。

公开案例目前均为单文件小补丁，评分测试是可见且受保护的 unittest 适配器，未跑上游完整套件。模型可能见过公开历史修复。没有真实用户截图、隐藏测试、跨更多仓库的泛化或训练结果。

## 运行问题

早期本地模型启动使用 180 秒等待门限；失败记录显示权重仅加载到 50%。现改为默认 900 秒、每 15 秒显示等待进度，并在模型日志中区分导入、权重加载与处理器加载阶段。失败目录保留，没有将其计作某种截图读取方法的修复失败。

## 可审阅材料

- [完整模型报告](artifacts/v2/real_model_report.md)
- [逐题结果、预算、测试、定位与读取诊断](artifacts/v2/real_model_report.json)
- [同题配对结果](artifacts/v2/paired_comparisons.json)
- [源码与产物指纹](artifacts/v2/verification.json)
- [公开案例来源](public_cases.md)
- [演示步骤和贡献边界](demo_guide.md)
