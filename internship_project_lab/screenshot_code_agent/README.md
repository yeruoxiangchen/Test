# 支持代码截图与报错截图的代码修复 Agent

基于 **CoreCoder 0.8.0**，输入仓库、问题描述、截图和测试命令，输出补丁、独立测试结果和执行轨迹。
第一版使用现成模型，不包含训练。默认采用 API，截图裁剪、普通 OCR 和仓库测试在 CPU 上执行。

2026-10-27 新增 **面向实际使用的 CoreCoder 修复流程**：本地仓库索引与代码定位、定位/复现/修改分步执行、多段补丁编辑、公共测试驱动的最多两轮纠错、保留调用额度的断点恢复，以及可查看截图、diff、测试和下载交付包的 HTML 报告。使用 `bash scripts/run_repair_workflow.sh`；[输入示例、四卡启动和监视命令](docs/scale_repair_20261025.md#2026-10-27corecoder工程修复流程)。这是一套可选的单 Agent 流程，已有 mini 对照入口仍保留；正式100题验收由用户启动，尚无这套流程的新模型成功率。

没有四张空闲卡时，用 `bash scripts/wait_and_run_workflow.sh`：先准备输入，再每10秒检查八张卡，自动选择连续两次空闲的任意四张启动同一轮验收。推荐通过 systemd 后台运行，见[自动等卡命令](docs/scale_repair_20261025.md#自动等待任意四张空闲卡)。

2026-10-28 支持给新的自动选卡任务加 `--allow-shared-gpus`：允许卡上已有进程占用不超过2048MiB，同时仍要求剩余显存满足模型预算及1536MiB余量，连续两次采样并在启动前复查；优先选完全空闲的卡。当前v2已自行选中2、3、4、5并开始评测，无需为更改排队规则重启。该选项不能防止其他任务在模型启动后增大显存占用。

2026-10-28 发现当前工作流的 OCI 隐藏评分脚本挂载失败，旧报告中的“0通过、0运行异常”不能解释为修复成功率。源码已修复并完成单题真实验证；运行中的冻结版本保持原样。使用 `bash scripts/regrade_workflow.sh run --wait-for-source` 等待原批次结束后，仅在 CPU 上重评已有补丁，另存结果。见[修复说明和后台命令](docs/scale_repair_20261025.md#2026-10-28隐藏评分挂载修复与补丁重评)。

2026-10-28 **工作流 v2** 修复完整请求的 token 压缩、公共测试基线比较和复现证据核验。新增 `register_reproduction`，固定断言脚本并分别在原代码与补丁代码上验证；既有测试失败与新增回归分开记录。阶段额度耗尽会标为缺少完成证据，后续修复仍可补充复现。新评测使用独立输出目录；[修复验证与新四卡评测命令](docs/scale_repair_20261025.md#2026-10-28工作流v2完整请求计数与可执行反馈)。原冻结任务及历史成绩保留，尚无 v2 模型成功率。

2026-10-27 新增 **CoreCoder / mini-SWE-agent 同模型对照入口**：固定 mini v2.4.6，复用现有 Qwen3-Coder 四卡服务；各跑同一批100题，参考测试仅在最终评分时安装。新数据、环境副本、缓存、日志和结果统一放 `/data/zjr/screenshot_code_agent/harness_compare_20261027`。短测试已通过，正式对照尚未启动；[启动与监视命令](docs/scale_repair_20261025.md#2026-10-27mini-swe-agent接入与新协议对照入口)。旧协议工具修复后的 Coder 为19/100，新协议需要重新建立两臂成绩。

2026-10-24 新增 **v6 执行修正**：多行工具反馈、重复失败恢复、有界上下文、决策/摘要预算及准确截断状态。
使用 `configs/v6-local.json` 启用；12题开发回归已完成：v5为1/12，v6为3/12（新增修复2题，原成功保留）。这不是未见任务泛化成绩。
实现、检查结果和后台命令见 [v6 执行修正与回归](docs/v6_execution_20261024.md)。

同日新增 [四卡 Coder 与 Full 扩容入口](docs/scale_4gpu_20261024.md)：元数据与500道候选清单已准备，目标开发100题／独立测试300题。10月25日完成[预检修复并恢复队列](docs/scale_repair_20261025.md)：修复依赖、镜像认证及评分ID适配，开发集先行；已有真实样例恢复验收，尚无新增模型成绩。

开发与实验的完整时间线见 [连续实验记录（2026-10-21 至 2026-10-28）](docs/experiment_history.md)，
包含各阶段代码更新、数据准备、失败恢复、正式结果与证据链接；时间统一为 UTC。

```text
截图 + 问题 + 仓库
    → oracle / CPU OCR / 视觉模型读取
    → CoreCoder Agent：检索 → 核对 → 修改 → 测试 → 根据反馈继续修改
    → 导出包含新增/删除文件的补丁
    → 在干净副本重放补丁并运行固定测试
    → patch.diff + result.json + 日志
```

## 第一版的实际实现范围

| 复用 CoreCoder | 本项目实现 |
|---|---|
| 未修改的 `Agent.chat()` 循环、上下文压缩、`Tool`/`LLMResponse` 接口 | 截图资源注册、可复现裁剪、结构化视觉读取、CPU OCR、准确文本对照 |
| 模型调用工具并读取反馈的执行模式 | 所有文件与命令工具的 Bubblewrap 适配、固定测试工具 |
| 主循环对工具错误的恢复 | 跨代码/视觉调用的预算、控制信号、结构化事件、后台状态与停止 |
| 工具接口约定 | 仓库快照、完整补丁导出、独立重放验证、合成数据和配对报告 |

上游固定提交：`bd64cfe7ad2789fa0ab1b3a689ad160567b5cdb0`，许可证 MIT。来源见
[CoreCoder](https://github.com/he-yufeng/CoreCoder) 和 [UPSTREAM.json](UPSTREAM.json)。
上游完整源码保持不变，位于 `vendor/CoreCoder`，由安装脚本固定版本下载。

## 环境安装

Linux、Python 3.10+，需 `/usr/bin/bwrap`、`/usr/bin/prlimit`、`/usr/bin/python3`、Git 和 DejaVu Sans Mono。
Bubblewrap 必须能创建用户、进程、网络命名空间。本项目不会在隔离不可用时退回主机执行。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
bash scripts/setup.sh
.venv/bin/sca doctor
```

独立虚拟环境为 `.venv`；没有修改其他项目的环境、配置和代码。实际验证过的运行依赖见
`requirements.lock.txt`。`doctor` 仅报告凭据是否存在，不输出密钥。

默认 Bubblewrap 运行时挂载系统 `/usr` 等只读运行库，内置微型任务为 Python 标准库任务。
SWE-bench 来源任务另用只读的独立 Python 依赖前缀。其他依赖仓库需配置对应运行时；
不把其他项目的 Conda 环境或整个 `/home` 挂载给 Agent。
仓库初始快照上限为 10,000 文件 / 100 MB，第一版拒绝符号链接和设备等特殊文件。

## 无 API 的工程验证

```bash
bash scripts/validate.sh
```

该脚本构建 20 个自编合成修复任务，每题有错误版本、参考补丁、固定测试、真实失败日志和两种截图主题。
它执行隔离测试和 `oracle,ocr` 两组**已知补丁回放**。

**回放是执行链路验证，不是模型修复成功率。** `--model replay` 从独立脚本读取已知编辑，
报告会标出 `model_inference: false`；它不证明 Agent 能独立定位或修复这些问题。
真实模型必须使用 `--model api`，本地模型服务也走这一接口。

单任务示例：

```bash
.venv/bin/sca run --task examples/code/task_light.json \
  --reader ocr --model replay --run-id offline_example
```

完整生成数据和审核记录位于 `datasets/generated/v1`，回归报告位于 `outputs/validation_*`。
生成目录被 Git 忽略，可以使用 `build-demo-data --output 新目录` 重建，既有数据目录不会被覆盖。

## 接入真实模型

将 `.env.example` 中的配置通过 shell 导出，或用 `python-dotenv` 的命令行加载；程序不自动读取其他项目的 `.env`。
代码模型和视觉模型可以使用不同服务。API 需支持兼容 Chat Completions 的协议，代码模型需支持工具调用。

```bash
export SCA_CODE_API_KEY='在服务器环境中配置'
export SCA_CODE_BASE_URL='你的兼容接口地址/v1'
export SCA_CODE_MODEL='你的代码模型'
export SCA_VISION_API_KEY='在服务器环境中配置'
export SCA_VISION_BASE_URL='你的兼容接口地址/v1'
export SCA_VISION_MODEL='你的视觉模型'
```

复制 `configs/api.example.json` 后，填写模型的每百万输入/输出 token 单价。
配置费用上限时，程序要求提供价格；本地端点可显式填 `0`。单价缺失不会被悄悄算成免费。

```bash
.venv/bin/sca run --task examples/error/task_light.json \
  --config configs/api.json --reader vlm --model api --background
```

代码模型只看到问题、截图工具返回的文本和隔离仓库内的文件。
视觉模型只看到截图，不读取源代码、参考转写、参考补丁或测试答案。

## 自己的仓库

直接传入仓库、截图、描述和固定测试命令，无需编写任务 JSON：

```bash
.venv/bin/sca repair --repo /path/to/repository \
  --image /path/to/error.png --problem '说明预期行为及当前问题' \
  --test-command 'python3 -m unittest discover -s tests -v' \
  --config configs/api.json --reader vlm --output-root /tmp/sca-runs --background
```

可重复 `--image` 提供多张截图，使用 `--protect repair_tests` 额外保护测试目录。
截图会复制到独立运行目录，状态/停止和补丁输出与 `run` 相同。

需要精确转写对照、数据分组或复用任务时，创建任务 JSON；路径相对这个 JSON 解析：

```json
{
  "task_id": "my_bug",
  "repo": "../my_repository",
  "problem": "说明预期行为及当前问题，截图中的内容无需重复转写。",
  "test_command": "python3 -m unittest discover -s tests -v",
  "protected_paths": ["tests"],
  "images": [{"image_id": "error", "path": "error.png"}],
  "provenance": "user_provided"
}
```

运行输出必须位于输入仓库之外；如果仓库包含本项目默认输出目录，使用 `--output-root /tmp/sca-runs`。
程序复制初始仓库后才执行，不直接修改输入仓库。
固定测试命令需能在错误版本上复现失败；如果它已通过，结果标为 `baseline_passed`，不声称修复成功。
该命令应由用户/任务作者指定，且能覆盖预期行为。

## 后台运行、状态和停止

`run --background` / `evaluate --background` 返回 `run_dir`、PID 和 `worker.log` 路径：

```bash
SCA_RUN_DIR='替换为返回的完整运行目录'
.venv/bin/sca status --run-dir "$SCA_RUN_DIR"
tail -n 30 "$SCA_RUN_DIR/events.jsonl"
.venv/bin/sca stop --run-dir "$SCA_RUN_DIR"
```

批量任务的进度在 `status.json`，每题的事件在 `runs/<task>_<reader>/events.jsonl`。
停止命令检查进程启动时间，向已登记的 worker 发送取消信号；worker 终止自己正在运行的沙箱进程组。

## 三路评测

```bash
.venv/bin/sca evaluate --manifest datasets/generated/v1/tasks.jsonl \
  --config configs/api.json --readers oracle,ocr,vlm --model api \
  --split dev --batch-max-cost-usd 10 --output outputs/api_dev --background
.venv/bin/sca report --runs outputs/api_dev
```

- `oracle`：截图中可见内容的准确转写；用户自带截图没有转写时不可用。
- `ocr`：固定 RapidOCR 1.4.4 / ONNX Runtime CPU 读取，按检测位置恢复文本行和近似缩进。
- `vlm`：视觉 API 读取，保留不确定字符和区域信息；JSON 无法解析时保留原始输出及错误标记。

三组使用同一代码模型、提示、任务和步数/时间预算，预先读取第一张截图，后续均可裁剪并复读。
原始任务先按组分配 dev/test，再生成不同主题；主题变化不计作新独立任务。
主报告保留所有尝试（包括失败/超时），并输出同题配对结果、测试结果、耗时、调用数、token、费用及读取 CER。
CER 对空格/换行敏感，是首次整图转写与可见参考的编辑距离除以参考字符数。
没有参考的真实截图不会产生伪造的 CER。

内置 20 题属于自编合成工程回归任务；另外提供 3 个 Boltons 公开历史修复案例。
上述 23 题的真实用户截图数量仍为 0，均为合成图，也不属于 SWE-bench；下节另行接入官方来源的派生任务。

## 公开仓库案例与完整评测

```bash
git clone https://github.com/mahmoud/boltons.git vendor/boltons
.venv/bin/sca build-public-data --source vendor/boltons --output datasets/generated/boltons_v1
.venv/bin/sca combine-manifests \
  --manifest datasets/generated/v1/tasks.jsonl \
  --manifest datasets/generated/boltons_v1/tasks.jsonl \
  --output datasets/generated/suite_v2/tasks.jsonl
```

构建器只读导出固定错误提交，实际执行原始失败与官方源码补丁后的测试。
输入仓库保留完整历史版本文件和 BSD 许可证，不含 Git 历史、参考补丁和转写答案。
三个案例分别覆盖函数复制、URL 连续导航、双向字典更新；包含 1 个代码截图任务、2 个报错截图任务。
三题及其所有主题都属于同一个 `boltons` 仓库组、同一个 `demo` 集，不作为独立测试集。
固定测试是本项目根据上游回归场景编写的标准库适配器和附加检查，没有运行上游完整 pytest 测试套件。

```bash
# 已知补丁回放，检查公开任务和独立验证链路，不调用模型
bash scripts/validate.sh --public-only

# 确认 GPU 7 空闲后，20 个合成任务 × 3 路 = 60 次真实模型修复
SCA_GPU=7 bash scripts/run_local_eval.sh

# 20 个合成任务 + 3 个公开历史案例，共 69 次真实模型修复
SCA_GPU=7 SCA_MANIFEST=datasets/generated/suite_v2/tasks.jsonl bash scripts/run_local_eval.sh
```

报告在 `report.json` / `report.md` 中按来源与 dev/test/demo 分别列出成功率、读取 CER 和调用统计。
`scheduled_runs` 与 `unfinished_or_not_attempted` 明确记录未完成数量，失败任务保留在各自分母中。
逐题保留初始复现、补丁重放、最终测试和参考文件重合诊断；文件重合不等同于定位正确。
来源与演示步骤见 [docs/public_cases.md](docs/public_cases.md) 和 [docs/demo_guide.md](docs/demo_guide.md)。
SWE-bench Multimodal 的任务和运行环境核对见 [docs/benchmark_scope.md](docs/benchmark_scope.md)。

## 官方来源数据与评测

已接入 SWE-bench Lite 原任务、测试补丁和官方评分器。固定选定的 10 题中，
9 题通过错误版本/参考补丁预检，1 题因完整测试超时单独记录。
另接入 CodeOCR 作者发布的代码数据、原渲染函数与重建指标，构建 10 份完整代码、20 张图像的小规模读取评测。

```bash
bash scripts/setup_official.sh
.venv/bin/python -u scripts/prepare_official.py build --prefix "$PWD/.runtime/python39/python"
SCA_GPU=7 bash scripts/run_official_checked.sh
```

本机执行使用独立 Python 3.9.20 前缀与 Bubblewrap，结果属于截图派生子集。
官方测试可见且受保护，未运行官方 Docker 隐藏测试协议。
本轮 27 次对照已完成：准确文本 1/9、OCR 0/9、视觉读取 1/9；剔除缓存后独立复验判定完全一致。
CodeOCR 的 40 次有效读取中，视觉模型在 1×/2×下的归一化 CER 为 1.05%/1.03%，OCR 为 5.39%/8.22%。
这些小规模结果不能证明整体修复成功率提升。
使用范围与环境适配见 [官方数据说明](docs/official_benchmarks.md)，
运行结果见 [本轮报告](docs/experiment_report_v3.md)，
原因复核见 [失败分析](docs/failure_review_v3.md)。
当前交付源码已修复本轮发现的检索、补丁缓存和 Unicode 日志问题，完整验证 60 项通过；
单题开发回归及其与原成绩的区别见 [工程修复验证](docs/post_benchmark_fixes_v3.md)。

CodeOCR 全量重建数据评测已完成：100 份代码、100 个仓库、200 张完整代码图像，OCR/VLM 共 400 次读取，耗时约 70 分钟。
普通 OCR 在 1×/2×下的归一化 CER 为 5.71%/7.98%，视觉模型为 1.13%/1.56%。
请求异常为零；视觉模型有 5 次 JSON 格式异常，原始输出仍计入评分。模型已停止，结果与诊断已归档。
结果见 [全量实验报告](docs/experiment_report_codeocr_full_v4.md) 和 [读取诊断](docs/codeocr_full_v4_analysis.md)，
运行方法见 [全量评测说明](docs/codeocr_full_v4_run.md)。v5 新增检查后，完整工程测试 83 项通过。

v5 九题回归已完成 27 次模型运行与独立重放，三组各通过 1/9，见 [回归报告](docs/experiment_report_v5_regression.md)。
50 题、6 个仓库的三组修复对照已完成 150 次：准确文本 1/50、OCR 1/50、VLM 3/50。
150 份补丁额外独立重放，判定全部一致；与 80 次截图消融合计 230 次主运行，模型均已释放。
最终结果见 [v5 总报告](docs/experiment_report_v5_supplementary.md)，运行状态和停止命令见 [补测执行说明](docs/supplementary_v5_run.md)。
截图功能消融已完成：80 次运行通过 76 次，来自同一批 20 个微型任务；实际裁剪和额外重读均为 0，因此尚不能证明这两个功能带来修复收益。四组成绩、失败轨迹与解释见 [消融分析](docs/experiment_report_v5_ablation.md)。
视觉响应可恢复含原始换行的 JSON，歧义和截断输出保留原文；对旧结果的独立诊断见 [JSON 恢复检查](docs/json_recovery_v5.md)，未替换 CodeOCR 原成绩。

配置 `screenshots.allow_crop` 可关闭裁剪；`screenshots.allow_reread: false` 在统一首次读取后移除全部截图工具。
这两个开关用于功能消融，仓库工具和数值预算保持一致。新增评测使用 [v5 配置](configs/v5-local.json)，不与旧预算成绩混合。

## 运行产物

| 文件 | 内容 |
|---|---|
| `patch.diff` | 包括新增/删除文件的 Git 补丁 |
| `result.json` | 最终验证状态、失败阶段、使用量和费用 |
| `baseline_test.log` | 初始错误版本测试输出 |
| `verification_test.log` | 干净副本应用补丁后的固定测试输出 |
| `events.jsonl` | 未压缩的模型请求/响应、工具、裁剪和执行事件；图像负载以哈希替代 |
| `conversation.json` | CoreCoder 最终上下文，可能已发生压缩 |
| `screenshots/` | 固定输入截图和可追溯裁剪 |
| `baseline/`, `baseline_check/`, `workspace/`, `verification/` | 初始快照、复现副本、Agent 工作区和独立验证副本 |

单题预算同时覆盖代码模型、上下文总结和视觉模型；每次请求先做输入/输出费用预留，
按返回 usage 和配置单价结算，无 usage 或请求失败时保留预留金额。
费用依赖用户提供的价格及供应商计费语义，不能保证与供应商最终账单逐项一致。
Agent 总时间限制包含准备和初始复现；最终独立验证另受单命令超时限制。
LLM/Agent 的“完成”声明不参与评分。

## 可选：本地真实模型验证

`scripts/local_model_api.py` 是仅绑定 `127.0.0.1`、带令牌校验的单 GPU 开发端点，
通过 Transformers 加载本地 Qwen3-VL-8B，只用于小规模验证。主项目没有 PyTorch 依赖。
Qwen 消息接入参考[官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。

本机推理环境为独立 `.model-env`，从现有环境离线复制后使用，只读加载基础权重，没有修改原环境。
`bash scripts/run_local_smoke.sh` 默认在 GPU6 上运行 2 个任务、3 种读取方式，完成后通过 trap 释放模型。
该脚本需要先确认所选 GPU 空闲、配置相应模型环境和权重路径，支持 `SCA_GPU`、`SCA_MODEL_PATH`、`SCA_PORT`。
模型启动默认最多等待 900 秒，可通过 `SCA_STARTUP_TIMEOUT` 调整；启动器每 15 秒打印等待进度，模型日志记录运行库导入、权重加载和处理器加载阶段。
模型日志、PID、评测和峰值显存记录写入 `outputs/local_smoke_*`。

`run_local_eval.sh` 复用相同模型与执行预算，默认选择全部 20 题；支持以下参数：

| 环境变量 | 默认值 / 作用 |
|---|---|
| `SCA_LIMIT` | 小测为 `2`、完整评测为 `0`（全部），也可指定正整数 |
| `SCA_MANIFEST` | `datasets/generated/v1/tasks.jsonl` |
| `SCA_SPLIT` | 不设置时选全部，也可选 `dev`、`test`、`demo` |
| `SCA_READERS` | `oracle,ocr,vlm` |
| `SCA_CONFIG` | `configs/local-smoke.json`，三组共用 |
| `SCA_GPU` / `SCA_PORT` | `6` / `18765` |
| `SCA_STARTUP_TIMEOUT` | `900` 秒 |
| `SCA_RUN_DIR` | 指定本次输出目录，要求不存在 |

批量执行日志逐题打印 `[完成数/总数]`、任务、读取方式、结果和耗时。

开发端点使用贪心生成、8192 token 上下文、单次最多 1024 新 token。
这些是烟雾验证条件，不代表模型默认长上下文配置或生产服务吞吐。

## 测试和工程边界

```bash
bash scripts/validate.sh --tests-only
```

测试覆盖费用预留、错误参数与计费、截图隐私/裁剪、OCR 实际读取、网络及环境隔离、受保护测试、
路径越界、重复匹配拒绝、子进程超时终止、新增/删除文件补丁、端到端回放与验证状态。

Bubblewrap 是本地隔离执行层；它依赖宿主 Linux 内核，第一版不把它描述为面向任意敌对代码的完整托管平台。
CoreCoder 的高级子 Agent、MCP 和原生主机工具未向本项目的任务模型注册。
训练、RL、UI 截图修复等扩展入口保留在工具/模型/任务接口上，第一版没有实现这些实验。

后续记录见 [docs/implementation.md](docs/implementation.md) 与 [docs/experiment_report.md](docs/experiment_report.md)。
完整评测与公开仓库案例进度见 [docs/experiment_report_v2.md](docs/experiment_report_v2.md)。
已人工检查的失败轨迹见 [docs/failure_review_v2.md](docs/failure_review_v2.md)。

完成的模型批次可用以下命令生成中文分析和证据归档；`--wait-seconds` 可等待后台任务完成：

```bash
.venv/bin/python scripts/archive_evaluation.py \
  --runs /path/to/run/evaluation --output docs/artifacts/new_run \
  --report docs/new_run_report.md --wait-seconds 7200
```

归档器只接受完整模型批次，失败任务保留在分母中；不会对未完成批次生成最终成绩。
