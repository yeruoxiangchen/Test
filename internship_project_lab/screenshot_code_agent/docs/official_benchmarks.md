# 官方来源数据与运行方法

本轮增加 SWE-bench Lite 任务适配，以及 CodeOCR 作者发布数据的独立读取评测。
来源固定在 [BENCHMARK_UPSTREAM.json](../BENCHMARK_UPSTREAM.json)。
现有 20 个自编题与 3 个 Boltons 案例继续用于工程回归和演示。

## SWE-bench Lite 截图派生任务

从官方 `test` split 的 300 题中，按固定规则选择 Pytest 和 SymPy 各 5 题。
规则为：按 instance ID 排序，取前五个包含 2–35 行、至多 2200 字符 fenced block 的任务。
先按仓库分组，将 Pytest 分为开发组、SymPy 分为测试组，再生成浅色/深色截图。
这两个分组是本项目内部划分，原始来源仍为官方 test split。

每题保留完整错误提交的源码、原 Issue、原测试补丁、参考源码补丁和官方评分列表。
从原 Issue 中移出一个预先选定的代码/输出块，用截图替代；截图没有来自参考修复的新增内容。
所有方法接收同样的周围描述，同样的仓库、测试和代码模型预算。
没有人工声称所有选定任务的图像都必不可少。

当前预检：10 个固定任务中 9 个验证了错误版本失败、参考补丁通过；
`sympy__sympy-11870` 的完整三角函数测试在 120 秒上限内未结束，单独记为环境/预算预检失败。
任务保留在 `all_tasks.jsonl` 与审核报告中，模型评测使用已通过预检的 `tasks.jsonl`。
它未因模型解不出而被删除；首次预备模型调用发生在另一个 Pytest 任务。

```text
datasets/generated/swebench_lite_v3/
  selection.json              # 固定任务 ID、原始 split、选择规则和数据指纹
  selected_records.json       # 原始记录；含参考答案，不向 Agent 挂载
  tasks.jsonl                 # 9 个通过预检的任务
  all_tasks.jsonl             # 10 个原始选定任务
  build_report.json           # 包括预检失败的全部记录
  <instance_id>/
    repo/                     # Agent 输入仓库，装入官方测试补丁
    task_light.json           # 当前工具、运行时与截图输入
    benchmark.json            # 仓库之外的可信评分元数据
    reference.patch           # 仓库之外的参考补丁
    baseline_test.log
    reference_test.log
    audit.json
    screenshot_light.png
    screenshot_dark.png
```

评分加载未修改的官方日志解析器与 `get_eval_report`，同时核对 FAIL_TO_PASS 和 PASS_TO_PASS。
日志完整保存；只限制送给代码模型的工具反馈长度。模型异常后已产生的修改也会导出并独立验证。
官方测试补丁已安装在 Agent 可见仓库中，并设为只读保护；这不是隐藏测试协议。
参考源码补丁、评分元数据和配对参考文本在仓库之外，不向代码工具挂载。

## 运行环境

本机没有 Docker CLI/daemon。Docker Hub 直连和代理连接失败，Daocloud 缓存站请求返回 403。
因此本次使用独立的 Python 3.9.20 前缀和 Bubblewrap，挂载只读依赖、可写仓库副本，禁用网络与 GPU 设备。
依赖为固定的兼容组合，见 [任务环境锁定文件](../configs/swebench-python39.lock.txt)，
没有逐项复现每个官方 Docker 镜像。

测试调用提取自官方 `eval.sh`，测试补丁在构建时装入，环境安装与 Git 重置步骤由构建器处理。
Pytest 原源码需要安装时生成的版本文件；构建器用上游 `setuptools_scm` 和该任务的版本字段生成并保护它。
官方评分器的纯评分模块单独加载，跳过会导入 Docker/云端收集器并自动加载 dotenv 的包初始化器。
官方源码文件保持原样。

这属于 **SWE-bench Lite 截图派生子集，使用官方测试与评分器、适配运行环境**。
结果需要与原始排行榜成绩区分。Docker/OCI 执行入口已提供，但本机未完成该路径的真实镜像运行验证。

## CodeOCR 读取评测

使用作者仓库 `reconstruction/dataset/dataset.json` 中的 100 个代码文件。
按 ID 排序选择前 10 个不超过 80 行、2200 字符的完整文件，来自 10 个仓库。
用上游渲染函数和退化函数生成 1×、2×共 20 张图片，仓库分组先于图像变体。
初版字体不支持部分中文；最终读取数据位于 `datasets/generated/codeocr_v3_unicode`，
使用固定版本 Noto Sans Mono CJK 字体，按字体字符映射检查全部可见字符。
修正前后使用完全相同的代码文件与参考文本；旧图片和 40 次读取保留为数据质量诊断。
SWE-bench 中的数学字符 `ℯ` 也在对应任务首次送入模型前修正，三组共享相同输入。
1×/2×为缩小后放大回原尺寸的退化倍数，不表示视觉 token 数减半。

直接比较 CPU OCR 与视觉模型，共 40 次读取。
采用上游归一化 CER、WER、行相似度，补充原始 CER、精确匹配、Python 语法检查。
全部 10 个参考文件可通过本机 Python 语法解析。语法检查不会运行这些文件。

上游辅助模块引用了缺失的 `AIHUBMIX_BASE_URL` 常量。
适配器通过 AST 加载两个未修改的纯函数，避免无关的 API 配置导入；渲染模块正常导入。
本轮没有复现其所有下游理解实验，也没有调用 LLM judge。

## 重建与运行

本机主项目使用 Python 3.12；源代码归档的安全提取需要相应 `tarfile` filter 支持。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
bash scripts/setup.sh
bash scripts/setup_official.sh

# 会下载固定错误提交，执行全部十题的错误版本/参考补丁预检；可后台运行
.venv/bin/python -u scripts/prepare_official.py build \
  --prefix "$PWD/.runtime/python39/python"

# 一张空闲 GPU 上运行三路修复与 CodeOCR 读取
SCA_GPU=7 bash scripts/run_official_checked.sh
```

已存在的任务与审计不会直接覆盖。需要重做某次失败预检时，先将该任务目录保留到另一个审核目录，
再重新构建；源代码归档缓存可以复用。变更任务选择应使用新的数据版本，并保留原始清单。

只复现一题，可以创建一份只包含其 `task_file` 的清单，然后设置 `SCA_MANIFEST`、`SCA_LIMIT=1`。
`SCA_SKIP_CODEOCR=1` 可以仅运行修复评测。数据构建、模型推理和已知补丁回放是不同检查。
只运行读取评测可执行 `SCA_GPU=7 bash scripts/run_codeocr_local.sh`。
`run_official_checked.sh` 在模型运行前检查字形，并在结束后导出剔除缓存的补丁、进行独立复验。
不要在长任务运行中原地修改其脚本或运行源码。

已完成的 [三路报告](experiment_report_v3.md)、[失败复核](failure_review_v3.md) 和
[对照后的工程修复](post_benchmark_fixes_v3.md) 分别保存了原成绩与后续回归。
原 27 次比较的源码快照在 `docs/artifacts/v3/source_snapshot.zip`；当前源码包含后续工程修复，
重跑时应使用新输出目录，不能将后续结果混入原成绩。

## 原始资源

- [SWE-bench Lite 数据](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite)
- [SWE-bench 评分工具](https://github.com/SWE-bench/SWE-bench)
- [SWE-bench 官方任务与环境定义](https://github.com/SWE-bench/swe-bench-tasks)
- [CodeOCR 作者实现与数据](https://github.com/YerbaPage/CodeOCR)
- [SWE-bench Multimodal](https://www.swebench.com/multimodal)：本轮尚未运行，后续需要视觉布局理解与 JavaScript/浏览器环境。
