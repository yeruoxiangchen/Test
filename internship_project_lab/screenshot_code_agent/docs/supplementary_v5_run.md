# v5 补充评测执行说明

本轮补充修复任务、截图功能消融和异常处理测试。CodeOCR v4 的 400 条原始成绩不变。

2026-10-23 状态更新：两组共 230 次主运行和最终归档均已完成，模型均已释放。
跨阶段代码更新和 UTC 时间线见 [连续实验记录](experiment_history.md)。下方后台命令保留供查询和复现时参考。

## 已完成

- 旧 9 题 × oracle / OCR / VLM 的 27 次同预算回归与独立补丁重放：三组各通过 1/9，均为 pytest-5227。见 [回归报告](experiment_report_v5_regression.md)。
- 83 项工程测试通过，包括真实 Bubblewrap 隔离、超时清理、保护测试、坏图片落盘、JSON 恢复、首次读取后移除截图工具。
- CodeOCR 保存的 5 条格式错误中，3 条可在不改写代码的前提下恢复；另 2 条保留原文和失败状态。无新模型调用，见 [解析检查](json_recovery_v5.md)。
- 扩展集 60 个候选全部预检，52 个通过；按冻结顺序取 50 个。截图字形全部可见，没有因模型结果筛题。
- 扩展仓库三路评测已完成 150/150 次：oracle 1/50、OCR 1/50、VLM 3/50；150 份补丁额外独立重放，5 次通过，全部与原判定一致。GPU 6 模型已释放，详见 [总报告](experiment_report_v5_supplementary.md)。
- 实际 Chromium 截图 20 张：10 张 CodeMirror 代码、10 张 xterm 报错。终端回放实际隔离测试输出。来源为受控微型任务，自然用户截图数仍为 0。
- 截图功能消融已完成 80/80 次，通过 76 次；四组分别为 20/20、18/20、19/20、19/20。裁剪与额外重读调用均为 0，分差不能归因为这两个操作。GPU 7 模型已释放，详见 [完整消融分析](experiment_report_v5_ablation.md)。

| 扩展修复来源 | 进入评测 |
|---|---:|
| pytest-dev/pytest | 7 |
| sympy/sympy | 26 |
| sphinx-doc/sphinx | 11 |
| pylint-dev/pylint | 2 |
| pallets/flask | 1 |
| mwaskom/seaborn | 3 |

pytest/SymPy 为开发组，其余四仓库为测试组；分组先于截图变体。原始公开来源均为 SWE-bench Lite test split，当前内部划分不等于官方训练/测试协议。

## 后台任务

| 服务 | GPU | 内容 |
|---|---:|---|
| `sca-v5-extended-eval.service` | 6 | 50 题 × 准确文本 / CPU OCR / VLM = 150 次修复，然后额外重放 150 份补丁 |
| `sca-v5-ablation-eval.service` | 7 | 20 题 × 原图 / 降质图 × 只读一次 / 允许裁剪重读 = 80 次修复；另有 2 次不计入主表的端到端预检 |
| `sca-v5-report.service` | 无 | 等待两组完成，核对数据和配置、归档证据，自动生成报告 |

每次调用使用现成 Qwen3-VL-8B-Instruct，代码与视觉输出上限 2048 token；18 次代码/压缩调用、50 次工具、最多 3 次截图读取、1200 秒，单命令和请求各 180 秒。Agent 压缩阈值 8192，服务上下文 16384。

同一比较的模型和数值预算一致。旧 v3/九题回归使用代码输出 1024、压缩阈值 4096，因此新实验单独报告。只读一次组仍先获得同一张整图证据，再移除截图工具；工具是否实际被使用会单独统计。

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent

# 总进度、当前任务、已完成数；不启动模型，也不占显卡
watch -n 15 '.venv/bin/python scripts/run_v5_supplementary.py status --output outputs/v5_supplementary'

# 逐题进度；尚未开始的阶段日志可能暂时不存在
tail -F outputs/v5_supplementary/extended/three_readers/evaluation.log \
  outputs/v5_supplementary/ablation/clean_available/evaluation.log \
  outputs/v5_supplementary/ablation/clean_initial_only/evaluation.log \
  outputs/v5_supplementary/ablation/degraded_available/evaluation.log \
  outputs/v5_supplementary/ablation/degraded_initial_only/evaluation.log \
  outputs/v5_supplementary_report.log

# 模型启动或请求日志
tail -F outputs/v5_supplementary/extended/model_server.log \
  outputs/v5_supplementary/ablation/model_server.log

# 仅停止本项目这轮任务，包括其模型和子进程
systemctl --user stop sca-v5-extended-eval.service \
  sca-v5-ablation-eval.service sca-v5-report.service
```

每组结束后自动释放显卡；失败和取消会保留状态与日志，不会报为全部成功。服务使用 `--collect`，结束后 unit 可能从 systemd 中消失，以输出目录里的 `status.json` 和 `analysis_status.json` 为准。

最终自动报告为 `docs/experiment_report_v5_supplementary.md`，证据归档为 `docs/artifacts/v5_supplementary/`。`outputs/v5_supplementary/analysis_status.json` 的 `complete: true` 表示归档和最终报告也完成；仅模型日志 HTTP 200 或某组结束不足以表示全套评测完成。

## 可复现入口与边界

活动运行使用独立冻结源码 `outputs/v5_eval_source/`，共 74 个源码/配置/测试文件，指纹与压缩包同名保存。代码主目录后续的文档和分析脚本修改不会进入这些运行。

扩展数据：`datasets/generated/swebench_extended_v5/`；候选、最终清单、每题错误/参考版本测试及历史失败均保留。输入截图、转写、外部评分元数据和依赖配置已在模型调用前生成指纹。测试可见、受保护；参考补丁和评分元数据不进入 Agent 工作区。使用适配 Python/Bubblewrap 环境，未运行官方 Docker 镜像，不是排行榜成绩。

预检和适配限制见 [预检说明](extended_preflight_v5.md)。20 个微型任务主要验证工程闭环，不代表未见真实仓库。没有自然用户截图、SFT/RL 或策略创新结论。

构建脚本为 `prepare_extended.py`、`build_tool_captures.py`、`prepare_v5_ablation.py`；它们拒绝覆盖既有数据。失败预检仅可显式 `--retry-failed`，历史目录和私有依赖前缀会先归档。模型结果出现后不要改本轮候选或输入。

截图采集依赖单独固定在 `configs/capture/package.json` 和 `package-lock.json`：Playwright 1.56.1、CodeMirror 5.65.20、xterm 5.5.0；浏览器为对应 Chromium 141.0.7390.37。新环境可以先复制这两个文件到 `.capture-runtime/`，运行 `npm ci --prefix .capture-runtime --ignore-scripts`，再使用该目录的 Playwright CLI 安装 `chromium --only-shell`，将 `PLAYWRIGHT_BROWSERS_PATH` 指向项目 `.cache/capture-browsers`。

本地模型依赖另存 `configs/model-python311.freeze.txt`，来自项目独立 Python 3.11 环境的实际包元数据，不是对其他项目环境的导出。任务运行时的需求与实际依赖位于 `.runtime/extended_v5/*.requirements.txt` / `*.freeze.txt`，最终归档一并保存。模型权重沿用本机只读文件，不包含在源码包里。

`max_wall_seconds` 控制 Agent 调用预算；导出补丁后，独立验证使用自己的单命令超时，因此完整任务墙钟可能超过 1200 秒。服务还设有外层运行时上限，停止服务会清理其模型和测试子进程。
