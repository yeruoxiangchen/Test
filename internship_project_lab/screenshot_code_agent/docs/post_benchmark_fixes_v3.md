# 对照结束后的工程修复

2026-10-22。原 27 次 SWE-bench 来源修复对照已归档，运行期源码与启动指纹一致。
归档中的 `source_snapshot.zip` 保留产生这些成绩的实现。
当前交付源码随后加入三个修复：字面文件 grep、剔除生成缓存的补丁导出、按物理行读取 Unicode JSONL。
CoreCoder 上游源码、模型、任务选择和 Agent 决策循环均未修改。

修改差异与验证证据见 [post_v3 归档](artifacts/post_v3/validation.json) 和
[源码差异](artifacts/post_v3/runtime_fixes.patch)。

| 检查 | 结果 |
|---|---|
| 独立源码副本中的相关回归 | 19 项通过，3.04 秒 |
| 交付源码的完整验证 | 60 项通过，17.30 秒；Ruff、编译、Bash 语法和隔离探测均通过 |
| 固定开发题 `pytest-dev__pytest-5227` / OCR | 原对照失败；修复后单次独立回归通过 |
| 同题代码/压缩调用 | 原 18 次，修复后 8 次 |
| 同题工具请求 | 原 17 次，修复后 6 次 |
| 修复后耗时 | 106.799 秒，含模型与测试，不含模型启动 |
| 导出补丁 | 510 字节，仅修改 `src/_pytest/logging.py`，干净副本测试与官方评分通过 |
| 此开发回归 PyTorch 峰值已分配显存 | 17.29 GiB，一张 RTX 3090；不代表其他任务的峰值或推理服务上限 |

代码模型、OCR、截图、工具/调用/时间预算及模型参数与原对照相同；本地服务端口从 18768 换为 18769，完整配置字节因此不同。
此回归使用隔离源码副本和单独输出目录，
不追加到原来的三路 27 次成绩，不将一题的改善外推为修复成功率提升。没有据此调整已固定的 SymPy 测试组。
该结果说明工程修复在一个曾受检索问题影响的开发题上可以完成闭环，并不能分离每个改动的独立因果效果。

复现一题并自动检查输入字形、导出补丁及独立复验：

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
SCA_GPU=7 SCA_SKIP_CODEOCR=1 \
  SCA_MANIFEST=datasets/generated/official_pilot_v3/tasks.jsonl \
  SCA_LIMIT=1 SCA_READERS=ocr bash scripts/run_official_checked.sh
```

上述开发清单来自本轮预备运行，只含一个固定任务。若需自己创建，
从 `datasets/generated/swebench_lite_v3/tasks.jsonl` 取相应记录时，应将 `task_file` 转成绝对路径。
已有验证结果可直接查看 [dev_report.md](artifacts/post_v3/dev_report.md)，无需重新占用显卡。

原批次发生的汇总错误通过独立修正的报告模块恢复，未修改任何原始事件、补丁或结果；
恢复脚本和报告模块的指纹也已保存。原归档的源码一致性保护会拒绝以修改后的源码重写该批次归档；
当前源码产生的新结果应使用新目录和新的实验版本。
