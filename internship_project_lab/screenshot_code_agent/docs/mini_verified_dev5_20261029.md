# 2026-10-29：五题失败后继续修复执行与验收流程

针对mini_session五题0/5，新增显式`mini_verified`模式；原输出、冻结代码和分数保持不变，mini上游DefaultAgent未改动。这是本地适配及验证流程的改版，不是原样mini成绩。

完成更新：最终2/5，详见[逐题结果与预算分析](mini_verified_dev5_results_20261030.md)。下方启动状态保留历史记录。

## 改动

- 非字符串、NUL命令拒绝本次执行，记录工具错误并返回模型，保留额度计费；真实API错误、执行异常分别报告。
- 新增`edit_file`工具：旧文本必须唯一匹配，在任务自己的Python中解析新源码，语法不合法则不写入。仍保留bash执行能力，不声称所有shell修改都具有事务保证。
- 新增`check_patch`工具，周期性每12个动作及提交时自动检查。按原始基线独立重放，运行可见行为断言和已有公开测试模块，对比原代码测试结果，区分旧失败与新增回归。没有比较依据、收集错误或超时不能视为通过。
- 通过检查的补丁由控制器保存；退出时新候选未通过则交付最后通过检查的版本，并记录选择来源。原工作区最后候选另存。若从未通过检查，仍保留失败候选并按原分母评分，不假装成功。
- 补充已知五题开发断言，包括PyVista cells替换、pvlib正常构造与可选参数、无效参数组合及相对导入行为。所有新增断言属于开发监督；最终隐藏评分不提前执行。
- 备份排除规则补充`.tmp`，新增`recoverable_tool_errors`与`agent_errors`汇总。

## 检查及边界

短检查覆盖真实mini循环遇到NUL后继续、安全编辑不破坏文件、公开回归失败阻止提交、最后好候选在额度结束后保留，以及原5种OCI环境。

已离线验证上一轮5份失败补丁均被新验收拒绝；此前v4成功的astroid-984、pydicom-793补丁作为正向对照仍能通过。这些是已知补丁验证，不代表新模型运行会成功。测试阶段发现并修复测试夹具的pytest解释器选择及PyVista插件自动加载问题，真实PyVista仓库验证仍保留原插件。

冻结版本真实环境预检最终 **85项通过，0失败、0跳过**，耗时132.52秒。包含全部5份旧失败补丁的拒绝及2份既有成功模型补丁的接受。

检查输出：`/data/zjr/screenshot_code_agent/mini_verified_checks_20261029/`。正式启动必须通过冻结版本真实环境预检，收据在新运行目录`session_preflight.json`。Ruff及Shell语法检查通过。

## 运行

入口：`scripts/run_mini_verified_dev5.sh`；配置：`configs/mini-verified-dev5.json`。
同样5题、原Qwen3-Coder BF16、TP4、两题并发，96次模型调用和1800秒任务预算保持不变。公开检查消耗任务时间与工具预算；原始评分协议不变。新输出全部在`/data/zjr/screenshot_code_agent/mini_verified_dev5_20261029/`。

```bash
watch -n 15 'bash /home/zjr/Tracker/internship_project_lab/screenshot_code_agent/scripts/run_mini_verified_dev5.sh status'
```

后台服务`sca-mini-verified-dev5-20261029.service`已启动，控制器PID2060247，systemd确认active/running。实际启动参数指定GPU2、4、5、7，显存阈值每卡19723MiB，满足条件后自动加载。模型结果尚未产生。

```bash
journalctl --user -fu sca-mini-verified-dev5-20261029.service -o cat
```

本版仍是无截图的仓库修复后端开发回归。不能把已知五题上新增的断言或测试通过数写成未见任务成功率。
