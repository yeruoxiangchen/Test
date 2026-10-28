# SWE-bench Multimodal 适配范围核对

核对日期：2026-10-22。仅检查公开资料和本机命令入口，没有构建其镜像或提交官方评测。

原论文主要面向用户可见的 JavaScript 软件，包括界面、图表、地图等视觉问题，
图像可出现在问题或测试中。它适合后续界面修复扩展，与第一版代码/终端报错转写任务的范围不同。
来源：[原论文](https://arxiv.org/abs/2410.03859)。

官方任务库当前变更记录将 test 数量记为 480，并提供 Dockerfile 与 eval 脚本生成器。
因此不能沿用旧文档中“test 一律不能本地运行”的结论；后续需按具体版本核对。
本次读取的官方任务库 HEAD 为 `8d3ed32f37af9ceb8b5778b92a0088384ed057d4`。
来源：[官方任务库](https://github.com/SWE-bench/swe-bench-multimodal-tasks)、
[固定版本变更记录](https://github.com/SWE-bench/swe-bench-multimodal-tasks/blob/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/CHANGELOG.md)。

| 项目 | 第一版当前情况 | 后续所需适配 |
|---|---|---|
| 视觉证据 | 代码、终端错误截图的文本与区域信息 | 界面布局、颜色和渲染差异的视觉解释 |
| 仓库运行时 | Bubblewrap 内系统 Python 标准库 | 官方任务级 Docker 环境及 JavaScript 依赖 |
| 验证 | 固定 unittest、私有快照和干净副本重放 | 官方 test patch、评测脚本和结果解析 |
| 数据来源 | 20 个自编合成任务、3 个公开历史修复；截图均为合成 | 官方 instance、问题图像/测试图像与版本绑定 |
| 环境验收 | 本项目实际沙箱与回归已通过 | 先在选定实例跑 gold patch、核对浏览器与测试日志 |

官方变更记录显示，部分任务需要 Chrome、Puppeteer、Cypress 或特定 Node 版本，
并记录了浏览器版本、外部 CDN 和输出流顺序造成的环境失败。
这类基础设施问题需要与 Agent 修复失败分别统计。
来源：[官方变更记录](https://github.com/SWE-bench/swe-bench-multimodal-tasks/blob/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/CHANGELOG.md)。

本机本次终端的 `PATH` 中没有发现 `docker` 命令；未验证 Docker daemon 或浏览器评测环境。
第一版保留现有任务、模型、工具和报告接口，以后可增加 Docker 执行后端与官方任务适配器。
当前报告没有 SWE-bench Multimodal 成绩，不将 Boltons 案例换算成该基准的分数。
