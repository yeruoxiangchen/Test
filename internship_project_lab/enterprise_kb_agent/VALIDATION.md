# v0.1 验证记录 · 2026-10-28

> 以下保留初版验证现场。后续已完成真实单卡模型接入和评测基础设施，最新状态见 [EVAL_PROGRESS_20261028.md](EVAL_PROGRESS_20261028.md) 与 [EVALUATION.md](EVALUATION.md)。不要将下文的“未配置模型”当成最新评测状态。

实现范围：第二项“企业本地知识库与问答系统”。独立编写应用、业务接口和工具调用 harness，参考成熟产品流程，不 fork 整个产品。

## 运行状态

- 本地预览：`http://127.0.0.1:8768`，由 `bash scripts/run.sh start` 后台启动。
- 当前模式：`extractive`，提供 BM25 关键词检索和带引用的原文摘录，无 GPU 使用。
- 3 份虚构示例资料全部完成入库。
- 管理员凭据仅保存到 `/data/zjr/enterprise_kb_agent/credentials.txt`，权限 600；本记录不包含密码。
- 源码：`/home/zjr/Tracker/internship_project_lab/enterprise_kb_agent`。
- 数据、环境、缓存及测试产物：`/data/zjr/enterprise_kb_agent`。

## 已执行检查

| 检查 | 结果 | 含义 |
|---|---|---|
| Python 依赖安装 / pip check | 通过 | 独立 Python 3.12.7 venv；已保存 requirements.lock |
| Python 语法 / JavaScript 语法 | 通过 | Python AST 解析与 node --check |
| pytest | 18 passed，70.78 秒 | API、权限、解析、检索、harness 控制行为 |
| 演示问题 | 12/12 完成；首位检索 12/12 指向预期资料 | 仅虚构小样例的功能检查，非真实业务准确率 |
| 浏览器端到端 | 通过，0 page errors | 实际操作网页的完整流程 |
| 后台服务 | 已启动，健康检查通过 | 当前回环地址 CPU 预览服务 |

单元/集成测试覆盖：

- 上传、去重、索引、删除与向量清理、重新索引；PDF 页码与 DOCX 表格。
- 无效文件、空文件、路径清理、上传限制、跨来源写请求。
- 成员与管理员知识库隔离、原文下载权限、会话与运行记录所有权。
- 无证据返回、取消、一会话一个活动任务、反馈和事件 SSE。
- 脚本模型驱动的真实 harness 流程：search → read_chunk → finish。
- 伪造引用被拒绝后纠正、缺少行内引用、未知工具、轮数限制、取消与模型错误。
- 检查点恢复保留完整工具消息和已消耗轮数；已完成答案不重复写入。
- HTTP 模型接口请求包含真实工具 schema；向量检索的模型配置隔离与权限过滤。

浏览器验证覆盖：登录 → 建库 → 上传 → 后台入库 → 提问 → 回答 → 来源预览 → 评价 → 管理员反馈列表 → 刷新后会话保留 → 移动端无横向溢出。

## 报告和截图

- [演示问题报告](/data/zjr/enterprise_kb_agent/artifacts/evaluation-20261028-050510/report.json)
- [浏览器报告](/data/zjr/enterprise_kb_agent/artifacts/browser-20261028-050648/report.json)
- [问答页面](/data/zjr/enterprise_kb_agent/artifacts/browser-20261028-050648/chat.png)
- [文档管理](/data/zjr/enterprise_kb_agent/artifacts/browser-20261028-050648/documents.png)
- [移动端](/data/zjr/enterprise_kb_agent/artifacts/browser-20261028-050648/mobile.png)

首次截图检查发现服务器缺少中文字体，随后将 Noto Sans CJK 和许可放入本项目缓存，使用独立 fontconfig 再次完成浏览器验证。未修改系统字体。

## 尚未验证的部分

- 未配置真实大模型服务；没有实际 LLM 生成质量或 Agent 任务成功率成绩。模型调用逻辑通过脚本响应及 HTTP mock 验证。
- 未部署真实语义向量模型；默认运行的是 BM25。向量/RRF 实现已做隔离与权限测试，未测真实召回效果。
- Dockerfile/Compose 已提供，没有在本次运行中构建容器。
- 未接入真实企业数据，未进行并发压测、生产部署或模型训练。

下一步可以在 `/data/zjr/enterprise_kb_agent/private.env` 配置支持工具调用的模型服务，重启后执行 `bash scripts/run.sh evaluate --agent`，再使用真实领域资料进行人工审核。此步骤不会自动占用 GPU；需要另行确定已有接口或模型部署方式。
