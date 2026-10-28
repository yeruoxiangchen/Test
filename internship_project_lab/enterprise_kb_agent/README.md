# 知页 · 企业本地知识库与问答系统

从独立代码仓库组织的可运行应用：管理员上传文档，成员提问、查看原文依据，管理员处理反馈。包含网页、身份与权限、持久化入库队列、检索、模型工具调用 harness、运行记录、测试与部署入口。

应用流程参考 QAnything 的文档问答，以及 ChatterMate 的业务系统组织方式；没有 fork 或复制它们的应用代码。当前使用自编写的有界工具调用循环，没有将其称为 LangGraph/Agno 实现。

## 快速使用

```bash
cd /home/zjr/Tracker/internship_project_lab/enterprise_kb_agent
bash scripts/run.sh setup
bash scripts/run.sh init
bash scripts/run.sh demo
bash scripts/run.sh start
bash scripts/run.sh status
```

浏览器打开 `http://127.0.0.1:8768`。管理员账号密码保存在 `/data/zjr/enterprise_kb_agent/credentials.txt`，权限为 600；`init` 不覆盖已有账号。查看凭据：

```bash
cat /data/zjr/enterprise_kb_agent/credentials.txt
```

在自己的电脑访问远程服务器时，可以使用 VS Code 端口转发，或者：

```bash
ssh -N -L 8768:127.0.0.1:8768 zjr@你的服务器地址
```

然后在本机浏览器打开相同地址。默认只监听回环地址。

前台运行、停止和查看日志：

```bash
bash scripts/run.sh serve
bash scripts/run.sh stop
tail -f /data/zjr/enterprise_kb_agent/logs/server.log
```

`start` 以独立进程组后台启动；`stop` 校验 PID、启动时间和命令行后，仅终止本项目服务。已完成任务和会话保留。自动重启服务建议采用容器或自行配置 systemd。

## 两种模式

| 模式 | 行为 | 需要的资源 |
|---|---|---|
| `extractive`，默认 | 检索并摘录原文，附带来源；页面明确显示“原文检索” | CPU，无模型、无 API Key |
| `agent` | 模型决定检索、读取证据或再次检索，调用 finish 输出带引用的回答 | 支持工具调用的 chat-completions 兼容模型服务 |

原文模式用于立即体验入库、检索、引用与系统流程，不是模型问答效果，也不证明 Agent 的任务成功率。模型模式调用失败会显示失败，不会悄悄替换为原文模式。

### 配置模型

把 `private.env.example` 复制到 `/data/zjr/enterprise_kb_agent/private.env` 并设置权限 600。在文件中填写：

```bash
KB_MODE=agent
KB_LLM_BASE_URL=http://127.0.0.1:8000/v1
KB_LLM_MODEL=你实际部署的模型名称
KB_LLM_API_KEY=如果服务需要则填写
```

模型需支持 `tools` / `tool_calls` 和 `tool_choice=required`，包含 search、read_chunk、list_documents 和 finish。正式评测还需准确的 Token 计数接口 `KB_TOKENIZE_URL`。配置后重启：

```bash
bash scripts/run.sh stop
bash scripts/run.sh start
bash scripts/run.sh doctor --live
bash scripts/run.sh evaluate --agent
```

最后一个命令会实际调用模型，在独立目录执行 12 道演示问题；可能耗时，输出逐题进度。只填接口并不能证明该模型的工具调用质量，需要完成此验证及真实资料测试。日常启动应用不会自动申请 GPU；需要时可显式执行 `bash scripts/run.sh model start --gpu 1`。模型服务与网页、评测队列分别管理，不使用其他项目密钥。

2026-10-28 已完成真实 Qwen3-8B 工具调用连通验证，并构建 40 份公开技术资料、100 开发题/300 测试题的冻结候选评测集。详细数据来源、独立人工复核边界、指标、固定 RAG/Agent 对照和后台监视命令见 [EVALUATION.md](EVALUATION.md)。演示题通过不等于真实问答质量达标。

### 可选语义向量检索

默认检索采用 SQLite FTS5/BM25，对中文做双字切分，对英文做词切分。它是关键词检索，不冒称语义检索。

配置 `/embeddings` 兼容服务后启用稠密检索与 BM25 的 RRF 融合：

```bash
KB_EMBED_BASE_URL=http://127.0.0.1:8001/v1
KB_EMBED_MODEL=你实际部署的向量模型
KB_EMBED_API_KEY=如果服务需要则填写
```

修改地址或模型后，在文档页重新索引。索引按地址和模型隔离；失败时保留 BM25，并展示降级提示。当前向量检索采用小规模精确扫描，适用于原型和小知识库；没有宣称支持大规模企业向量库。相似度过滤值和 BM25 覆盖率是初始工程配置，尚未经过业务集调参。

## 已实现功能

- 用户登录、退出、管理员/成员角色、Cookie 会话；管理员创建成员。
- 多知识库；全员可见或仅管理员可见。检索、文档读取、下载和模型工具均执行权限过滤；会话与任务归属单个用户。
- PDF、DOCX、TXT、Markdown、CSV 上传；同库内容去重，持久化队列，解析状态、错误信息、重新索引和删除。
- 文档解析在限时、限内存子进程运行；PDF 保留页码，DOCX 保留段落与表格文本；文本分块含重叠。
- 问答与历史会话、引用原文预览/下载、任务停止、执行记录及回答反馈。
- 模型 harness：工具白名单、参数验证、检索后才能读取片段、最多 6 轮模型请求/12 次工具调用、来源编号校验、连接超时与有限重试。
- 默认 `bounded_v2` 策略按当前进度开放工具：搜索后先读证据，最多两个不同查询，重复查询和已读片段不重复执行；预留最后两轮用于提交答案和修正格式。
- 已读完整原文在上下文中各保留一份，加入上下文前准确计算 Token 预算，不能容纳的新证据明确拒绝加入；当前问题和已读证据不被截断。完整模型和工具响应仍保存在事件轨迹中。
- 持久化证据状态与来源；服务重启后重新排队未完成任务，从完整轮次检查点恢复。已完成答案和状态原子写入，防止重复落库。不同策略的旧检查点明确拒绝混用。
- REST API 与执行事件 SSE；网页当前使用轮询更新状态，答案生成完成后显示，不模拟逐 token 流式输出。
- 运行脚本、Dockerfile/Compose、接口和 harness 测试、浏览器 E2E、独立演示评估命令。

## 文件与运行数据

```text
enterprise_kb_agent/
  kb/app.py          API、身份与权限边界
  kb/agent.py        自编写 Agent harness、工具和引用校验
  kb/bounded.py      v2 工具阶段限制、证据去重和上下文准入
  kb/retrieval.py    BM25、可选向量/RRF 检索
  kb/ingestion.py    上传与索引任务
  kb/extract.py      受资源约束的文档解析
  kb/worker.py       入库与问答后台任务
  kb/db.py           SQLite 状态与事件存储
  kb/web/            中文响应式网页
  kb/cli.py          初始化、启动、监视与评估
  kb/context.py      总 Token 预算与完整工具消息保护
  kb/benchmark/      证据标注、冻结数据、批量推理、评分与配对报告
  tests/            自动化验证
  scripts/          运行与浏览器验证
```

数据库、上传文件、运行环境、缓存、测试和截图全部位于 `/data/zjr/enterprise_kb_agent/`。`run.sh` 设置 TMPDIR、TORCH_HOME、HF_HOME、HF_HUB_CACHE、TORCH_EXTENSIONS_DIR、MPLCONFIGDIR 等；Tracker 仅存源码和小文档。

## 验证命令

```bash
bash scripts/run.sh test
bash scripts/run.sh evaluate
bash scripts/run.sh browser-setup
bash scripts/run.sh browser-test
```

接口与 harness 测试使用临时隔离数据库，包括越权、来源伪造、预算终止、取消、恢复和模型错误。模型行为在单元测试中由脚本化响应驱动，这是协议/工程验证，不是实际模型成绩。

浏览器 E2E 自动创建独立测试服务，验证登录、建库、上传、入库、提问、引用、反馈、会话恢复及窄屏显示，报告和截图写入 `/data/.../artifacts/browser-*/`。`evaluate` 使用代码内自编写的虚构示例资料，不能当作真实企业准确率。

## Docker

先准备已有运行目录，并导出与目录所有者一致的 UID/GID：

```bash
export KB_UID=$(id -u)
export KB_GID=$(id -g)
docker compose build
docker compose run --rm knowledge-desk python -m kb.cli init
docker compose run --rm knowledge-desk python -m kb.cli demo
docker compose up -d
```

不要同时启动本地服务和指向同一数据库的容器服务。容器内的模型地址需在容器网络中可访问；不能直接把宿主机的回环地址当作容器外服务地址。

## 当前边界

- 当前是单组织、小规模、单应用进程版本；没有 SSO、复杂部门权限、多实例任务队列或生产高可用承诺。
- 扫描 PDF、图片 OCR、旧版 `.doc`、复杂版面表格还未接入；不支持的文档明确报错。
- 来源校验验证“编号存在、片段读取过、权限有效、行内引用一致”；它不能自动证明回答的每个结论都受原文支持。
- 已接通单卡 Qwen3-8B 并验证工具协议；实际业务问答质量和任务完成率仍需完整评测与独立复核。语义向量服务尚未部署。
- 文档只作为数据，不开放模型执行 Python、Shell 或任意写入工具；本版工具只读知识。
- 反馈用于管理员审阅，尚未用于 SFT/RL。无模型训练任务。

后续依据开发集配对结果和独立复核处理检索漏召回、回答缺乏依据与跨文档问题，再接入向量服务形成检索策略对照。冻结测试集不用于调参。

## 参考

- 产品流程参考：[QAnything](https://github.com/netease-youdao/QAnything/tree/qanything-v2)、[ChatterMate](https://github.com/chattermate/chattermate.chat)。
- 实际依赖：FastAPI、Uvicorn、HTTPX、SQLite FTS5、pypdf、python-docx；测试使用 pytest、Playwright。
- 接口与解析依据：[FastAPI 文件上传](https://fastapi.tiangolo.com/tutorial/request-files/)、[SQLite FTS5](https://www.sqlite.org/fts5.html)、[pypdf 文本提取](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)。
