# 公开仓库案例

这三题的错误源码和参考修复来自 Boltons 的实际历史提交。截图为本项目合成渲染，
固定测试是根据公开回归场景编写的 `unittest` 适配器，并增加兼容性检查。
它们用于第一版开发演示；三个任务共享一个仓库分组并全部置于 `demo`。

| 任务 | 错误提交 | 官方修复 | 输入 | 固定测试覆盖 |
|---|---|---|---|---|
| `boltons_copy_function_kwdefaults` | `0341ab3d79e6430c0e6a4149df9c92679988a534` | [c2b04b4](https://github.com/mahmoud/boltons/commit/c2b04b4cb16410d7bf9ff781c78dee3ca908dc32) | 代码截图 | 关键字默认值、必需关键字、闭包、普通默认值、自定义属性 |
| `boltons_url_chained_navigation` | `270e974975984f662f998c8f6eb0ebebd964de82` | [395f690](https://github.com/mahmoud/boltons/commit/395f690f4a24331c4554e2169ac18a15955a4eab) | 报错截图 | 六种连续导航、单次导航、父目录、原 URL 不变 |
| `boltons_one_to_one_empty_update` | `985df86bbf511b244ce6acf7a9c3b8a0d09d0fb3` | [6cac49c](https://github.com/mahmoud/boltons/commit/6cac49ce4eba506a3582e8089bcc347c6b2fa70f) | 报错截图 | 空字典/列表/迭代器、关键字更新、非空更新、值冲突、逆映射一致性 |

每题导出完整错误版本，包含约 114 个文件、0.9 MB 源码与文档。保留 BSD 许可证。
不向 Agent 提供 Git 对象、修复提交信息、参考补丁、回放脚本或可见内容转写文件。
评分测试在仓库内可见且只读；这是能通过固定回归测试的工程验证，不包含隐藏测试成绩。

## 构建审核

`sca build-public-data` 检查固定父提交，逐题执行如下步骤：

1. `git archive` 导出错误版本，安装只依赖标准库的测试适配器。
2. 在 Bubblewrap 中复现错误；已通过或超时均拒绝成为修复任务。
3. 从固定官方修复提交读取修改后的源码，再在同样的隔离环境中验证测试通过。
4. 恢复原始错误源码，导出上游源码补丁；参考资料保留在输入仓库外。
5. 先固定仓库分组与 `demo` 分配，再渲染浅色/深色变体；规范清单只选浅色。
6. 记录源码、补丁和许可证 SHA-256，以及初始/参考测试日志。

`provenance.json` 为逐题审核记录，`build_report.json` 为汇总。
问题描述保留用户可理解的现象，具体文件、代码或异常由截图、仓库搜索和测试提供。
这些公开小补丁可能已出现在基础模型的预训练数据中，不能用于测量未见任务泛化。

## 输入来源

源码库：[mahmoud/boltons](https://github.com/mahmoud/boltons)。
数据构建配方：[public_data.py](../src/screenshot_agent/public_data.py)。
官方修复只用于构建验证、已知补丁回放和离线诊断；模型修复评测使用 `--model api`。
目前没有从真实用户收集的截图，后续补充时须单独记录拍摄来源和验证协议。
