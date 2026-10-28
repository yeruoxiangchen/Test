# CodeOCR 全量重建数据运行

本轮已完成全部 400 次读取，耗时 69.62 分钟。请求异常为零；5 次视觉输出 JSON 格式异常保留原文并计入评分。
两个模型进程均正常退出，自动报告与诊断完成，1019 个归档文件指纹核验通过。
结果见 [全量报告](experiment_report_codeocr_full_v4.md) 与 [诊断分析](codeocr_full_v4_analysis.md)。
`watch` 不会随被监视任务结束自动退出，可以按 Ctrl+C 关闭监视。

本轮固定使用 CodeOCR 提交 `411fc815f97ed13ecafa246b6042bb17dbc40748` 发布的全部 100 份重建代码，来自 100 个仓库。
每份代码保留全文，生成 1×/2×两档图像，共 200 张；RapidOCR CPU 与 Qwen3-VL-8B 各读 200 次，合计 400 次。
没有代码长度、语法或模型结果过滤。参考代码 100/100 可以通过 `ast.parse`，它们不会被执行。
全部截图为合成输入。本轮不运行 CodeOCR 的问答、补全、克隆和摘要任务，也未包含 1.5×/4×/6×/8×档位。

数据位于 `datasets/generated/codeocr_full_v4`，结果位于 `outputs/codeocr_full_v4`。
两个分片按排序后的源代码 ID 轮流分配，各 50 个源任务、100 张图、200 次读取；同一代码的变体保持在同一分片。
使用空闲 GPU 6、7，未修改或停止其他项目的任务。启动前 67 项完整检查通过，耗时 17.56 秒。

保留与小规模试验相同的 2048 输出 token 上限；服务上下文扩大到 16384 以接收较宽图像。
使用固定 CJK 字体；三份代码缺失的 ✅/❌/📊 由固定 Emoji 字体补齐单字形，未替换参考字符。
字体、源数据指纹及设置在 [CODEOCR_FULL_UPSTREAM.json](../CODEOCR_FULL_UPSTREAM.json)。

查看实时状态：

```bash
cd /home/zjr/Tracker/internship_project_lab/screenshot_code_agent
watch -n 10 '.venv/bin/python scripts/run_codeocr_full.py --output outputs/codeocr_full_v4 --status'
```

查看逐条进度：

```bash
tail -f outputs/codeocr_full_v4.log outputs/codeocr_full_v4/shard_0.log outputs/codeocr_full_v4/shard_1.log
```

停止本轮模型、评测与分析等待进程：

```bash
systemctl --user stop sca-codeocr-full-v4.service sca-codeocr-full-v4-analysis.service
```

两个分片正常结束后会自动汇总、核对 400 条唯一记录、检查运行期源码与输入指纹，并归档预测、事件、数据包和源码快照。
报告自动生成于 `docs/experiment_report_codeocr_full_v4.md`，诊断分析位于 `docs/codeocr_full_v4_analysis.md`。
异常请求、JSON 格式异常和达到输出上限的线索分别记录；出现失败不会换题或重试以获取更好的成绩。
源码或输入在运行中变化时，归档器会拒绝完整实验声明。请不要原地修改活动运行的脚本和源码。
模型在退出清理时采集显存并停止；整个服务的时间上限为四小时。

新机器准备数据并重跑（先完成主项目 `scripts/setup.sh`；本地模型环境见 README）：

```bash
bash scripts/setup_codeocr_full.sh
.venv/bin/python -u scripts/run_codeocr_full.py --output outputs/codeocr_full_new --gpus 6,7
```

运行目录必须是新的；本轮固定归档目录已经存在后，应为下一次实验使用新的归档版本，避免覆盖结果。
