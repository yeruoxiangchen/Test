# 视觉响应 JSON 恢复检查（v5）

离线重新解析 CodeOCR 全量实验中已保存的 5 条不合规视觉输出；没有调用模型、补写代码或替换 v4 成绩。
仅允许 JSON 字符串中的原始控制字符；保留代码转义，不猜测缺失引号或截断内容。无法恢复时保留原始响应并显式标记。

| 样本 | 解析状态 | 原 CER | 重解析 CER | Python 语法通过 |
|---|---|---:|---:|---|
| codeocr_038_ratio1 | relaxed_json | 8.31% | 0.10% | True |
| codeocr_038_ratio2 | relaxed_json | 7.73% | 0.10% | True |
| codeocr_052_ratio1 | relaxed_json | 8.96% | 0.12% | True |
| codeocr_052_ratio2 | unparsed_text | 8.06% | 8.06% | False |
| codeocr_067_ratio2 | unparsed_text | 17.24% | 17.24% | False |

可恢复 3/5 条。这里只证明解析器能挽回部分格式损失，不能证明原本未识别清楚的图片能被读对。
新增集成测试还覆盖歧义/截断输出保留、转义不被重复解码、坏图片失败落盘、请求超时记账，以及首次读取后隐藏截图工具。
这些属于系统健壮性改进，不属于模型训练或 Agent 推理策略创新。
