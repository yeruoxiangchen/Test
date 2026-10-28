# 内置示例

`code/task_light.json`：代码截图，加法函数存在运算符错误。
`error/task_light.json`：实际测试失败日志截图，均值函数分母错误。

两个示例均为自编合成数据，`repo/` 是错误版本。参考补丁、准确转写和回放脚本存放在仓库之外。
API 模式不读取回放脚本或参考补丁。`replay` 模式用于无模型的已知编辑工程检查。

```bash
.venv/bin/sca run --task examples/code/task_light.json --reader ocr --model replay
.venv/bin/sca run --task examples/error/task_light.json --config configs/api.json --reader vlm --model api
```
