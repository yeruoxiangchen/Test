# CodeOCR 全量读取诊断

分析基于已完成的 400 条原始读取，不重试失败样本，不修改预测或主报告。
表内两档输入合并平均；同一代码的两张变体属于同一个源任务。请求异常按 CER=1 计入。

| 分组 | 读取方式 | 源代码数 | 读取数 | CER | 请求异常 | JSON 异常 | 达到输出上限线索 | 语法可解析 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 不含中文 | ocr | 70 | 140 | 6.37% | 0 | 0 | 0 | 0 |
| 不含中文 | vlm | 70 | 140 | 1.23% | 0 | 3 | 0 | 101 |
| 含中文 | ocr | 30 | 60 | 7.96% | 0 | 0 | 0 | 0 |
| 含中文 | vlm | 30 | 60 | 1.61% | 0 | 2 | 0 | 41 |
| 50–60 行 | ocr | 23 | 46 | 5.90% | 0 | 0 | 0 | 0 |
| 50–60 行 | vlm | 23 | 46 | 0.94% | 0 | 0 | 0 | 34 |
| 61–80 行 | ocr | 44 | 88 | 7.60% | 0 | 0 | 0 | 0 |
| 61–80 行 | vlm | 44 | 88 | 1.62% | 0 | 5 | 0 | 66 |
| 81–115 行 | ocr | 33 | 66 | 6.49% | 0 | 0 | 0 | 0 |
| 81–115 行 | vlm | 33 | 66 | 1.26% | 0 | 0 | 0 | 42 |
| 全部 | ocr | 100 | 200 | 6.84% | 0 | 0 | 0 | 0 |
| 全部 | vlm | 100 | 200 | 1.34% | 0 | 5 | 0 | 142 |

达到 2048 输出 token 仅表示可能截断或重复生成，不直接认定根因。当前本地服务的 finish_reason 不能准确区分长度截断。
JSON 异常输出保留原文并计入字符误差；语法可解析不代表转写正确。按语言/长度分组是描述性分析，不能单独证明这些特征造成错误。

| 档位 | 配对源任务 | VLM CER 更低 | OCR CER 更低 | 相同 |
|---|---:|---:|---:|---:|
| 1 | 100 | 92 | 8 | 0 |
| 2 | 100 | 93 | 7 | 0 |

## ocr 字符误差最高的十次读取

- `Wegent_backend_app_models_namespace.py` / 2×：CER=17.01%，JSON 合规=True，输出上限线索=False。
- `JanusVLN_src_qwen_vl_model_vggt_layers_patch_embed.py` / 2×：CER=16.72%，JSON 合规=True，输出上限线索=False。
- `scope_src_scope_server_download_progress_manager.py` / 2×：CER=16.24%，JSON 合规=True，输出上限线索=False。
- `haproxy-openmanager_backend_models_waf.py` / 2×：CER=15.95%，JSON 合规=True，输出上限线索=False。
- `DA-2_src_da2_model_spherevit.py` / 2×：CER=15.13%，JSON 合规=True，输出上限线索=False。
- `reverse-api-engineer_src_reverse_api_action_recorder.py` / 1×：CER=14.98%，JSON 合规=True，输出上限线索=False。
- `K2-Think-SFT_src_llamafactory_webui_manager.py` / 1×：CER=13.79%，JSON 合规=True，输出上限线索=False。
- `deeppoint-ai_lib_crawlers_douyin_new_cache_abs_cache.py` / 2×：CER=13.61%，JSON 合规=True，输出上限线索=False。
- `NOVIX_backend_app_llm_gateway_providers_anthropic_provider.py` / 2×：CER=13.60%，JSON 合规=True，输出上限线索=False。
- `eastmoney_src_analysis_post_market.py` / 2×：CER=13.52%，JSON 合规=True，输出上限线索=False。

## vlm 字符误差最高的十次读取

- `me-cli_app_menus_util.py` / 2×：CER=17.24%，JSON 合规=False，输出上限线索=False。
- `Aether_src_middleware_rate_limit_config.py` / 2×：CER=11.39%，JSON 合规=True，输出上限线索=False。
- `home-assistant-vibecode-agent_app_utils_yaml_editor.py` / 2×：CER=9.77%，JSON 合规=True，输出上限线索=False。
- `deeppoint-ai_lib_crawlers_douyin_new_cache_abs_cache.py` / 1×：CER=8.96%，JSON 合规=False，输出上限线索=False。
- `agentscope-runtime_src_agentscope_runtime_sandbox_enums.py` / 1×：CER=8.31%，JSON 合规=False，输出上限线索=False。
- `deeppoint-ai_lib_crawlers_douyin_new_cache_abs_cache.py` / 2×：CER=8.06%，JSON 合规=False，输出上限线索=False。
- `agentscope-runtime_src_agentscope_runtime_sandbox_enums.py` / 2×：CER=7.73%，JSON 合规=False，输出上限线索=False。
- `batch-data-pipeline_src_config.py` / 2×：CER=7.59%，JSON 合规=True，输出上限线索=False。
- `spec-kitty_src_specify_cli_dashboard_handlers_base.py` / 2×：CER=7.57%，JSON 合规=True，输出上限线索=False。
- `batch-data-pipeline_src_config.py` / 1×：CER=7.34%，JSON 合规=True，输出上限线索=False。

逐条预测和模型响应见主实验归档；这些诊断不构成修复成功率或真实用户截图效果。
本轮记录视觉请求 200 次，输入 token 476110，输出 token 131291。
本地 API 标价为零，硬件与运行时间成本另计。
