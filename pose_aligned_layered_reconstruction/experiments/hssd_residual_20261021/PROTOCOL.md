# HSSD16 同外观基线的 SS 残差对照，2026-10-21

这是在原训练对象上的诊断实验，尚不是论文 heldout 实验。原 HSSD/正式2K代码、
数据、checkpoint、失败门与诊断记录保持原状；新实现和输出独立版本化。

## 固定的模型与对照

每域都加载其已有的 no_layer500 raw 模型作为共同外观基线，冻结全部权重，
包括 LaRI、两条原外观注入、view gate、主干和 LoRA。原 clean/occluded16、
原8视图、原 SS latent不变。本轮只修改 SS，SLat仍是原30K冻结模型。

三组新增完全相同的 residual：12→128→128 MLP、LayerNorm、24个128→1024
无bias投影。投影全零初始化，初始预测严格等于已有 no_layer500；共同外观
分支始终保留，不被几何替代。没有新增几何 view gate。

| 组 | 新分支输入 | 训练/评估是否使用GT几何 |
|---|---|---|
| appearance_control | 共同冻结外观projector的12维输出 | 否 |
| oracle | GT射线层的12维几何匹配摘要 | 是，仅诊断上界 |
| predicted | 冻结LaRI预测的12维几何匹配摘要 | 否 |

所有组均用原外观视图权重与视锥有效性融合。几何组的support、归属、置信度
保留在输入特征中，不再额外乘一次target_support缩小融合幅度；不将微小
target_support归一化成强证据。融合后LayerNorm，最终幅度由零初始化投影学习。
视锥全无效时输出严格为零；无投影bias，不靠常量偏置形成残差。
匹配尺度仍为.02，相对容差0，候选32和原LaRI不变，避免同时改预测器和核宽。
这是待验证的新接口，不能因信号更强就认定重建更好；需检查远区误补全和mesh。

这个实验同时建立嵌套外观基线和新的融合接口，不能把改善全部归因于其中某一个
改动；后续论文需要逐项消融。appearance_control用相同参数量、初始化与新增
训练预算，专门控制“多训练一个残差”的收益。oracle不计入正常推理成绩。

## 训练与恢复

clean/occluded × 三组，共六个单卡任务，每组200步、batch1、8视图、AdamW
betas(.9,.95)、wd.01、lr1e-4、20步warmup后constant、bf16、p_uncond=0。
仅flow velocity MSE，不追加LaRI监督loss或SLat训练。全组残差初始化种子20261021。
对象顺序由epoch种子决定；每步t和噪声由20261021+step*1009独立生成，缓存和
重启不改变序列。checkpoint只保存新残差、optimizer和原数据/基线/源码身份。
每20步和最终保存last.pt。恢复严格核验合同并从checkpoint步数重放；超出
checkpoint的日志先留存为interrupted记录，不能把重复行当额外训练步。

训练只允许预检通过的代码。预检验证零初始化与已有外观输出逐位相等、三组
初始化/参数量相同、两步后encoder和投影收到梯度、共同基线无修改，以及
25步SS采样、CFG负分支隔离、SS解码。预检不执行完整SLat/mesh评测。

## 完整 mesh 评估

训练完成后停止，单独启动评估。每域16对象、seed42/43，每组32个新mesh，
六组共192个新mesh。对照父模型是已有no_layer500，不是最初SS30K。
原no_layer500的64个基线mesh仅在manifest、checkpoint、采样、种子、原源码
与逐文件hash一致时复用；不会错用原full或SS30K的mesh。

SS/SLat均25步，cfg5、interval[.5,1]、rescale_t3；沿用原master noise和坐标。
GT来自样本绑定的HSSD原mesh；原坐标直接比较，不做ICP、旋转/平移/尺度拟合，
不各自归一化。检查复用GT与绑定源mesh的PLY往返一致。全部对象固定相同相机
渲染GT/基线/新模型图库，计算CD、F-score与结构指标。

空SS或无效mesh记录为失败，不能删对象后汇总成功子集。运行环境/OOM异常直接
报错，保留已完成记录供同身份恢复。训练完成、采样完成与拟合通过分开记录。
原工程拟合判定仍为mean CD改善>0且至少8/16对象改善，非统计显著性证明。
另外必须比较predicted与参数/预算匹配的appearance_control，报告oracle差距、
逐对象/种子结果和失败率；不自动扩量。遮挡SLat旧失败仍是独立未解决问题。
