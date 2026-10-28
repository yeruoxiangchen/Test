# DepthBalance16：一次限定的深度监督假设检验，2026-10-24

固定原HSSD遮挡域16个训练物体、原8视角、相同mesh/latent及标签。
本轮只有uniform与balanced两个新方法，不修改旧实验源文件、权重、协议或数据。
不是泛化实验，也不预设LaRI路线正确；不在得到结果后增加本轮步数、挑选checkpoint或替换案例。

## 唯一主动变化：深度损失的归一化

uniform严格复用旧layer_loss。balanced将有效有监督交点分成互斥三组：
非目标、前面没有有效非目标交点的目标、前面有有效非目标交点的隐藏目标。
每物体使用所有8视角；分别计算每组SmoothL1均值，再对非空组等权平均。
三组齐全时权重各1/3；缺组时只对存在的组重新归一化；无有效点时深度损失为零。
没有把组均值相加以人为增加深度loss总尺度；标签只用于监督分组，不作为预测分支的推理输入。
“exposed_target”包括同目标后侧交点，不冒充严格可见第一表面。

stop(.25)、ownership(.25)、confidence(.1)、cross-view(.1)的公式与权重不变。
confidence仍使用exp(-depth_error/.05)的detach软目标。共享骨干随新的深度梯度变化，其他head输出
可能随训练变化；这是监督修改的间接效果，不宣称只改变了深度数值。
几何5层/32候选、网络容量、图像预处理、候选点、tau=.02、SS point接口均不变。

## 公平训练与恢复

两组LaRI都从相同随机初始化训练1000步，seed20261020；不将新组续训与旧1000步直接比较。
batch=1物体×8视角，AdamW lr1e-4、betas(.9,.95)、wd.01，warmup20后恒定，clip1，bf16。
每epoch使用seed+epoch固定对象排列，记录逐步uid/lr、初始化哈希和实际参数量。
每组最终raw权重固定后，分别训练相同point residual SS adapter200步，seed20261021；
冻结相同no_layer500外观基线，只替换其中被预测分支调用的LaRI权重。
SS同初始化/容量/对象顺序、同stateless diffusion time/noise，lr/warmup/优化器与既有CellQuery保持一致。
SS训练不更新LaRI或外观基线。每20步及正常限步退出保存；恢复只接受相同数据与代码协议。
失败不悄悄删样本；数值失败单独保留；完成记录不触发重新训练。

## 评估与运行边界

FP32测两组全部16训练物体的目标、首层、隐藏目标、背景深度误差、归属、confidence、stop，
保留相机与GT64坐标检查。只报告训练拟合诊断，不将MAE下降作为最终重建通过。
新SS为2组×16物体×3采样seed=96条，种子42/43/44；噪声保留原全局对象索引。
采样25步、CFG5、interval[.5,1]、rescale_t3，与旧实验一致。
复用并核验原CellQuery的base、appearance_control、oracle_point共144条SS作为参照。
三个采样seed不是三个训练seed。训练seed只有一个。

run/start-run只执行训练、LaRI审计、SS及汇总，然后停止在ss_complete_review_required。
用户审阅后才单独start-mesh：新两组各4对象×3seed，共24个mesh。
对象沿用Causal4原全局索引0/2/5/14（i076/i083/i075/i108），不根据本轮结果重新选择。
它们曾按旧SS结果事后选择，只能诊断，不能作为随机总体估计。
复用Causal4相同三参照36个mesh，共60条配对记录、12张GT+五方法共享四视角图库。
冻结同一SLat30K及解码器、按全局坐标匹配噪声。保存PLY重读评分，禁止独立旋转/平移/缩放/ICP。

主要比较balanced−uniform，同时看相对同容量appearance_control；oracle提供正确几何参照，
不是数学上的性能上界。保留对象等权均值、逐对象及逐seed方向、CD/F-score与实际结构。
若深度改善而SS/mesh无稳定收益，不能据此扩量；应重新评估条件注入、SLat和贡献范围。
不依据单个loss或单个seed自动通过/扩大训练；任何完成状态都需人工审阅。
单卡串行；长训练和采样交由用户启动监视。代理只做CPU检查及两组各2步的实际入口预检、
有限前向和单条25步SS，不做正式训练或全量mesh。预检输出独立，不作为正式模型或研究结果。
