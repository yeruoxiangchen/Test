# LaRI Causal4：固定模型的分项诊断，2026-10-23

目的：定位LaRI预测与GT几何的差异，并核对SS变化对应的实际mesh。
不训练新权重，不改变旧数据、latent、损失、采样器或原实验协议，不自动扩量。
本轮是最后一轮限定诊断，结果用于决定一次有依据的修改或调整假设。

## 已有LaRI的拟合情况

读取clean/occluded最终LaRI1000原始权重绑定的32份FP32预测缓存。
逐张量确认它们与CellQuery残差所用no_layer500内的冻结LaRI完全一致。
按全部场景层、目标层、首层目标、前方有非目标的隐藏目标层、非目标层分别统计
深度误差、阈值覆盖、存在性、归属和置信度；stop另报告ray级混淆矩阵。
confidence与训练软目标exp(-error/.05)比较，不冒充误差<.02的概率校准。
检查监督覆盖、场景层目标比例、GT是否超出输出区间、相机射线/深度往返、
GT目标交点与独立GT64体素中心的距离。坐标检查不拟合任何新变换。
训练历史只描述已有1000步趋势；本次不宣称训练已收敛或证明架构容量上限。

## 固定案例

只对occluded域原始全局索引0/2/5/14（i076/i083/i075/i108）做新生成。
它们分别为首对象、oracle cell最大退化、oracle point及预测cell较大收益、预测cell最大退化。
选择依据是已经完成的SS结果，属于事后诊断，不是随机测试集。所有新干预执行前冻结名单，
不按新结果替换案例。所有案例保留seed42/43/44，全局对象索引不重新编号。

## SS分项替换：192条

冻结两个CellQuery200残差checkpoint：predicted_point与oracle_point。
每个模型在相同输入、相同随机噪声下比较8种条件：predicted、gt_depth、
gt_existence、gt_ownership、confidence_one、gt_depth_confidence_one、
gt_depth_existence_ownership、oracle。2模型×4对象×3seed×8条件=192条SS。
双模型用于观察对条件分布的依赖；不同模型间的差异不能归为某单个head的因果效应。

干预在预测/GT按生产规则插值后进行，随后调用完全相同的12维feature构造。
GT depth仅在采样GT存在概率>.5的层替换，其他层保留预测值，避免无效占位深度污染。
只换existence不会重算另外三个已经插值的量，故是查询层面的分项干预，不是可部署head替换。
tau=.02、相对容差0、SS16原点查询、外观权重、残差编码/注入、主干、CFG全部不变。
置信度设为1是诊断，不是建议部署；GT干预会改变条件分布，不能直接作为模型改进成绩。

SS25步、CFG5、interval[.5,1]、rescale_t3、bf16；初始noise种子为
seed*1000003+原全局index*1009，与CellQuery一致。原生条件12维特征需与旧实现一致，
原生SS输出必须与已有CellQuery坐标完全相同，否则停止审阅复现差异。
空输出/非有限输出保留为失败，不允许只汇总成功案例。SS汇总在同一个模型内相对predicted条件比较。

## 固定mesh案例：72条

直接使用现有CellQuery六种方法（base、appearance_control、oracle_point/cell、
predicted_point/cell）的SS支撑，4对象×3seed×6方法=72条mesh。此队列不使用干预后的SS，
避免同时改变几何条件和mesh输入来源。SLat30K及decoder完全冻结，复用原25步与按全局坐标匹配的噪声。
单进程串行、复用已验证的节省显存decoder，SS与mesh阶段在不同子进程执行。

mesh采用to_trimesh(transform_pose=False)，保存PLY重读后对原始canonical GT评分；
禁止独立居中、缩放、旋转、ICP。记录双方原生bbox，CD/F-score及结构指标。
每案例每seed生成GT＋六方法并排、共享四个固定相机的图库，共12张。
小集汇总仅供诊断，不作为全体16对象或跨场景主成绩。全套方法仍须在独立场景完成mesh评估。

## 判读与运行边界

首先看原生回放是否一致，再看单项替换在两个固定模型下是否有可重复方向、
改善哪些对象、是否同时增加额外表面；均值和单个seed不能单独宣布成功。
若定位到明确预测误差，可提出一次有依据的预测器/监督修改；如未定位，不盲目增加训练步数、
候选点数或继续调融合。当前oracle优势不能直接证明多层表示必要，也不能仅归因于depth。

CPU审计与短GPU预检由代理完成；长SS与mesh队列由用户启动监视。
run仅执行上述192条SS和72条mesh，完成后停在diagnostics_complete_review_required。
也可分别运行ss/mesh；任何失败停止，不接续旧实验、不自动启动训练。
