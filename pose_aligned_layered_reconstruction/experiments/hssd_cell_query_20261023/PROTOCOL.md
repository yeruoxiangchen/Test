# HSSD CellQuery16：第一轮限定接口实验，2026-10-23

假设：SS16格点不一定落在表面上，格点中心的窄核查询可能损失有效几何。
本实验只改变几何查询的空间汇聚，检验正确/预测几何能否改善SS与最终mesh。
这是假设检验，不预先宣称修复或论文创新已经成立。

## 固定方法与预算

| arm | 12维条件来源 | 查询 |
|---|---|---|
| appearance_control | 已有冻结外观投影 | 原格点 |
| oracle_point | GT分层几何 | 原格点 |
| oracle_cell | GT分层几何 | 格内27点平均 |
| predicted_point | 冻结LaRI | 原格点 |
| predicted_cell | 冻结LaRI | 格内27点平均 |

cell为16³每个格子中的3×3×3中点积分，三个轴偏移为{-1/48,0,1/48}。
每个子点沿用原12维匹配和tau=.02，取完整特征向量的算术平均；不逐通道取max，
不按GT选点，不将低support强行归一化。旋转作用于点和偏移，固定原grid_transform。
格中心的外观融合权重、视锥有效性、12→128→128残差编码器、LayerNorm及
24个零初始化128→1024无bias投影完全不变。每组3,164,160个训练参数。

每域使用已有no_layer500 raw为冻结基线，包括主干/LoRA、原外观注入、gate、LaRI。
clean16/occluded16同旧manifest、同样本、同GT、同latent，每对象8视图。
训练10组，每组200步；固定seed20261021、batch1、bf16、AdamW lr1e-4、
betas(.9,.95)、wd.01、20步warmup后constant、clip1，仅训练SS残差flow MSE。
对象顺序、diffusion time和noise沿用旧残差的逐步独立生成器，避免缓存影响随机流。
新版本重新训练全部五组，不能混用旧200步权重冒充新协议结果。

## 阶段与停止条件

1. CPU几何诊断：原32条样本，读取旧记录绑定的FP32 LaRI缓存，对照point/cell。
   统计GT64表面所落SS16格与远区格的支持度。该支持度不是生成质量，也不是自动筛选门槛。
2. GPU工程预检：五组各2步，验证零初始化等于原基线、梯度、恢复、父模型冻结；
   一个25步SS轨迹及解码。不启动正式训练或SLat长测试。
3. 用户启动train+SS：训练10组，随后对六种方法（含冻结base）、两域、16对象、
   三个采样seed42/43/44评估SS，共576条。全部保留，包括空/非有限输出。
4. SS结束停在ss_complete_review_required，绝不自动接mesh或扩大样本。
   用户可单独启动mesh：同576条SS支撑，统一冻结SLat30K和decoder，完整采样后
   输出原生坐标CD/F-score、逐对象/种子差异、对象bootstrap与全部图库。

SS/SLat均25步、CFG5、interval[.5,1]、rescale_t3。SS noise种子
seed*1000003+index*1009；SLat沿用原master_noise/sparse_noise按坐标取同一噪声。
mesh写PLY后重新读取评分，原生GT无独立居中/旋转/缩放/ICP；四视角图库相机固定。
主对照是oracle_cell−oracle_point、predicted_cell−predicted_point以及
两者相对appearance_control。冻结base只作参照。报告全部预定对象和三个种子，
不按最好seed、最佳checkpoint或成功子集发布主结果。

若oracle也不能稳定兑现几何价值，优先重新考虑接口假设；若oracle有效而predicted
无效，再修改预测器。平均数、逐对象一致性、种子波动、缺失和额外表面需结合审阅。
本轮不设方便过关的新数值门槛。最多再进行一轮有依据的诊断修订，然后转独立验证
或调整假设；不能无限在16对象上选方案。三采样seed不等于三训练seed。

## 验证集锁定

使用本地另外两个场景102343992、102344022，各16对象，共32独立资产家族。
固定家具类别轮转、seed20261023的SHA顺序、同资产最小instance作为平局规则；
排除训练场景全部背景和目标出现过的资产家族，也排除两验证场景间重复家族。
选择不读取模型成绩。selection.json与原资产/配置/场景/元数据SHA绑定。
这是开发验证集，不是最终测试集；最终测试另行建集，禁止将本名单用于梯度更新。
known pose、visible target mask、canonicalization都是任务条件，不声称无位姿重建。

当前锁定与原生mesh坐标检查不等于已经完成场景RGB、遮挡标签、视图可见率或
mesh→latent验收。selection.json是选择清单，不冒充SceneDataset训练manifest。
正式跨场景实验前要补齐上述数据，并保持这32个对象，不按结果自动替换难例。

## 身份与运行

源码、协议、样本和父权重绑定到run/checkpoint/evaluation；修改源码后拒绝续跑。
原pilot、旧诊断、旧残差文件不修改。新输出仅在outputs/hssd_cell_query_20261023。
单GPU、单子进程、queue.lock互斥，失败停止，长队列由用户启动监视。
