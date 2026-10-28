# Operator audit v1, 2026-10-25

固定前一轮柜架index=2/seed42。使用同一已保存的第一层投影与时间嵌入，
按原FrozenSLat30K.forward转fp16作为input_blocks.0输入。
按原配置重建SparseResBlock3d(128,1024,128)，严格加载同一已验证base safetensors中的该块全部权重；
该块没有LoRA或30K新增参数，保持原参数dtype、bf16 autocast及Native稀疏卷积路径。

三个独立进程各两次，仅记录原forward，不重写数学方程。
逐子模块记录输入和输出，只有同输入、不同输出才作为子模块归因证据。
Native算子额外保存有效邻接对（剔除未初始化的无效尾部）、实际forward GEMM缓存描述符、split-K与累加dtype。
未预设卷积或算法缓存一定是根因。源文件及旧结果保持不变。

两个新进程分别用同一历史latent解码一次；只加载相同decoder，不加载无关Flow，
沿用原显存补丁、无外层autocast、相同GT和native canonical坐标。比较21个原有记录点及最终mesh。
每GPU子阶段600秒上限，失败停止并保留attempt；本轮不训练、不扩数据。
允许结果不一致；不能将一个对象的变化范围当成整个数据集的噪声上界。

如定位到明确的数值路径，最小运行修正及后续公平mesh复核使用新的协议文件、独立输出，
不更改本轮冻结来源或覆盖原成绩。较长完整mesh复核交给用户启动监视。
