> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-16]]
> Title : pytorch基础
> Keywords : #pytroch #张量
---
# 张量
## 简介
**几何代数中定义的张量是基于向量和矩阵的推广**， 从某种角度来说，张量就是一个数据容器
| 张量维度 | 代表含义                                     |
| -------- | -------------------------------------------- |
| 0维张量  | 代表的是标量（数字）                         |
| 1维张量  | 代表的是向量                                 |
| 2维张量  | 代表的是矩阵                                 |
| 3维张量  | 时间序列数据 股价 文本数据 单张彩色图片(RGB) |
除此之外，在我们机器学习的工作中，我们通常需要处理的不止是一张图或者是一段话，而是一组（集合），如下所示：
一张图的表达式为`(width, height, channel) = 3D`，而一组图的表达式为`(batch_size, width, height, channel) = 4D`

张量和Numpy的多维数组比较类似，但是`torch.tensor`提供了GPU计算和自动求梯度等更多功能，这些使 Tensor 这一数据类型更加适合深度学习。

PyTorch中的张量一共支持9种数据类型，每种数据类型都对应**CPU**和**GPU**的两种子类型，如下表所示
| 数据类型       | PyTorch类型                | CPU上的张量        | GPU上的张量             |
| -------------- | -------------------------- | ------------------ | ----------------------- |
| 32位浮点数     | torch.float32/torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64位浮点数     | torch.float64/torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16位浮点数     | torch.float16/torch.half   | N/A                | torch.cuda.HalfTensor   |
| 8位无符号整数  | torch.uint8                | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8位带符号整数  | torch.int8                 | torch.CharTensor   | torch.cuda.CharTensor   |
| 16位带符号整数 | torch.int16/torch.short    | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32位带符号整数 | torch.int32/torch.int      | torch.IntTensor    | torch.cuda.IntTensor    |
| 64位带符号整数 | torch.int64/torch.long     | torch.LongTensor   | torch.cuda.LongTensor   |
| 布尔型         | torch.bool                 | torch.BoolTensor   | torch.cuda.BoolTensor   |
## 创建tensor
[Pytorch 用户手册-tensor](https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/)
常用的`tensor`构建方法
|        函数         |                        功能                        |
|:-------------------:|:--------------------------------------------------:|
|    Tensor(sizes)    |                    基础构造函数                    |
|    tensor(data)     |                   类似于np.array                   |
|     ones(sizes)     |                        全1                         |
|    zeros(sizes)     |                        全0                         |
|     eye(sizes)      |                  对角为1，其余为0                  |
|  arange(s,e,step)   |                 从s到e，步长为step                 |
| linspace(s,e,steps) |               从s到e，均匀分成step份               |
|  rand/randn(sizes)  | rand是\[0,1)均匀分布；randn是服从N(0，1)的正态分布 |
|  normal(mean,std)   |         正态分布(均值为mean，标准差是std)          |
|     randperm(m)     |                      随机排列                      |
## 张量操作
