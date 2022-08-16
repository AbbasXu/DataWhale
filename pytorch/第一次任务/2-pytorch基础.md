> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-16]]
> Title : pytorch基础
---
# 张量
## 概念
**几何代数中定义的张量是基于向量和矩阵的推广**， 从某种角度来说，张量就是一个数据容器
| 张量维度 | 代表含义                                     |
| -------- | -------------------------------------------- |
| 0维张量  | 代表的是标量（数字）                         |
| 1维张量  | 代表的是向量                                 |
| 2维张量  | 代表的是矩阵                                 |
| 3维张量  | 时间序列数据 股价 文本数据 单张彩色图片(RGB) |
PyTorch中的张量一共支持9种数据类型，每种数据类型都对应**CPU**和**GPU**的两种子类型，如下表所示
| 数据类型                 | PyTorch类型                  | CPU上的张量            | GPU上的张量                 |
|----------|----------------------------|--------------------|-------------------------|
| 32位浮点数   | torch.float32/torch.float  | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64位浮点数   | torch.float64/torch.double | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16位浮点数   | torch.float16/torch.half   | torch.HalfTensor   | torch.cuda.HalfTensor   |
| 8位无符号整数  | torch.uint8                | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8位带符号整数  | torch.int8                 | torch.CharTensor   | torch.cuda.CharTensor   |
| 16位带符号整数 | torch.int16/torch.short    | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32位带符号整数 | torch.int32/torch.int      | torch.IntTensor    | torch.cuda.IntTensor    |
| 64位带符号整数 | torch.int64/torch.long     | torch.LongTensor   | torch.cuda.LongTensor   |
| 布尔型      | torch.bool                 | torch.BoolTensor   | torch.cuda.BoolTensor   |
