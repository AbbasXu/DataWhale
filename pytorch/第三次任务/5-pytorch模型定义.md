> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-18]]
> Title : pytorch基础
> Keywords : #pytroch #模型搭建 #训练  
---
# 模型定义的方式
## 回顾
- `Module `类是 `torch.nn` 模块里提供的一个模型构造类 (`nn.Module`)，是所有神经⽹网络模块的基类，我们可以继承它来定义我们想要的模型；
- PyTorch模型定义应包括两个主要部分：各个部分的初始化（`__init__`）；数据流向定义（`forward`）
- 基于nn.Module，我们可以通过Sequential，ModuleList和ModuleDict三种方式定义PyTorch模型。
## Sequential
- 使用场景：当模型的前向计算为简单串联各个层的计算时
- 排列方式：
	- 直接排列

```

```
