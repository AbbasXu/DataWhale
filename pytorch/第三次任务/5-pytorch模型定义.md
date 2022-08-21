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
	net = nn.Sequential(
	        nn.Linear(784, 256),
	        nn.ReLU(),
	        nn.Linear(256, 10), 
	        )
	```
	- 使用OrderedDict
	```
	net2 = nn.Sequential(collections.OrderedDict([
	          ('fc1', nn.Linear(784, 256)),
	          ('relu1', nn.ReLU()),
	          ('fc2', nn.Linear(256, 10))
	          ]))
	```
- 特点：<font color=Red>简单、易读且不需要写forward函数；但同时不够灵活</font>
## ModuleList
- 特点：<font color=Red>可以像list那样进行append和extend操作，但该方法不会定义一个网络，需要使用forward函数对顺序进行确定</font>
## ModuleDict
- 特点：<font color=Red>ModuleDict和ModuleList的作用类似，只是ModuleDict能够更方便地为神经网络的层添加名称。</font>

注：ModuleList和ModuleDict的使用场景为某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“

---
# 利用模型块快速搭建复杂网络
## U-net简介
U-Net是分割 (Segmentation) 模型的杰作，在以医学影像为代表的诸多领域有着广泛的应用。U-Net模型结构如下图所示，通过残差连接结构解决了模型学习中的退化问题，使得神经网络的深度能够不断扩展。
![](https://obsidian-1305958072.cos.ap-guangzhou.myqcloud.com/obsidian_img/202208212125041.png)
组成U-Net的模型块主要有如下几个部分：
- 