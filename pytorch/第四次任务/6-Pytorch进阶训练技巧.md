> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-18]]
> Title : pytorch基础
> Keywords : #pytroch #学习率 #损失函数 #模型微调 #数据增强 #调参
---
# 自定义损失函数
## 以函数的方式定义
直接以函数定义的方式定义一个自己的函数。
特点：<font color=red>简单</font>
```
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```
## 以类的方式定义
特点：<font color=red>常用</font>
损失函数类就需要继承自`nn.Module`类

