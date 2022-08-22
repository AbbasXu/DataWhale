> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-18]]
> Title : pytorch基础
> Keywords : #pytroch #学习率 #损失函数 #模型微调 #数据增强 #调参
---
# 自定义损失函数
## 损失函数与层的共性
本质上来说，损失函数和自定义层有着很多类似的地方，他们都是通过对输入进行函数运算，得到一个输出，这也就是层的功能。只不过层的函数运算比较不一样，可能是线性组合、卷积运算等，但终归也是函数运算，正是基于这样的共性，所以我们可以统一的使用nn.Module类来定义损失函数，而且定义的方式也和前面的层是大同小异的。
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
以DiceLoss为例，其数学公式如下所示： 
$$DSC = \frac{2|X∩Y|}{|X|+|Y|}$$

```
class DiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()
        
    def forward(self,inputs,targets,smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

# 使用方法    
criterion = DiceLoss()
loss = criterion(input,targets)
```

## 通过nn.functional直接定义函数来完成
一般情况下，损失函数是没有参数信息和状态需要维护的，所以更多的时候我们没有必要小题大做，自己去定义一个损失函数的类，我们只需要一个计算的数学函数即可，nn.functional里面定义了一些常见的函数，当然也包括一些常见的损失函数，如下：

```
@weak_script
def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
 
@weak_script
def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
 
@weak_script
def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
 
@weak_script
def margin_ranking_loss(input1, input2, target, margin=0, size_average=None,
                        reduce=None, reduction='mean'):
 
@weak_script
def hinge_embedding_loss(input, target, margin=1.0, size_average=None,
                         reduce=None, reduction='mean'):
 
@weak_script
def multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
 
@weak_script
def soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
 
@weak_script
def multilabel_soft_margin_loss(input, target, weight=None, size_average=None,
                                reduce=None, reduction='mean'):
@weak_script
def cosine_embedding_loss(input1, input2, target, margin=0, size_average=None,
                          reduce=None, reduction='mean'):
@weak_script
def multi_margin_loss(input, target, p=1, margin=1., weight=None, size_average=None):
```
注：
<font color=red>在自定义损失函数时，涉及到数学运算时，我们最好全程使用PyTorch提供的张量计算接口，这样就不需要我们实现自动求导功能并且我们可以直接调用cuda</font>
