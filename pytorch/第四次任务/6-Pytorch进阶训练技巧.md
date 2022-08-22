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
# 动态调整学习率
## 使用官方scheduler
PyTorch已经在`torch.optim.lr_scheduler`为我们封装好了一些动态调整学习率的方法供我们使用
-   [`lr_scheduler.LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR)
-   [`lr_scheduler.MultiplicativeLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR)
-   [`lr_scheduler.StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR)
-   [`lr_scheduler.MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)
-   [`lr_scheduler.ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR) 
-   [`lr_scheduler.CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)
-   [`lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
-   [`lr_scheduler.CyclicLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
-   [`lr_scheduler.OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)
-   [`lr_scheduler.CosineAnnealingWarmRestarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
注：
我们在使用官方给出的`torch.optim.lr_scheduler`时，需要将`scheduler.step()`放在`optimizer.step()`后面进行使用。
## 自定义scheduler
方法是自定义函数`adjust_learning_rate`来改变`param_group`中`lr`的值。以需要学习率每30轮下降为原来的1/10为例。

```
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```
# 模型微调-torchvision
## 模型微调的流程
1.  在源数据集(如ImageNet数据集)上预训练一个神经网络模型，即源模型。
2.  创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
3.  为目标模型添加一个输出⼤小为⽬标数据集类别个数的输出层，并随机初始化该层的模型参数。  
4.  在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。
![](https://obsidian-1305958072.cos.ap-guangzhou.myqcloud.com/obsidian_img/202208221449538.png)
