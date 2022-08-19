> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-18]]
> Title : pytorch基础
> Keywords : #pytroch #模型搭建 #训练 #损失函数 
---
# 基本配置
## 常用的包
- 表格处理-pandas 
- 图像处理-cv2
- 可视化-pyecharts、matplotlib、seaborn
- 下游分析、指标计算-scikit-learn。
## 超参数设置
- batchsize 
- 初始学习率
- 训练次数
- GPU配置 
## GPU配置 
GPU的设置有两种常见的方式

```
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
```
# 数据读入
数据输入的过程可以定义自己的Dataset类来实现快速读取，，定义的类需要继承PyTorch自身的Dataset类。主要包含三个函数：
- `__init__`: 用于向类中传入外部参数，同时定义样本集
- `__getitem__`: 用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
- `__len`__: 用于返回数据集的样本数
