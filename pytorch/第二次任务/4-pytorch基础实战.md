> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-19]]
> Title : pytorch基础实战
> Keywords : #pytroch #分类 #实战 
---
# 基础实战——FashionMNIST时装分类
## 数据集介绍
**FashionMNIST数据集**
FashionMNIST数据集中包含已经预先划分好的训练集和测试集，其中训练集共60,000张图像，测试集共10,000张图像。每张图像均为单通道黑白图像，大小为28\*28pixel，分属10个类别。
![](https://obsidian-1305958072.cos.ap-guangzhou.myqcloud.com/obsidian_img/202208191503909.png)

结果如下所示：
```
Epoch: 1 	Training Loss: 0.268449
Epoch: 1 	Validation Loss: 0.057068, Accuracy: 0.981800
Epoch: 2 	Training Loss: 0.073943
Epoch: 2 	Validation Loss: 0.035796, Accuracy: 0.987800
Epoch: 3 	Training Loss: 0.052537
Epoch: 3 	Validation Loss: 0.030474, Accuracy: 0.989200
Epoch: 4 	Training Loss: 0.042508
Epoch: 4 	Validation Loss: 0.026520, Accuracy: 0.991500
Epoch: 5 	Training Loss: 0.035499
Epoch: 5 	Validation Loss: 0.026349, Accuracy: 0.991200
Epoch: 6 	Training Loss: 0.030913
Epoch: 6 	Validation Loss: 0.022135, Accuracy: 0.992000
Epoch: 7 	Training Loss: 0.027241
Epoch: 7 	Validation Loss: 0.021995, Accuracy: 0.992900
Epoch: 8 	Training Loss: 0.022869
Epoch: 8 	Validation Loss: 0.021141, Accuracy: 0.992500
Epoch: 9 	Training Loss: 0.021894
Epoch: 9 	Validation Loss: 0.019971, Accuracy: 0.993700
Epoch: 10 	Training Loss: 0.020737
Epoch: 10 	Validation Loss: 0.019363, Accuracy: 0.992900
Epoch: 11 	Training Loss: 0.017129
Epoch: 11 	Validation Loss: 0.020346, Accuracy: 0.993600
Epoch: 12 	Training Loss: 0.016904
Epoch: 12 	Validation Loss: 0.020827, Accuracy: 0.992900
Epoch: 13 	Training Loss: 0.015159
Epoch: 13 	Validation Loss: 0.019825, Accuracy: 0.994200
Epoch: 14 	Training Loss: 0.013941
Epoch: 14 	Validation Loss: 0.021533, Accuracy: 0.993400
Epoch: 15 	Training Loss: 0.013948
Epoch: 15 	Validation Loss: 0.022042, Accuracy: 0.993200
Epoch: 16 	Training Loss: 0.012846
Epoch: 16 	Validation Loss: 0.019770, Accuracy: 0.993400
Epoch: 17 	Training Loss: 0.011409
Epoch: 17 	Validation Loss: 0.018787, Accuracy: 0.993200
Epoch: 18 	Training Loss: 0.011451
Epoch: 18 	Validation Loss: 0.020755, Accuracy: 0.993200
Epoch: 19 	Training Loss: 0.010547
Epoch: 19 	Validation Loss: 0.018624, Accuracy: 0.994700
Epoch: 20 	Training Loss: 0.009446
Epoch: 20 	Validation Loss: 0.026274, Accuracy: 0.991900
```

由训练过程可以看到