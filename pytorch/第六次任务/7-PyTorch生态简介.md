> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-28]]
> Title : pytorch基础
> Keywords : #pytroch #torchvision #PytorchVideo #torchtext  
---
# torchvision
## torchvision.datasets
`torchvision.datasets`主要包含了一些我们在计算机视觉中常见的数据集，在==0.10.0版本==的`torchvision`下，有以下的数据集：
| Names     | Names        | Names         | Names      |
| --------- | ------------ | ------------- | ---------- |
| Caltech   | CelebA       | CIFAR         | Cityscapes |
| EMNIST    | FakeData     | Fashion-MNIST | Flickr     |
| ImageNet  | Kinetics-400 | KITTI         | KMNIST     |
| PhotoTour | Places365    | QMNIST        | SBD        |
| SEMEION   | STL10        | SVHN          | UCF101     |
| VOC       | WIDERFace    |               |            |
## torchvision.transforms
一些对图像的数据预处理或者是增强的方法
## torchvision.models
PyTorch官方也提供了一些预训练好的模型供我们使用
-   **Classification**
| Names     | Names        | Names         | Names      |
| --------- | ------------ | ------------- | ---------- |
| AlexNet     | VGG          | ResNet    | SqueezeNet    |
| DenseNet    | Inception v3 | GoogLeNet | ShuffleNet v2 |
| MobileNetV2 | MobileNetV3  | ResNext   | Wide ResNet   |
| MNASNet     | EfficientNet | RegNet    | 持续更新          |
-   **Semantic Segmentation**
| FCN ResNet50              | FCN ResNet101               | DeepLabV3 ResNet50 | DeepLabV3 ResNet101 |
|---------------------------|-----------------------------|--------------------|---------------------|
| LR-ASPP MobileNetV3-Large | DeepLabV3 MobileNetV3-Large | 未完待续               |
-   **Object Detection，instance Segmentation and Keypoint Detection**
| Faster R-CNN | Mask R-CNN | RetinaNet | SSDlite |
|--------------|------------|-----------|---------|
| SSD          | 未完待续       |
- **Video classification**
| ResNet 3D 18 | ResNet MC 18 | ResNet (2+1) D |
|--------------|--------------|----------------|
## torchvision.ops
包括但不仅限于NMS，RoIAlign（MASK R-CNN中应用的一种方法），RoIPool（Fast R-CNN中用到的一种方法）
## torchvision.utils

