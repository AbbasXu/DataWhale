> [!abstract] Note Information
> Author : AbbasXu
> Date : [[2022-08-16]]
> Title : pytorch的安装
> Keywords : #pytroch #环境配置

---
# Start
#系统：Mac/Windows/Linux
	由于本人在以往的课设项目、实验室项目中已经多次使用过pytorch，对pytorch的安装方式基本上已经都尝试了一遍，包括在三大系统上都已经做过尝试，因此在本次记录中我会着重于我在pytorch的安装中遇到的问题和需要注意的点，具体过程不再赘述。
## Mac
本人的机器型号为MacBook air M1，因此在本来安装的时候是安装的Miniconda，后续pytorch支持M1版本之后，又需要更换至Arm版的anaconda才能安装。
![](https://obsidian-1305958072.cos.ap-guangzhou.myqcloud.com/obsidian_img/202208161655180.png)

> [!info] 安装代码
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

输入下面命令进行验证是否安装成功。
> [!info] 验证代码
python
import torch
torch.__version__
torch.device("mps")

如下图显示证明已经支持GPU加速
![](https://obsidian-1305958072.cos.ap-guangzhou.myqcloud.com/obsidian_img/202208161137429.png)
## Windows/Linux
针对Windows和Linux(Ubuntu)，仅需要注意在安装GPU版本的pytorch时，注意CUDA、CUDNN与pytorch版本直接的[对应关系](https://pytorch.org/get-started/previous-versions/)
同时注意不同的CUDA版本是可以兼容的，只需要在环境配置中调整成对应的版本就好。（windwos使用环境变量配置）
	