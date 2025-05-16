**任务1：**
微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类

基本要求：

(1) 训练集测试集按照 [Caltech-101]( [https://data.caltech.edu/records/mzrjq-6wc02**Links to an external site.**](https://data.caltech.edu/records/mzrjq-6wc02)) 标准；
(2) 修改现有的 CNN 架构（如AlexNet，ResNet-18）用于 Caltech-101 识别，通过将其输出层大小设置为 **101 **以适应数据集中的类别数量，其余层使用在ImageNet上预训练得到的网络参数进行初始化；
(3) 在 Caltech-101 数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调；
(4) 观察不同的超参数，如训练步数、学习率，及其不同组合带来的影响，并尽可能提升模型性能；
(5) 与仅使用 Caltech-101 数据集从随机初始化的网络参数开始训练得到的结果  **进行对比** ，观察预训练带来的提升。

提交要求：
（1） ** 仅提交pdf格式的实验报告** ，报告中除对模型、数据集和实验结果的基本介绍外，还应包含用Tensorboard可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的accuracy变化；
（2） 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应**包含**实验代码所在的**github repo链接**及模型 **权重的下载地址** 。

```
python task1-random.py #使用 Caltech-101 数据集从随机初始化的网络参数开始训练
```

```
python task1-pretrained.py #使用 Caltech-101 数据集，其余层使用在ImageNet上预训练得到的网络参数进行初始化，开始训练
```

