**任务2：**
在VOC数据集上训练并测试模型 Mask R-CNN 和 Sparse R-CNN** **

基本要求：
（1） 学习使用现成的目标检测框架——如[mmdetection]([https://github.com/open-mmlab/mmdetection**Links to an external site.**](https://github.com/open-mmlab/mmdetection))——在VOC数据集上训练并测试目标检测模型Mask R-CNN 和Sparse R-CNN；
（2） 挑选4张测试集中的图像，通过可视化**对比**训练好的Mask R-CNN第一阶段产生的proposal box和最终的预测结果，以及Mask R-CNN 和Sparse R-CNN的**实例分割**与**目标检测**可视化结果；
（3） 搜集三张不在VOC数据集内包含有VOC中类别物体的图像，分别可视化并比较两个在VOC数据集上训练好的模型在这三张图片上的目标检测/实例分割结果（展示bounding box、instance mask、类别标签和得分）；

提交要求：
（1） ** 仅提交pdf格式的实验报告** ，报告中除对模型、数据集和实验结果的介绍外，还应包含用Tensorboard可视化的训练过程中在训练集和验证集上的loss曲线和验证集上的mAP曲线；
（2） 报告中应提供详细的实验设置，如训练测试集划分、网络结构、batch size、learning rate、优化器、iteration、epoch、loss function、评价指标等。
（3） 代码提交到自己的public github repo，repo的readme中应清晰指明如何进行训练和测试，训练好的模型权重上传到百度云/google drive等网盘，实验报告内应**包含**实验代码所在的**github repo链接**及模型 **权重的下载地址** 。


### Mask R-CNN

```
python Mask-R-CNN/train_test_mask_rcnn.py --mode train #训练模型
```

```
python Mask-R-CNN/train_test_mask_rcnn.py --mode test --data_dir path/to/VOC2012 --model_path best_model.pth #测试整个数据集
```

```
python Mask-R-CNN/train_test_mask_rcnn.py --mode test --model_path mask_rcnn_best_model.pth --test_image outer-test/image4.jpg --output_path output/mask-rcnn/result8.jpg #测试单张图像，输出最终预测结果
```

```
python Mask-R-CNN/train_test_mask_rcnn.py --mode test --model_path mask_rcnn_best_model.pth --test_image outer-test/image4.jpg --output_path output/mask-rcnn/result8.jpg  --output_proposals  #测试单张图像，输出proposals
```

```
tensorboard --logdir logs #查看tensorboard日志
```

### Sparse R-CNN

mmdetection-main框架下载链接：https://github.com/open-mmlab/mmdetection

```
python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py --word_dirs word_dirs/sparse_rcnn_r50_fpn_1x_coco #生成配置文件
```

```
python tools/train.py work_dirs/sparse_rcnn_r50_fpn_1x_coco/sparse-rcnn_r50_fpn_1x_coco.py #训练模型
```

```
python singletest.py #测试单张图片，指定图片文件路径
```

```
 python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/sparse_rcnn_r50_fpn_1x_coco/20250516_081836/vis_data/20250516_081836.json     --keys loss --out results.png --legend loss  #绘制loss、mAP曲线（通过指定不同keys）
```

```
python singletest.py #测试单张图片，指定图片文件路径
```

