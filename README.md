# CVMidterm

## task1

caltech 数据集下载链接：https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1

模型训练：在文件目录下运行：

```
python task1-random.py #使用 Caltech-101 数据集从随机初始化的网络参数开始训练
```

```
python task1-pretrained.py #使用 Caltech-101 数据集，其余层使用在ImageNet上预训练得到的网络参数进行初始化，开始训练
```

## task2

VOC数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
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
python tools/dataset_converters/pascal_voc.py \
    /path/to/VOCdevkit/VOC2012 \
    --out-dir /path/to/VOCdevkit/coco \
    --nproc 8 \
    --split 2012  #将VOC2012数据集转换为Coco格式，同时注意修改生成的配置文件中的数据集目录名和数据集类别数量及具体类名
```

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
