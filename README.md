# CVMidterm

## task1

模型训练：在文件目录下运行：

```
python task1-random.py #使用 Caltech-101 数据集从随机初始化的网络参数开始训练
```

```
python task1-pretrained.py #使用 Caltech-101 数据集，其余层使用在ImageNet上预训练得到的网络参数进行初始化，开始训练
```

## task2

### Mask R-CNN


```
python Mask-R-CNN/train_test_mask_rcnn.py --mode train #训练模型
```

```
python Mask-R-CNN/train_test_mask_rcnn.py --mode test --data_dir path/to/VOC2012 --model_path best_model.pth #测试整个数据集
```

```
python Mask-R-CNN/train_test_mask_rcnn.py --mode test --model_path mask_rcnn_best_model.pth --test_image outer-test/image4.jpg --output_path output/mask-rcnn/result8.jpg #测试单张图像
```

```
tensorboard --logdir logs #查看tensorboard日志
```




