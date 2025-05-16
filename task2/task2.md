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


# Mask R-CNN

训练模型：python Mask-R-CNN/train_test_mask_rcnn.py --mode train 

测试整个数据集：python Mask-R-CNN/train_test_mask_rcnn.py --mode test --data_dir path/to/VOC2012 --model_path best_model.pth

测试单张图像：python Mask-R-CNN/train_test_mask_rcnn.py --mode test --model_path mask_rcnn_best_model.pth --test_image outer-test/image4.jpg --output_path output/mask-rcnn/result8.jpg

查看tensorboard日志：tensorboard --logdir logs

# Sparse R-CNN

python Sparse-R-CNN/train_test_sparse_rcnn.py --mode train 


python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc0712.py --work-dir work_dirs/sparse_rcnn_voc

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
conda-repo-cli 1.0.114 requires requests>=2.31.0, but you have requests 2.28.2 which is incompatible.
conda-repo-cli 1.0.114 requires urllib3>=2.2.2, but you have urllib3 1.26.20 which is incompatible.
jupyterlab-server 2.27.3 requires requests>=2.31, but you have requests 2.28.2 which is incompatible.    5.12:  安装openmin 报错信息



Package                 Version

---

absl-py                 2.2.2
addict                  2.4.0
Brotli                  1.0.9
cachetools              5.5.2
certifi                 2024.8.30
charset-normalizer      3.3.2
contourpy               1.1.1
cycler                  0.12.1
fonttools               4.57.0
future                  1.0.0
google-auth             2.40.1
google-auth-oauthlib    1.0.0
grpcio                  1.70.0
idna                    3.7
importlib_metadata      8.5.0
importlib_resources     6.4.5
kiwisolver              1.4.7
Markdown                3.7
markdown-it-py          3.0.0
MarkupSafe              2.1.5
matplotlib              3.7.5
mdurl                   0.1.2
mkl-fft                 1.3.1
mkl-random              1.2.2
mkl-service             2.4.0
mmcv                    2.0.0rc4
mmdet                   3.3.0
mmengine                0.10.7
numpy                   1.24.3
oauthlib                3.2.2
opencv-python           4.11.0.86
packaging               25.0
pillow                  10.4.0
pip                     24.2
platformdirs            4.3.6
protobuf                5.29.4
pyasn1                  0.6.1
pyasn1_modules          0.4.2
pycocotools             2.0.7
Pygments                2.19.1
pyparsing               3.1.4
PySocks                 1.7.1
python-dateutil         2.9.0.post0
PyYAML                  6.0.2
requests                2.32.3
requests-oauthlib       2.0.0
rich                    14.0.0
rsa                     4.9.1
scipy                   1.10.1
setuptools              75.1.0
shapely                 2.0.7
six                     1.16.0
tensorboard             2.14.0
tensorboard-data-server 0.7.2
termcolor               2.4.0
terminaltables          3.1.10
tomli                   2.2.1
torch                   1.11.0
torchaudio              0.11.0
torchvision             0.12.0
tqdm                    4.67.1
typing_extensions       4.11.0
urllib3                 2.2.3
Werkzeug                3.0.6
wheel                   0.44.0
yapf                    0.43.0
zipp                    3.20.2
