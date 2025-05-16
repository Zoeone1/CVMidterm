import cv2
import os
import mmcv
import torch
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample

# 配置和权重路径
config_file = 'work_dirs/sparse_rcnn_r50_fpn_1x_coco/sparse-rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/sparse_rcnn_r50_fpn_1x_coco/epoch_12.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
classes = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)

# 创建结果保存目录
result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)

# 输入图片目录
input_dir = 'outer-test'

# 检查目录是否存在
if not os.path.isdir(input_dir):
    print(f"错误：目录 '{input_dir}' 不存在！")
    exit(1)

# 获取目录下的所有图片文件
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# 处理所有图片
for img_file in tqdm(image_files, desc='处理图片'):
    img_path = os.path.join(input_dir, img_file)
    
    # 读取图片
    img = mmcv.imread(img_path)
    
    # 推理
    result = inference_detector(model, img)
    
    # 创建数据样本对象
    data_sample = DetDataSample()
    data_sample.pred_instances = result.pred_instances
    data_sample.set_metainfo({'img_shape': img.shape[:2]})
    
    # 初始化可视化器（不使用 dataset_meta 参数）
    visualizer = DetLocalVisualizer(
        vis_backends=[],  # 不使用后端，直接返回图像
        save_dir=None
    )
    
    # 设置类别信息
    visualizer.dataset_meta = {'classes': classes}
    
    # 可视化
    visualizer.add_datasample(
        name='result',
        image=img,
        data_sample=data_sample,
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=None
    )
    
    # 获取可视化结果
    vis_img = visualizer.get_image()
    
    # 保存结果
    result_path = os.path.join(result_dir, f'result_{os.path.basename(img_file)}')
    cv2.imwrite(result_path, vis_img)

print(f"所有推理结果已保存至 {result_dir} 目录")