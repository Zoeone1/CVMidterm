import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
import random
import requests
from io import BytesIO
import cv2
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from tqdm import tqdm  # 进度条
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import SequentialSampler
from torch.utils.data import Subset
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

from torchvision.transforms import functional as F


class ResizeTransform:
    def __init__(self, max_size=400):
        self.max_size = max_size

    def __call__(self, img):
        h, w = img.shape[-2:]
        scale = self.max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return F.resize(img, [new_h, new_w])


def get_transform(train):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    #transforms_list.append(transforms.Resize((400, 400)))
    transforms_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]  # ImageNet标准差
    ))
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms_list)


def collate_fn(batch):
    return tuple(zip(*batch))


class VOC2012Dataset:
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')

        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']

        split_file = os.path.join(root_dir, 'ImageSets', 'Main', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, f'{image_name}.jpg')
        annotation_path = os.path.join(self.annotation_dir, f'{image_name}.xml')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []
        masks = []

        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index(obj_name))

            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask[ymin:ymax, xmin:xmax] = 1
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def get_maskrcnn_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        num_classes
    )

    return model


def calculate_mAP(results, targets, iou_threshold=0.5):
    aps = []
    for result, target in zip(results, targets):
        if len(result['boxes']) == 0 or len(target['boxes']) == 0:
            aps.append(0.0)
            continue

        # 确保所有张量都在CPU上并转换为NumPy数组
        pred_boxes = result['boxes'].cpu().numpy() if torch.is_tensor(result['boxes']) else np.array(result['boxes'])
        true_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else np.array(target['boxes'])
        pred_labels = result['labels'].cpu().numpy() if torch.is_tensor(result['labels']) else np.array(result['labels'])
        true_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else np.array(target['labels'])

        # 确保标签是1维数组
        pred_labels = pred_labels.flatten()
        true_labels = true_labels.flatten()

        # 简化的mAP计算
        ious = []
        for i, pred_box in enumerate(pred_boxes):
            for j, true_box in enumerate(true_boxes):
                # 只有当类别匹配时才计算IoU
                if pred_labels[i] == true_labels[j]:
                    iou = calculate_iou(pred_box, true_box)
                    ious.append(iou)

        # 如果IoU大于阈值且类别正确，则认为检测正确
        if len(ious) > 0:
            correct = np.mean([iou > iou_threshold for iou in ious])
            aps.append(correct)
        else:
            aps.append(0.0)

    return np.mean(aps) if aps else 0.0


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    box1 = np.asarray(box1).flatten()
    box2 = np.asarray(box2).flatten()

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10, log_dir=None):
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    best_mAP = 0.0

    # 启用混合精度训练
    scaler = GradScaler()

    # 梯度累积步数
    accumulation_steps = 4
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 训练阶段
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Train', unit='batch')
        for i, (images, targets) in enumerate(train_progress):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 使用混合精度进行前向传播
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses = losses / accumulation_steps

            # 反向传播
            scaler.scale(losses).backward()

            # 梯度累积更新
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += losses.item() * accumulation_steps
            train_progress.set_postfix({'Train Loss': losses.item() * accumulation_steps})

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # 验证阶段 - 使用更小的批量和更严格的内存管理
        model.eval()

        val_loss, val_mAP = evaluate_model(model, val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val mAP: {val_mAP:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('mAP/val', val_mAP, epoch)

        # 保存最佳模型
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            save_model(model, 'mask_rcnn_best_model.pth')
            print(f"New best model saved with mAP: {best_mAP:.4f}")
        
        scheduler.step()

        # 释放验证过程中占用的缓存
        torch.cuda.empty_cache()

    writer.close()
    
    # 打印TensorBoard启动命令
    print(f"\n训练完成！可以使用以下命令启动TensorBoard查看可视化结果：")
    print(f"tensorboard --logdir={log_dir}")
    print(f"然后在浏览器中打开 http://localhost:6006/")
    
    return model


def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    results = []
    targets = []
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for images, batch_targets in data_loader:
            images = list(image.to(device) for image in images)
            batch_targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
            
            # 计算验证损失
            model.train()  # 临时切换到训练模式以获取损失
            loss_dict = model(images, batch_targets)
            
            # 检查输出类型
            if isinstance(loss_dict, dict):
                # 处理损失字典
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
            elif isinstance(loss_dict, list):
                # 处理损失列表
                print("警告: 模型返回了损失列表而不是字典，尝试另一种处理方式")
                if all(isinstance(l, dict) for l in loss_dict):
                    losses = sum(sum(l.values()) for l in loss_dict)
                    total_loss += losses.item()
                else:
                    print("无法处理的损失格式，跳过此批次")
                    continue
            else:
                print(f"未知的损失格式: {type(loss_dict)}，跳过此批次")
                continue
            
            # 切换回评估模式获取预测结果
            model.eval()
            outputs = model(images)
            results.extend([{k: v.cpu() for k, v in out.items()} for out in outputs])
            targets.extend([{k: v.cpu() for k, v in t.items()} for t in batch_targets])
            
            # 释放不再需要的张量
            del images, batch_targets, loss_dict, outputs
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    # 计算mAP
    mAP = calculate_mAP(results, targets) if results else 0
    
    return avg_loss, mAP


def save_model(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': model.classes if hasattr(model, 'classes') else None
    }, path)


def load_model(path, num_classes=20):
    checkpoint = torch.load(path)
    model = get_maskrcnn_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if checkpoint['classes'] is not None:
        model.classes = checkpoint['classes']
    
    model.to(device)
    return model


def test_single_image(model, image_path, output_path=None, threshold=0.5):
    """测试单张图像"""
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 转换图像为模型输入格式
    transform = get_transform(train=False)
    img_tensor = transform(image)
    img_tensor = img_tensor.to(device).unsqueeze(0)
    
    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)[0]
    
    # 过滤低置信度预测
    keep = output['scores'] > threshold
    boxes = output['boxes'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()
    masks = output['masks'][keep].cpu().numpy() if 'masks' in output else None
    
    # 可视化结果
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title('Predictions')
    
    if masks is not None:
        for mask in masks:
            mask = mask[0]  # 移除通道维度
            plt.imshow(mask > 0.5, alpha=0.3, cmap='jet')
    
    # 绘制边界框和标签
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='b', facecolor='none')
        plt.gca().add_patch(rect)
        class_name = model.classes[label] if hasattr(model, 'classes') else str(label)
        plt.text(xmin, ymin, f'{class_name}: {score:.2f}', fontsize=10,
                bbox=dict(facecolor='b', alpha=0.5))
    
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"结果已保存到 {output_path}")
    else:
        plt.show()

def test_single_image_proposals(model, image_path, output_path=None, threshold=0.5):
    """测试单张图像并可视化Proposal"""
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from torchvision.models.detection.image_list import ImageList
    max_proposals=500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path).convert('RGB')
    transform = get_transform(train=False)
    img_tensor = transform(image).to(device).unsqueeze(0)
    model.eval()
    model.to(device)

    with torch.no_grad():
        image_size = img_tensor.shape[-2:]
        image_list = ImageList(img_tensor, [image_size])
        features = model.backbone(img_tensor)
        if isinstance(features, dict):
            features = list(features.values())
        anchors = model.rpn.anchor_generator(image_list, features)
        objectness_list, pred_bbox_deltas_list = model.rpn.head(features)
        objectness = torch.cat([o.flatten() for o in objectness_list])
        pred_bbox_deltas = torch.cat([d.permute(1, 2, 3, 0).reshape(-1, 4) for d in pred_bbox_deltas_list])
        proposals = model.rpn.box_coder.decode(pred_bbox_deltas, anchors)
        proposals = proposals.view(-1, 4)
        proposals = torchvision.ops.clip_boxes_to_image(proposals, image_size)
        scores = objectness.sigmoid().flatten()
        keep = scores > threshold
        proposals = proposals[keep]
        scores = scores[keep]
        keep = torchvision.ops.nms(proposals, scores, iou_threshold=0.7)
        proposals = proposals[keep]
        scores = scores[keep]
        if len(proposals) > max_proposals:
            keep = torch.topk(scores, max_proposals).indices
            proposals = proposals[keep]
            scores = scores[keep]
        proposals = proposals.cpu().numpy()
        scores = scores.cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title('RPN Proposals')

    for box, score in zip(proposals, scores):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=0.8,
            edgecolor=(1, 0, 0, 0.3),
            facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(xmin, ymin, f'{score:.2f}', fontsize=6, 
                bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"结果已保存到 {output_path}")
    else:
        plt.show()

def main(args):
    # 准备数据集
    print("准备VOC数据集...")
    voc_root = args.data_dir
    
    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)
    
    full_train_dataset = VOC2012Dataset(voc_root, split='train', transforms=train_transform)
    full_val_dataset = VOC2012Dataset(voc_root, split='val', transforms=test_transform)

    # 定义子集大小
    train_size = 5000
    val_size = 1000

    train_size = min(train_size, len(full_train_dataset))
    val_size = min(val_size, len(full_val_dataset))

    train_indices = torch.randperm(len(full_train_dataset))[:train_size].tolist()
    val_indices = torch.randperm(len(full_val_dataset))[:val_size].tolist()

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    num_classes = 20
    
    if args.mode == 'train':
        print("创建Mask R-CNN模型...")
        model = get_maskrcnn_model(num_classes)
        model.classes = train_dataset.dataset.classes  # 修正属性访问
        model.to(device)
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        
        log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(log_dir, exist_ok=True)
        
        print("开始训练Mask R-CNN模型...")
        model = train_model(model, train_loader, val_loader, optimizer, scheduler,
                            num_epochs=args.epochs, log_dir=log_dir)
        
        save_model(model, 'mask_rcnn_final_model.pth')
        print("训练完成，模型已保存")
    
    elif args.mode == 'test':
        print(f"加载模型 {args.model_path}...")
        model = load_model(args.model_path, num_classes)
        
        if args.test_image:
            if args.output_proposals:
                print(f"测试图像 {args.test_image}，输出 proposals...")
                test_single_image_proposals(model, args.test_image, args.output_path, threshold=args.threshold)
            else:
                print(f"测试图像 {args.test_image}，输出最终预测结果...")
                test_single_image(model, args.test_image, args.output_path, threshold=args.threshold)
        else:
            print("测试验证集...")
            val_loss, val_mAP = evaluate_model(model, val_loader)
            print(f"验证集结果 - Loss: {val_loss:.4f}, mAP: {val_mAP:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mask R-CNN训练和测试脚本')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                       help='运行模式: train 或 test')
    parser.add_argument('--data_dir', type=str, default='data/VOCdevkit/VOC2012',
                       help='VOC数据集目录路径')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='训练/测试的批大小')
    parser.add_argument('--epochs', type=int, default=15,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='学习率')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='测试时模型路径')
    parser.add_argument('--test_image', type=str, default=None,
                       help='测试单张图像的路径')
    parser.add_argument('--output_path', type=str, default=None,
                       help='测试结果的保存路径')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='检测阈值')
    parser.add_argument('--output_proposals', action='store_true',
                    help='是否只输出 proposals 而非最终预测结果（仅在 test_image 模式下生效）')

    args = parser.parse_args()
    main(args)