import json
import numpy as np
from pycocotools.coco import COCO

# 配置参数
input_ann_file = 'data/VOCdevkit/coco/voc12_val.json'  # 原始标注文件
output_ann_file = 'data/VOCdevkit/coco/voc12_val_subset.json'  # 输出子集标注文件
sample_ratio = 0.1  # 抽样比例（如 0.1 表示抽取 10%）
seed = 42  # 随机种子，确保可复现

# 加载原始 COCO 数据
coco = COCO(input_ann_file)
img_ids = list(coco.imgs.keys())

# 随机抽样图片 ID
np.random.seed(seed)
selected_img_ids = np.random.choice(img_ids, size=int(len(img_ids) * sample_ratio), replace=False)

# 构建子集数据，使用 get() 方法避免 KeyError
subset_data = {
    "info": coco.dataset.get("info", {}),  # 使用 get() 方法，若不存在则返回空字典
    "licenses": coco.dataset.get("licenses", []),
    "categories": coco.dataset["categories"],  # 假设 categories 字段一定存在
    "images": [coco.imgs[img_id] for img_id in selected_img_ids],
    "annotations": []
}

# 筛选对应的标注
if 'annotations' in coco.dataset:
    ann_ids = coco.getAnnIds(imgIds=selected_img_ids)
    subset_data["annotations"] = [coco.loadAnns(ann_id)[0] for ann_id in ann_ids]

# 保存子集标注文件
with open(output_ann_file, 'w') as f:
    json.dump(subset_data, f, indent=2)

print(f"已抽取 {len(selected_img_ids)} 张图片，保存至 {output_ann_file}")