import os

voc_dir = 'data/VOCdevkit/VOC2012/Annotations'  # 替换成你的Annotations目录路径
empty_files = []

for xml_file in os.listdir(voc_dir):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(voc_dir, xml_file)
        if os.path.getsize(xml_path) == 0:
            empty_files.append(xml_file)

if empty_files:
    print(f"发现{len(empty_files)}个空的XML文件:")
    for file in empty_files:
        print(file)
else:
    print("未发现空的XML文件")