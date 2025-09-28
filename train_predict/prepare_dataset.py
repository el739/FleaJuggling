#!/usr/bin/env python3
"""
数据预处理脚本 - 将标注数据组织为YOLO训练格式
"""

import os
import shutil
from pathlib import Path
import random

def create_dataset_structure():
    """创建YOLO训练所需的目录结构"""
    dataset_dir = Path("dataset")

    # 创建目录结构
    dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val"
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    return dataset_dir

def split_data(train_ratio=0.8):
    """将数据分割为训练集和验证集"""
    screenshots_dir = Path("screenshots")
    labelled_dir = Path("labelled")

    # 获取所有标注文件
    label_files = list(labelled_dir.glob("*.txt"))
    # 排除 classes.txt
    label_files = [f for f in label_files if f.name != "classes.txt"]

    # 检查对应的截图是否存在
    valid_pairs = []
    for label_file in label_files:
        image_name = label_file.stem + ".png"
        image_path = screenshots_dir / image_name
        if image_path.exists():
            valid_pairs.append((image_path, label_file))
        else:
            print(f"Warning: No image found for {label_file.name}")

    print(f"Found {len(valid_pairs)} valid image-label pairs")

    # 随机打乱并分割
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)

    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")

    return train_pairs, val_pairs

def copy_files(pairs, dataset_dir, split_name):
    """复制文件到对应的训练/验证目录"""
    images_dir = dataset_dir / "images" / split_name
    labels_dir = dataset_dir / "labels" / split_name

    for image_path, label_path in pairs:
        # 复制图片
        dest_image = images_dir / image_path.name
        shutil.copy2(image_path, dest_image)

        # 复制标注
        dest_label = labels_dir / label_path.name
        shutil.copy2(label_path, dest_label)

    print(f"Copied {len(pairs)} files to {split_name} set")

def create_data_yaml(dataset_dir):
    """创建YOLO训练配置文件"""
    data_yaml_content = f"""# 颠球游戏数据集配置
path: {dataset_dir.absolute()}  # 数据集根目录
train: images/train  # 训练图片路径 (相对于path)
val: images/val      # 验证图片路径 (相对于path)

# 类别数量
nc: 2

# 类别名称
names:
  0: hero      # 玩家
  1: ordinary  # 球
"""

    data_yaml_path = dataset_dir / "data.yaml"
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)

    print(f"Created data.yaml at {data_yaml_path}")

def validate_annotations():
    """验证标注文件格式"""
    labelled_dir = Path("labelled")
    classes_file = labelled_dir / "classes.txt"

    # 读取类别信息
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Classes: {classes}")

    # 检查几个标注文件的格式
    label_files = [f for f in labelled_dir.glob("*.txt") if f.name != "classes.txt"]

    print(f"\nValidating {len(label_files)} annotation files...")

    for i, label_file in enumerate(label_files[:3]):  # 检查前3个文件
        print(f"\nChecking {label_file.name}:")
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = parts
                    print(f"  Line {line_num}: class={class_id}, bbox=({x_center}, {y_center}, {width}, {height})")
                else:
                    print(f"  Line {line_num}: Invalid format - {line}")

def main():
    """主函数"""
    print("=" * 50)
    print("数据预处理 - 准备YOLO训练数据")
    print("=" * 50)

    # 验证标注格式
    print("1. 验证标注文件...")
    validate_annotations()

    # 创建数据集目录结构
    print("\n2. 创建数据集目录结构...")
    dataset_dir = create_dataset_structure()

    # 分割数据
    print("\n3. 分割训练和验证数据...")
    train_pairs, val_pairs = split_data(train_ratio=0.8)

    # 复制文件
    print("\n4. 复制文件到目标目录...")
    copy_files(train_pairs, dataset_dir, "train")
    copy_files(val_pairs, dataset_dir, "val")

    # 创建配置文件
    print("\n5. 创建YOLO配置文件...")
    create_data_yaml(dataset_dir)

    print("\n" + "=" * 50)
    print("数据预处理完成!")
    print("=" * 50)
    print(f"数据集目录: {dataset_dir.absolute()}")
    print("现在可以运行 train.py 开始训练")

if __name__ == "__main__":
    main()