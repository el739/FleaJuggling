#!/usr/bin/env python3
"""
YOLO训练脚本 - 颠球游戏目标检测
"""

import os
from pathlib import Path
from ultralytics import YOLO

def train_yolo():
    # 检查数据集路径
    dataset_dir = Path("dataset")
    data_yaml = dataset_dir / "data.yaml"
    
    if not data_yaml.exists():
        print(f"Error: data.yaml not found at {data_yaml}")
        print("Please run the dataset organization script first.")
        return
    
    # 检查训练和验证目录
    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    train_label_dir = dataset_dir / "labels" / "train"
    val_label_dir = dataset_dir / "labels" / "val"
    
    for dir_path, dir_name in [(train_img_dir, "train images"), 
                               (val_img_dir, "val images"),
                               (train_label_dir, "train labels"), 
                               (val_label_dir, "val labels")]:
        if not dir_path.exists():
            print(f"Error: {dir_name} directory not found at {dir_path}")
            return
        
        file_count = len(list(dir_path.glob("*.*")))
        print(f"{dir_name}: {file_count} files")
    
    print("\n" + "="*50)
    print("Starting YOLO Training...")
    print("="*50)
    
    # 加载预训练模型
    model = YOLO("yolov8n.pt")  # 或使用 yolov8s.pt, yolov8m.pt 等
    
    # 训练配置 - 颠球游戏专用配置（无数据增强）
    training_config = {
        'data': str(data_yaml),
        'epochs': 100,  # 减少epoch数，适合小数据集
        'imgsz': 640,
        'batch': 8,    # 减小batch size适合小数据集
        'device': 0,   # 使用GPU，如果没有GPU可以改为'cpu'
        'workers': 2,  # 减少工作进程数
        'patience': 20,  # 减少早停耐心值
        'save': True,
        'save_period': 10,
        'cache': False,
    }
    
    try:
        # 开始训练
        results = model.train(**training_config)
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        print(f"Best model saved at: runs/detect/train/weights/best.pt")
        print(f"Training results saved at: runs/detect/train/")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Please check your dataset and configuration.")

def validate_dataset():
    """验证数据集的完整性"""
    dataset_dir = Path("dataset")
    
    # 检查文件对应关系
    train_imgs = set(f.stem for f in (dataset_dir / "images" / "train").glob("*.*"))
    train_labels = set(f.stem for f in (dataset_dir / "labels" / "train").glob("*.txt"))
    val_imgs = set(f.stem for f in (dataset_dir / "images" / "val").glob("*.*"))
    val_labels = set(f.stem for f in (dataset_dir / "labels" / "val").glob("*.txt"))
    
    print("Dataset validation:")
    print(f"Train - Images: {len(train_imgs)}, Labels: {len(train_labels)}")
    print(f"Val - Images: {len(val_imgs)}, Labels: {len(val_labels)}")
    
    # 检查不匹配的文件
    train_unmatched = train_imgs - train_labels
    val_unmatched = val_imgs - val_labels
    
    if train_unmatched:
        print(f"Train set - Images without labels: {train_unmatched}")
    if val_unmatched:
        print(f"Val set - Images without labels: {val_unmatched}")
    
    # 检查标注文件内容
    sample_label_file = next((dataset_dir / "labels" / "train").glob("*.txt"), None)
    if sample_label_file:
        with open(sample_label_file, 'r') as f:
            lines = f.readlines()
        print(f"Sample annotations from {sample_label_file.name}:")
        for i, line in enumerate(lines[:3]):  # 显示前3行
            print(f"  {line.strip()}")
        if len(lines) > 3:
            print(f"  ... and {len(lines) - 3} more annotations")

if __name__ == "__main__":
    # 首先验证数据集
    print("Validating dataset...")
    validate_dataset()
    
    # 询问是否继续训练
    response = input("\nDo you want to start training? (y/n): ")
    if response.lower() in ['y', 'yes']:
        train_yolo()
    else:
        print("Training cancelled.")