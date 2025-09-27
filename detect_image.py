#!/usr/bin/env python3
"""
图片检测脚本 - 对单张图片进行YOLO检测并可视化结果
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def detect_image(image_path, model_path, conf_threshold=0.5, save_result=True):
    """对图片进行检测并可视化"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Processing image: {image_path}")

    # 进行检测
    results = model(str(image_path), conf=conf_threshold)

    # 获取检测结果
    result = results[0]

    # 加载原图
    image = Image.open(image_path)

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # 类别名称和颜色
    class_names = {0: "hero", 1: "ordinary"}
    colors = {0: 'green', 1: 'red'}

    # 绘制检测框
    if result.boxes is not None:
        boxes = result.boxes.data.cpu().numpy()
        print(f"Detected {len(boxes)} objects:")

        for i, box in enumerate(boxes):
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)

            print(f"  {i+1}. {class_names[cls]}: confidence={conf:.3f}, bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

            # 绘制边界框
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor=colors[cls], facecolor='none')
            ax.add_patch(rect)

            # 添加标签
            label = f"{class_names[cls]}: {conf:.2f}"
            ax.text(x1, y1-10, label, bbox=dict(boxstyle="round,pad=0.3",
                   facecolor=colors[cls], alpha=0.7), color='white', fontsize=10)

    ax.set_title(f"Detection Results - {image_path.name}")
    ax.axis('off')

    if save_result:
        output_path = image_path.parent / f"{image_path.stem}_detected.jpg"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Result saved to: {output_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='YOLO图片检测脚本')
    parser.add_argument('input_image', help='输入图片路径')
    parser.add_argument('-m', '--model', default='runs/detect/train/weights/best.pt',
                       help='模型权重路径（默认：runs/detect/train/weights/best.pt）')
    parser.add_argument('-c', '--conf', type=float, default=0.5,
                       help='置信度阈值（默认：0.5）')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果图片')

    args = parser.parse_args()

    # 输入路径
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return

    # 模型路径
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("Please make sure you have trained the model first.")
        return

    print("=" * 50)
    print("颠球游戏 - 图片检测")
    print("=" * 50)

    try:
        detect_image(input_path, model_path, args.conf, not args.no_save)
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()