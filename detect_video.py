#!/usr/bin/env python3
"""
视频检测脚本 - 对视频进行逐帧YOLO检测并输出结果视频
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def setup_colors():
    """设置类别颜色"""
    colors = {
        0: (0, 255, 0),    # hero - 绿色
        1: (255, 0, 0),    # ordinary (球) - 红色
    }
    return colors

def draw_detections(frame, results, colors, class_names):
    """在帧上绘制检测结果"""
    annotated_frame = frame.copy()

    if results[0].boxes is not None:
        boxes = results[0].boxes.data.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)

            # 绘制边界框
            color = colors.get(cls, (255, 255, 255))
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 绘制标签和置信度
            label = f"{class_names[cls]}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # 绘制标签背景
            cv2.rectangle(annotated_frame,
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)

            # 绘制标签文字
            cv2.putText(annotated_frame, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated_frame

def process_video(input_path, output_path, model_path, conf_threshold=0.5):
    """处理视频文件"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # 类别名称
    class_names = {0: "hero", 1: "ordinary"}
    colors = setup_colors()

    # 打开输入视频
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')

            # YOLO检测
            results = model(frame, conf=conf_threshold, verbose=False)

            # 绘制检测结果
            annotated_frame = draw_detections(frame, results, colors, class_names)

            # 写入输出视频
            out.write(annotated_frame)

    except KeyboardInterrupt:
        print(f"\nProcessing interrupted at frame {frame_count}")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\nProcessing completed! Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLO视频检测脚本')
    parser.add_argument('input_video', help='输入视频路径')
    parser.add_argument('-o', '--output', help='输出视频路径（默认：input_detected.mp4）')
    parser.add_argument('-m', '--model', default='runs/detect/train/weights/best.pt',
                       help='模型权重路径（默认：runs/detect/train/weights/best.pt）')
    parser.add_argument('-c', '--conf', type=float, default=0.5,
                       help='置信度阈值（默认：0.5）')

    args = parser.parse_args()

    # 输入路径
    input_path = Path(args.input_video)
    if not input_path.exists():
        print(f"Error: Input video not found: {input_path}")
        return

    # 输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_detected.mp4"

    # 模型路径
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("Please make sure you have trained the model first.")
        return

    print("=" * 50)
    print("颠球游戏 - 视频检测")
    print("=" * 50)
    print(f"输入视频: {input_path}")
    print(f"输出视频: {output_path}")
    print(f"使用模型: {model_path}")
    print(f"置信度阈值: {args.conf}")
    print("=" * 50)

    try:
        process_video(input_path, output_path, model_path, args.conf)
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()