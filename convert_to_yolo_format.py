#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将COCO格式标注转换为YOLO格式
支持旋转边界框数据转换
"""

import json
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def convert_coco_to_yolo(coco_file, images_dir, output_dir):
    """
    将COCO格式标注转换为YOLO格式
    
    Args:
        coco_file: COCO标注文件路径
        images_dir: 图像目录路径
        output_dir: 输出目录路径
    """
    
    # 创建输出目录结构
    output_dir = Path(output_dir)
    yolo_images_dir = output_dir / 'images'
    yolo_labels_dir = output_dir / 'labels'
    
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取COCO标注
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    print(f"正在转换COCO标注到YOLO格式...")
    print(f"输入: {coco_file}")
    print(f"输出: {output_dir}")
    
    # 创建图像ID到文件名的映射
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 按图像分组标注
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    converted_count = 0
    total_annotations = 0
    
    # 处理每个图像
    for image_id, image_info in tqdm(image_id_to_info.items(), desc="转换图像"):
        image_file = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
        # 复制图像文件
        src_image_path = Path(images_dir) / image_file
        dst_image_path = yolo_images_dir / image_file
        
        if src_image_path.exists():
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"警告: 图像文件不存在 {src_image_path}")
            continue
        
        # 创建对应的标签文件
        label_file = yolo_labels_dir / (Path(image_file).stem + '.txt')
        
        yolo_annotations = []
        
        # 处理该图像的所有标注
        if image_id in image_annotations:
            for ann in image_annotations[image_id]:
                # YOLO格式: class_id center_x center_y width height (归一化坐标)
                
                # 获取边界框信息
                if 'rotated_bbox' in ann and ann.get('annotation_type') == 'rotated_bbox':
                    # 使用旋转边界框计算的轴对齐边界框
                    bbox = ann['bbox']  # [x, y, w, h]
                else:
                    # 使用标准边界框
                    bbox = ann['bbox']  # [x, y, w, h]
                
                x, y, w, h = bbox
                
                # 转换为YOLO格式 (归一化的中心坐标和尺寸)
                center_x = (x + w / 2) / image_width
                center_y = (y + h / 2) / image_height
                norm_width = w / image_width
                norm_height = h / image_height
                
                # 确保坐标在[0,1]范围内
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                # YOLO类别ID (golf_club = 0)
                class_id = 0
                
                yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                yolo_annotations.append(yolo_line)
                total_annotations += 1
        
        # 写入标签文件
        with open(label_file, 'w') as f:
            f.write('\\n'.join(yolo_annotations))
        
        converted_count += 1
    
    print(f"\\n转换完成!")
    print(f"  转换图像数: {converted_count}")
    print(f"  转换标注数: {total_annotations}")
    print(f"  输出目录: {output_dir}")
    
    return converted_count, total_annotations

def create_yolo_dataset_yaml(output_dir, dataset_name="golf_club_detection"):
    """创建YOLO数据集配置文件"""
    
    yaml_content = f'''# YOLO数据集配置文件
# 高尔夫球杆检测数据集

# 数据集路径 (相对于此文件的路径)
path: {output_dir.absolute()}  # 数据集根目录
train: images  # 训练图像目录 (相对于path)
val: images    # 验证图像目录 (相对于path)

# 类别数量
nc: 1  # 类别数量

# 类别名称
names:
  0: golf_club  # 高尔夫球杆
'''
    
    yaml_file = output_dir / f'{dataset_name}.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"YOLO数据集配置文件已创建: {yaml_file}")
    return yaml_file

def split_yolo_dataset(yolo_dir, train_ratio=0.8):
    """将YOLO数据集分割为训练集和验证集"""
    
    yolo_dir = Path(yolo_dir)
    images_dir = yolo_dir / 'images'
    labels_dir = yolo_dir / 'labels'
    
    # 创建训练和验证目录
    train_images_dir = yolo_dir / 'train' / 'images'
    train_labels_dir = yolo_dir / 'train' / 'labels'
    val_images_dir = yolo_dir / 'val' / 'images'
    val_labels_dir = yolo_dir / 'val' / 'labels'
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    # 随机打乱
    import random
    random.shuffle(image_files)
    
    # 分割数据
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"分割YOLO数据集:")
    print(f"  训练集: {len(train_files)} 张图像")
    print(f"  验证集: {len(val_files)} 张图像")
    
    # 移动训练集文件
    for img_file in tqdm(train_files, desc="移动训练集"):
        # 移动图像
        shutil.move(str(img_file), str(train_images_dir / img_file.name))
        
        # 移动对应的标签
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.move(str(label_file), str(train_labels_dir / label_file.name))
    
    # 移动验证集文件
    for img_file in tqdm(val_files, desc="移动验证集"):
        # 移动图像
        shutil.move(str(img_file), str(val_images_dir / img_file.name))
        
        # 移动对应的标签
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.move(str(label_file), str(val_labels_dir / label_file.name))
    
    # 删除原始目录
    if images_dir.exists() and not list(images_dir.iterdir()):
        images_dir.rmdir()
    if labels_dir.exists() and not list(labels_dir.iterdir()):
        labels_dir.rmdir()
    
    return len(train_files), len(val_files)

def create_yolo_training_script(output_dir, config_file="golf_club_detection.yaml"):
    """创建YOLO训练脚本"""
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO高尔夫球杆检测训练脚本
"""

from ultralytics import YOLO
import torch

def train_yolo_golf_club_detector():
    """训练YOLO高尔夫球杆检测模型"""
    
    print("开始训练YOLO高尔夫球杆检测模型...")
    print(f"使用设备: {{'CUDA' if torch.cuda.is_available() else 'CPU'}}")
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用YOLOv8 nano模型
    
    # 训练参数
    results = model.train(
        data='{config_file}',             # 数据集配置文件
        epochs=100,                       # 训练轮数
        imgsz=640,                       # 图像尺寸
        batch=16,                        # 批次大小
        lr0=0.01,                        # 初始学习率
        weight_decay=0.0005,             # 权重衰减
        momentum=0.937,                  # 动量
        patience=50,                     # 早停耐心值
        save=True,                       # 保存检查点
        save_period=10,                  # 保存周期
        cache=False,                     # 不使用缓存(内存不足时)
        device=0 if torch.cuda.is_available() else 'cpu',  # 设备
        workers=4,                       # 数据加载器工作进程数
        project='runs/detect',           # 项目目录
        name='golf_club_yolo',           # 实验名称
        exist_ok=True,                   # 允许覆盖现有实验
        pretrained=True,                 # 使用预训练权重
        optimizer='SGD',                 # 优化器
        verbose=True,                    # 详细输出
        seed=42,                         # 随机种子
        deterministic=True,              # 确定性训练
        single_cls=True,                 # 单类检测
        rect=False,                      # 矩形训练
        cos_lr=True,                     # 余弦学习率调度
        close_mosaic=10,                 # 关闭马赛克增强的轮数
        resume=False,                    # 不恢复训练
        amp=True,                        # 自动混合精度
        fraction=1.0,                    # 使用数据集的比例
        profile=False,                   # 不进行性能分析
        # 数据增强参数
        hsv_h=0.015,                     # 色调增强
        hsv_s=0.7,                       # 饱和度增强
        hsv_v=0.4,                       # 明度增强
        degrees=0.0,                     # 旋转角度
        translate=0.1,                   # 平移
        scale=0.5,                       # 缩放
        shear=0.0,                       # 剪切
        perspective=0.0,                 # 透视变换
        flipud=0.0,                      # 上下翻转
        fliplr=0.5,                      # 左右翻转
        mosaic=1.0,                      # 马赛克增强
        mixup=0.0,                       # 混合增强
        copy_paste=0.0,                  # 复制粘贴增强
    )
    
    print("训练完成!")
    print(f"最佳模型保存在: {{results.save_dir}}")
    
    return results

if __name__ == "__main__":
    train_yolo_golf_club_detector()
'''
    
    script_file = output_dir / 'train_yolo.py'
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"YOLO训练脚本已创建: {script_file}")
    return script_file

def main():
    """主函数"""
    print("=" * 60)
    print("COCO到YOLO格式转换工具")
    print("=" * 60)
    
    # 配置路径
    train_coco_file = "dataset/train/annotations/instances.json"
    train_images_dir = "dataset/train/images"
    val_coco_file = "dataset/val/annotations/instances.json"
    val_images_dir = "dataset/val/images"
    output_dir = Path("yolo_dataset_full")
    
    # 检查输入文件
    train_exists = Path(train_coco_file).exists() and Path(train_images_dir).exists()
    val_exists = Path(val_coco_file).exists() and Path(val_images_dir).exists()
    
    if not train_exists:
        print(f"错误: 训练集文件不存在")
        print(f"  COCO文件: {train_coco_file}")
        print(f"  图像目录: {train_images_dir}")
        return
    
    print(f"✅ 找到训练集数据")
    if val_exists:
        print(f"✅ 找到验证集数据")
    else:
        print(f"⚠️  验证集数据不存在，仅使用训练集")
    
    # 创建临时输出目录
    temp_output_dir = output_dir / "temp"
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    total_count = 0
    total_annotations = 0
    
    # 转换训练集
    print("\\n1. 转换训练集...")
    train_count, train_annotations = convert_coco_to_yolo(
        train_coco_file, train_images_dir, temp_output_dir
    )
    total_count += train_count
    total_annotations += train_annotations
    
    # 转换验证集（如果存在）
    if val_exists:
        print("\\n2. 转换验证集...")
        val_count, val_annotations = convert_coco_to_yolo(
            val_coco_file, val_images_dir, temp_output_dir
        )
        total_count += val_count
        total_annotations += val_annotations
        print(f"  验证集转换: {val_count} 图像, {val_annotations} 标注")
    
    print(f"\\n📊 总计转换: {total_count} 图像, {total_annotations} 标注")
    
    # 分割数据集
    print("\\n3. 分割完整数据集...")
    train_files, val_files = split_yolo_dataset(temp_output_dir, train_ratio=0.8)
    
    # 移动到最终目录
    final_images_dir = output_dir / "images"
    final_labels_dir = output_dir / "labels"
    
    if final_images_dir.exists():
        shutil.rmtree(final_images_dir)
    if final_labels_dir.exists():
        shutil.rmtree(final_labels_dir)
    
    # 移动train和val目录到最终位置
    shutil.move(str(temp_output_dir / "train"), str(output_dir / "train"))
    shutil.move(str(temp_output_dir / "val"), str(output_dir / "val"))
    
    # 清理临时目录
    shutil.rmtree(temp_output_dir)
    
    # 创建配置文件
    print("\\n4. 创建YOLO配置文件...")
    yaml_file = create_yolo_dataset_yaml(output_dir, "golf_club_detection_full")
    
    # 创建训练脚本
    print("\\n5. 创建训练脚本...")
    script_file = create_yolo_training_script(output_dir, "golf_club_detection_full.yaml")
    
    print("\\n" + "=" * 60)
    print("转换完成! 🎉")
    print("=" * 60)
    print(f"📁 YOLO数据集目录: {output_dir}")
    print(f"📊 训练集: {train_files} 张图像")
    print(f"📊 验证集: {val_files} 张图像")
    print(f"📊 总图像数: {total_count}")
    print(f"📋 总标注数: {total_annotations}")
    print(f"⚙️  配置文件: {yaml_file}")
    print(f"🚀 训练脚本: {script_file}")
    
    print("\\n下一步操作:")
    print("1. 进入数据集目录: cd yolo_dataset_full")
    print("2. 开始训练: python train_yolo.py")
    print("3. 监控训练: python ../monitor_yolo_training.py --runs_dir runs/detect/golf_club_yolo")

if __name__ == "__main__":
    main() 