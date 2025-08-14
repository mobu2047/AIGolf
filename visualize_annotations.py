#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高尔夫球杆标注数据可视化工具
适配视频标注系统输出的数据格式
支持YOLO格式标注和JSON详细信息的可视化
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 固定路径配置
DATASET_DIR = Path(r"C:\Users\Administrator\Desktop\AIGolf\dataset")

class AnnotationVisualizer:
    """标注数据可视化器"""
    
    def __init__(self, dataset_dir=None):
        """
        初始化可视化器
        
        Args:
            dataset_dir: 数据集目录路径，默认使用固定路径
        """
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR
        self.images_dir = self.dataset_dir / "images"
        self.annotations_dir = self.dataset_dir / "annotations"
        
        print(f"🎯 标注数据可视化器已初始化")
        print(f"📁 数据集目录: {self.dataset_dir.absolute()}")
        print(f"📁 图像目录: {self.images_dir.absolute()}")
        print(f"📁 标注目录: {self.annotations_dir.absolute()}")
        
        # 检查目录是否存在
        self._check_directories()
    
    def _check_directories(self):
        """检查必要目录是否存在"""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.images_dir}")
        
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"标注目录不存在: {self.annotations_dir}")
    
    def get_image_files(self):
        """获取所有图像文件"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))
            image_files.extend(list(self.images_dir.glob(f"*{ext.upper()}")))
        
        return sorted(image_files)
    
    def load_yolo_annotation(self, image_path):
        """
        加载YOLO格式标注文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            list: YOLO标注列表，每个元素为 [class_id, x_center, y_center, width, height]
        """
        annotation_file = self.annotations_dir / f"{image_path.stem}.txt"
        
        if not annotation_file.exists():
            return []
        
        annotations = []
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append([class_id, x_center, y_center, width, height])
        except Exception as e:
            print(f"⚠️ 读取YOLO标注文件失败 {annotation_file}: {e}")
        
        return annotations
    
    def load_detailed_annotation(self, image_path):
        """
        加载详细标注信息（JSON格式）
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 详细标注信息，如果文件不存在则返回None
        """
        detail_file = self.annotations_dir / f"{image_path.stem}_detail.json"
        
        if not detail_file.exists():
            return None
        
        try:
            with open(detail_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取详细标注文件失败 {detail_file}: {e}")
            return None
    
    def yolo_to_bbox(self, yolo_annotation, img_width, img_height):
        """
        将YOLO格式坐标转换为边界框坐标
        
        Args:
            yolo_annotation: [class_id, x_center, y_center, width, height] (归一化坐标)
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            tuple: (x, y, w, h) 像素坐标
        """
        class_id, x_center, y_center, width, height = yolo_annotation
        
        # 转换为像素坐标
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # 计算左上角坐标
        x = x_center_px - width_px / 2
        y = y_center_px - height_px / 2
        
        return (x, y, width_px, height_px)
    
    def visualize_single_image(self, image_path, save_path=None, show_details=True):
        """
        可视化单个图像及其标注
        
        Args:
            image_path: 图像文件路径
            save_path: 保存路径（可选）
            show_details: 是否显示详细信息
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # 加载标注数据
        yolo_annotations = self.load_yolo_annotation(image_path)
        detailed_annotation = self.load_detailed_annotation(image_path)
        
        print(f"\n=== 图像信息 ===")
        print(f"📁 文件名: {image_path.name}")
        print(f"📏 尺寸: {img_width} x {img_height}")
        print(f"📝 YOLO标注数量: {len(yolo_annotations)}")
        print(f"📋 详细标注: {'有' if detailed_annotation else '无'}")
        
        # 创建matplotlib图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        
        # 显示图像
        ax.imshow(image_rgb)
        title = f'高尔夫球杆标注 - {image_path.name}'
        if len(yolo_annotations) > 0:
            title += f'\n(共{len(yolo_annotations)}个标注)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 绘制标注
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        
        # 优先使用详细标注信息绘制旋转边界框
        if detailed_annotation and 'rotated_corners' in detailed_annotation:
            self._draw_rotated_bbox(ax, detailed_annotation, colors[0])
            
            if show_details:
                self._show_detailed_info(detailed_annotation)
        
        # 绘制YOLO标注（轴对齐边界框）
        for i, yolo_ann in enumerate(yolo_annotations):
            color = colors[i % len(colors)]
            x, y, w, h = self.yolo_to_bbox(yolo_ann, img_width, img_height)
            
            # 绘制边界框
            rect = patches.Rectangle((x, y), w, h, 
                                   linewidth=3, 
                                   edgecolor=color, 
                                   facecolor=color,
                                   alpha=0.3)
            ax.add_patch(rect)
            
            # 绘制边框轮廓
            rect_outline = patches.Rectangle((x, y), w, h, 
                                           linewidth=3, 
                                           edgecolor=color, 
                                           facecolor='none',
                                           alpha=0.9)
            ax.add_patch(rect_outline)
            
            # 添加标签
            class_id = int(yolo_ann[0])
            class_name = "球杆" if class_id == 0 else f"类别{class_id}"
            
            ax.text(x + w/2, y - 20, f'{class_name} #{i+1}', 
                    fontsize=12, 
                    color='white', 
                    weight='bold',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))
            
            print(f"  📦 YOLO标注 {i+1}: 类别={class_id}, 边界框=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 可视化结果已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def _draw_rotated_bbox(self, ax, detailed_annotation, color):
        """绘制旋转边界框"""
        if 'rotated_corners' not in detailed_annotation:
            return
        
        rotated_corners = detailed_annotation['rotated_corners']
        
        # 创建多边形填充
        polygon = patches.Polygon(rotated_corners, 
                                linewidth=3, 
                                edgecolor=color, 
                                facecolor=color,
                                alpha=0.3)
        ax.add_patch(polygon)
        
        # 绘制边框
        polygon_outline = patches.Polygon(rotated_corners, 
                                        linewidth=3, 
                                        edgecolor=color, 
                                        facecolor='none',
                                        alpha=0.9)
        ax.add_patch(polygon_outline)
        
        # 计算中心点用于放置标签
        center_x = np.mean([corner[0] for corner in rotated_corners])
        center_y = np.mean([corner[1] for corner in rotated_corners])
        
        # 添加标签
        ax.text(center_x, center_y-30, '旋转边界框', 
                fontsize=12, 
                color='white', 
                weight='bold',
                ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))
        
        # 绘制球杆中心线（如果有端点信息）
        if 'points' in detailed_annotation:
            points = detailed_annotation['points']
            if len(points) == 2:
                point1, point2 = points
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
                       color='white', linewidth=4, alpha=0.8)
                ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
                       color=color, linewidth=2, alpha=1.0)
                
                # 标记端点
                ax.plot(point1[0], point1[1], 'o', color='white', markersize=8)
                ax.plot(point1[0], point1[1], 'o', color=color, markersize=6)
                ax.plot(point2[0], point2[1], 'o', color='white', markersize=8)
                ax.plot(point2[0], point2[1], 'o', color=color, markersize=6)
    
    def _show_detailed_info(self, detailed_annotation):
        """显示详细标注信息"""
        print(f"  🔍 详细标注信息:")
        print(f"    📝 标注模式: {detailed_annotation.get('mode', 'unknown')}")
        
        if 'points' in detailed_annotation:
            points = detailed_annotation['points']
            print(f"    📍 球杆端点: {points}")
        
        if 'length' in detailed_annotation:
            print(f"    📏 球杆长度: {detailed_annotation['length']:.1f}px")
        
        if 'angle' in detailed_annotation:
            print(f"    📐 球杆角度: {detailed_annotation['angle']:.1f}°")
        
        if 'club_width' in detailed_annotation:
            print(f"    📐 球杆宽度: {detailed_annotation['club_width']:.1f}px")
        
        if 'rotated_area' in detailed_annotation:
            print(f"    📊 旋转边界框面积: {detailed_annotation['rotated_area']:.1f}px²")
        
        if 'frame_info' in detailed_annotation:
            frame_info = detailed_annotation['frame_info']
            print(f"    🎬 视频信息:")
            print(f"      视频名称: {frame_info.get('video_name', 'unknown')}")
            print(f"      帧索引: {frame_info.get('frame_index', 'unknown')}")
            print(f"      时间戳: {frame_info.get('timestamp', 'unknown'):.2f}s")
    
    def visualize_multiple_samples(self, num_samples=20, save_dir="visualization_samples"):
        """
        可视化多个样本
        
        Args:
            num_samples: 要可视化的样本数量
            save_dir: 保存目录
        """
        # 创建保存目录
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 获取所有图像文件
        image_files = self.get_image_files()
        
        if not image_files:
            print("❌ 未找到图像文件！")
            return
        
        # 筛选有标注的图像
        images_with_annotations = []
        for img_path in image_files:
            yolo_annotations = self.load_yolo_annotation(img_path)
            if yolo_annotations:
                images_with_annotations.append(img_path)
        
        print(f"📊 数据集统计:")
        print(f"  总图像数: {len(image_files)}")
        print(f"  有标注的图像数: {len(images_with_annotations)}")
        print(f"  无标注的图像数: {len(image_files) - len(images_with_annotations)}")
        
        if not images_with_annotations:
            print("❌ 没有找到有标注的图像！")
            return
        
        # 随机选择样本
        selected_images = random.sample(images_with_annotations, 
                                      min(num_samples, len(images_with_annotations)))
        
        print(f"🎯 随机选择 {len(selected_images)} 张图像进行可视化")
        
        for i, image_path in enumerate(selected_images):
            print(f"\n{'='*60}")
            print(f"📸 可视化样本 {i+1}/{len(selected_images)}")
            
            save_path = save_dir / f"sample_{i+1:03d}_{image_path.stem}.png"
            self.visualize_single_image(image_path, save_path, show_details=True)
    
    def analyze_dataset_statistics(self):
        """分析数据集统计信息"""
        print(f"\n{'='*60}")
        print("📊 数据集统计分析")
        print(f"{'='*60}")
        
        image_files = self.get_image_files()
        
        if not image_files:
            print("❌ 未找到图像文件！")
            return
        
        # 统计信息
        total_images = len(image_files)
        images_with_yolo = 0
        images_with_detail = 0
        total_annotations = 0
        
        annotation_modes = {}
        video_sources = {}
        bbox_areas = []
        club_lengths = []
        club_angles = []
        
        for img_path in image_files:
            # 检查YOLO标注
            yolo_annotations = self.load_yolo_annotation(img_path)
            if yolo_annotations:
                images_with_yolo += 1
                total_annotations += len(yolo_annotations)
                
                # 计算边界框面积
                image = cv2.imread(str(img_path))
                if image is not None:
                    img_height, img_width = image.shape[:2]
                    for yolo_ann in yolo_annotations:
                        x, y, w, h = self.yolo_to_bbox(yolo_ann, img_width, img_height)
                        bbox_areas.append(w * h)
            
            # 检查详细标注
            detailed_annotation = self.load_detailed_annotation(img_path)
            if detailed_annotation:
                images_with_detail += 1
                
                # 统计标注模式
                mode = detailed_annotation.get('mode', 'unknown')
                annotation_modes[mode] = annotation_modes.get(mode, 0) + 1
                
                # 统计视频来源
                frame_info = detailed_annotation.get('frame_info', {})
                video_name = frame_info.get('video_name', 'unknown')
                video_sources[video_name] = video_sources.get(video_name, 0) + 1
                
                # 收集球杆信息
                if 'length' in detailed_annotation:
                    club_lengths.append(detailed_annotation['length'])
                if 'angle' in detailed_annotation:
                    club_angles.append(detailed_annotation['angle'])
        
        # 输出统计结果
        print(f"📁 总图像数: {total_images}")
        print(f"📝 有YOLO标注的图像: {images_with_yolo}")
        print(f"📋 有详细标注的图像: {images_with_detail}")
        print(f"📊 总标注数: {total_annotations}")
        
        if images_with_yolo > 0:
            print(f"📈 平均每张图像标注数: {total_annotations / images_with_yolo:.2f}")
        
        # 标注模式分布
        if annotation_modes:
            print(f"\n📝 标注模式分布:")
            for mode, count in annotation_modes.items():
                print(f"  {mode}: {count} 个")
        
        # 视频来源分布
        if video_sources:
            print(f"\n🎬 视频来源分布:")
            for video, count in sorted(video_sources.items()):
                print(f"  {video}: {count} 帧")
        
        # 边界框统计
        if bbox_areas:
            print(f"\n📦 边界框面积统计:")
            print(f"  最小面积: {min(bbox_areas):.1f}px²")
            print(f"  最大面积: {max(bbox_areas):.1f}px²")
            print(f"  平均面积: {np.mean(bbox_areas):.1f}px²")
            print(f"  中位数面积: {np.median(bbox_areas):.1f}px²")
        
        # 球杆长度统计
        if club_lengths:
            print(f"\n📏 球杆长度统计:")
            print(f"  最短长度: {min(club_lengths):.1f}px")
            print(f"  最长长度: {max(club_lengths):.1f}px")
            print(f"  平均长度: {np.mean(club_lengths):.1f}px")
        
        # 球杆角度统计
        if club_angles:
            print(f"\n📐 球杆角度统计:")
            print(f"  角度范围: {min(club_angles):.1f}° - {max(club_angles):.1f}°")
            print(f"  平均角度: {np.mean(club_angles):.1f}°")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高尔夫球杆标注数据可视化工具")
    parser.add_argument("--dataset_dir", "-d", type=str, 
                       help="数据集目录路径（默认使用固定路径）")
    parser.add_argument("--num_samples", "-n", type=int, default=20, 
                       help="可视化样本数量（默认20）")
    parser.add_argument("--save_dir", "-s", type=str, default="visualization_samples", 
                       help="保存目录（默认visualization_samples）")
    parser.add_argument("--single_image", "-i", type=str, 
                       help="可视化单个图像（指定图像文件名）")
    parser.add_argument("--stats_only", action="store_true", 
                       help="仅显示统计信息，不进行可视化")
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器
        visualizer = AnnotationVisualizer(args.dataset_dir)
        
        print("🏌️ 高尔夫球杆标注数据可视化工具")
        print("="*60)
        
        # 分析数据集统计信息
        visualizer.analyze_dataset_statistics()
        
        if args.stats_only:
            print("\n✅ 统计分析完成！")
            return
        
        if args.single_image:
            # 可视化单个图像
            image_path = visualizer.images_dir / args.single_image
            if image_path.exists():
                print(f"\n{'='*60}")
                print(f"📸 可视化单个图像: {args.single_image}")
                visualizer.visualize_single_image(image_path, show_details=True)
            else:
                print(f"❌ 图像文件不存在: {image_path}")
        else:
            # 可视化多个样本
            print(f"\n{'='*60}")
            print("📸 开始批量可视化...")
            visualizer.visualize_multiple_samples(args.num_samples, args.save_dir)
            
            print(f"\n{'='*60}")
            print("✅ 可视化完成！")
            print(f"📁 可以在 '{args.save_dir}' 目录中查看保存的图像")
    
    except Exception as e:
        print(f"❌ 运行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 