#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高尔夫球杆检测交互式标注工具 - YOLO格式版本
支持多种标注模式，自动生成YOLO格式标注文件
固定输入目录：C:\Users\Administrator\Desktop\AIGolf\videos
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import argparse
from datetime import datetime

# 标注模式配置
ANNOTATION_MODES = {
    'bbox': '轴对齐边界框',
    'rotated_bbox': '旋转边界框', 
    'line': '线段标注',
    'polygon': '多边形标注'
}

# 球杆宽度配置
MIN_CLUB_WIDTH_RATIO = 0.002  # 最小宽度比例
MAX_CLUB_WIDTH_RATIO = 0.008  # 最大宽度比例

# 屏幕适配配置
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800

# 固定输入目录
FIXED_INPUT_DIR = Path(r"C:\Users\Administrator\Desktop\AIGolf\videos")

class InteractiveAnnotator:
    """交互式标注器类"""
    
    def __init__(self, mode='rotated_bbox', output_format='yolo'):
        """
        初始化标注器
        
        Args:
            mode: 标注模式 ('bbox', 'rotated_bbox', 'line', 'polygon')
            output_format: 输出格式 ('yolo', 'coco')
        """
        self.mode = mode
        self.output_format = output_format
        
        # 固定输入目录
        self.input_dir = FIXED_INPUT_DIR
        
        # 标注状态
        self.current_image = None
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        
        # 标注数据
        self.club_points = []
        self.annotations = []
        
        # 控制标志
        self.drawing = False
        self.finish_annotation = False
        self.skip_image = False
        
        print(f"🎯 交互式标注器已初始化")
        print(f"📝 标注模式: {ANNOTATION_MODES.get(mode, mode)}")
        print(f"📄 输出格式: {output_format.upper()}")
        print(f"📁 固定输入目录: {self.input_dir.absolute()}")
    
    def annotate_dataset(self, output_dir=None):
        """
        标注固定目录中的图像
        
        Args:
            output_dir: 输出目录，如果为None则在输入目录创建annotations子目录
        """
        if not self.input_dir.exists():
            print(f"❌ 固定输入目录不存在: {self.input_dir}")
            print("请确保目录存在并包含图像文件")
            return
        
        # 设置输出目录
        if output_dir is None:
            output_path = self.input_dir / "annotations"
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(exist_ok=True)
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.input_dir.glob(f"*{ext}")))
            image_files.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"❌ 在目录 {self.input_dir} 中未找到图像文件")
            print("支持的图像格式: .jpg, .jpeg, .png, .bmp")
            return
        
        print(f"📁 在固定目录中找到 {len(image_files)} 个图像文件")
        print(f"📤 标注文件将保存到: {output_path}")
        
        # 开始标注流程
        self._show_instructions()
        
        annotated_count = 0
        skipped_count = 0
        
        for i, image_path in enumerate(image_files):
            print(f"\n📸 标注进度: {i+1}/{len(image_files)} - {image_path.name}")
            
            result = self.annotate_single_image(image_path)
            
            if result:
                # 保存标注结果
                self._save_annotation(image_path, result, output_path)
                annotated_count += 1
                print(f"✅ 已保存标注: {image_path.name}")
            else:
                skipped_count += 1
                print(f"⏭️ 跳过图像: {image_path.name}")
        
        print(f"\n🎉 标注完成!")
        print(f"✅ 成功标注: {annotated_count} 个图像")
        print(f"⏭️ 跳过: {skipped_count} 个图像")
        print(f"📁 标注文件保存在: {output_path}")
    
    def annotate_single_image(self, image_path):
        """
        标注单个图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 标注结果，如果跳过则返回None
        """
        # 重置状态
        self.club_points = []
        self.finish_annotation = False
        self.skip_image = False
        
        # 读取图像
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None
        
        # 缩放图像以适应显示
        self.display_image, self.scale_factor = self._resize_for_display(self.original_image)
        self.current_image = self.display_image.copy()
        
        # 创建窗口和设置鼠标回调
        cv2.namedWindow('标注图像', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('标注图像', self._mouse_callback)
        
        # 显示操作说明
        self._show_image_instructions(image_path)
        
        # 标注循环
        while True:
            cv2.imshow('标注图像', self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            # ESC键处理
            if key == 27:  # ESC
                if len(self.club_points) == 0:
                    self.skip_image = True
                else:
                    self.finish_annotation = True
            
            # 完成标注或跳过
            if self.finish_annotation or self.skip_image:
                break
        
        cv2.destroyWindow('标注图像')
        
        # 返回标注结果
        if self.skip_image:
            return None
        
        if len(self.club_points) >= 2:
            return self._process_annotation_result()
        
        return None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            
            # 将显示坐标转换为原始图像坐标
            original_point = self._scale_point_to_original((x, y))
            self.club_points.append(original_point)
            
            # 在显示图像上绘制点
            cv2.circle(self.display_image, (x, y), max(1, int(3 * self.scale_factor)), (0, 255, 0), -1)
            
            print(f"标注点 {len(self.club_points)}: 显示坐标({x}, {y}) -> 原始坐标{original_point}")
            
            # 如果已有两个点，绘制预览并完成标注
            if len(self.club_points) == 2:
                self._draw_annotation_preview()
                self.finish_annotation = True
    
    def _draw_annotation_preview(self):
        """绘制标注预览"""
        if len(self.club_points) < 2:
            return
        
        # 重新绘制显示图像
        self.display_image = self._resize_for_display(self.original_image)[0]
        
        # 绘制标注点
        for i, point in enumerate(self.club_points):
            display_point = self._scale_point_to_display(point)
            cv2.circle(self.display_image, display_point, max(1, int(3 * self.scale_factor)), (0, 255, 0), -1)
            cv2.putText(self.display_image, f"{i+1}", 
                       (display_point[0] + 5, display_point[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制连线
        if len(self.club_points) >= 2:
            point1_display = self._scale_point_to_display(self.club_points[0])
            point2_display = self._scale_point_to_display(self.club_points[1])
            cv2.line(self.display_image, point1_display, point2_display, (0, 255, 0), max(1, int(2 * self.scale_factor)))
        
        # 根据模式绘制不同的标注形状
        if self.mode in ['bbox', 'rotated_bbox', 'polygon']:
            self._draw_bounding_shape()
        
        # 显示信息
        self._draw_annotation_info()
    
    def _draw_bounding_shape(self):
        """根据模式绘制边界形状"""
        if len(self.club_points) < 2:
            return
        
        point1 = np.array(self.club_points[0])
        point2 = np.array(self.club_points[1])
        
        # 计算球杆信息
        direction = point2 - point1
        length = np.linalg.norm(direction)
        
        if length == 0:
            return
        
        # 计算自适应宽度
        img_height, img_width = self.original_image.shape[:2]
        club_width = self._calculate_adaptive_width(img_width, img_height, length)
        
        if self.mode == 'rotated_bbox' or self.mode == 'polygon':
            # 计算旋转边界框
            corners = self._calculate_rotated_bbox(point1, point2, club_width)
            if corners:
                # 转换为显示坐标并绘制
                display_corners = [self._scale_point_to_display(corner) for corner in corners]
                pts = np.array(display_corners, np.int32)
                
                if self.mode == 'polygon':
                    cv2.fillPoly(self.display_image, [pts], (0, 255, 255, 100))
                
                cv2.polylines(self.display_image, [pts], True, (255, 0, 0), max(1, int(2 * self.scale_factor)))
        
        elif self.mode == 'bbox':
            # 计算轴对齐边界框
            bbox = self._calculate_axis_aligned_bbox(point1, point2, club_width, img_width, img_height)
            if bbox:
                x, y, w, h = bbox
                # 转换为显示坐标
                display_x = int(x * self.scale_factor)
                display_y = int(y * self.scale_factor)
                display_w = int(w * self.scale_factor)
                display_h = int(h * self.scale_factor)
                
                cv2.rectangle(self.display_image, (display_x, display_y), 
                            (display_x + display_w, display_y + display_h), 
                            (0, 0, 255), max(1, int(2 * self.scale_factor)))
    
    def _draw_annotation_info(self):
        """绘制标注信息"""
        if len(self.club_points) < 2:
            return
        
        point1 = np.array(self.club_points[0])
        point2 = np.array(self.club_points[1])
        direction = point2 - point1
        length = np.linalg.norm(direction)
        angle = np.degrees(np.arctan2(direction[1], direction[0]))
        
        info_texts = [
            f"模式: {ANNOTATION_MODES.get(self.mode, self.mode)}",
            f"长度: {length:.1f}px",
            f"角度: {angle:.1f}°"
        ]
        
        # 在图像上显示信息
        for i, text in enumerate(info_texts):
            y_pos = 30 + i * 25
            cv2.putText(self.display_image, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(self.display_image, text, (9, y_pos - 1), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    def _process_annotation_result(self):
        """处理标注结果，生成最终的标注数据"""
        if len(self.club_points) < 2:
            return None
        
        point1 = np.array(self.club_points[0])
        point2 = np.array(self.club_points[1])
        direction = point2 - point1
        length = np.linalg.norm(direction)
        
        if length == 0:
            return None
        
        img_height, img_width = self.original_image.shape[:2]
        club_width = self._calculate_adaptive_width(img_width, img_height, length)
        
        result = {
            'mode': self.mode,
            'points': self.club_points[:2],
            'length': length,
            'angle': np.degrees(np.arctan2(direction[1], direction[0])),
            'club_width': club_width,
            'image_size': (img_width, img_height)
        }
        
        # 根据模式计算不同的几何信息
        if self.mode == 'line':
            result['center_point'] = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
        
        elif self.mode in ['bbox', 'rotated_bbox', 'polygon']:
            if self.mode == 'rotated_bbox' or self.mode == 'polygon':
                corners = self._calculate_rotated_bbox(point1, point2, club_width)
                result['rotated_corners'] = corners
                result['rotated_area'] = self._calculate_polygon_area(corners) if corners else 0
            
            # 计算轴对齐边界框（用于YOLO格式）
            bbox = self._calculate_axis_aligned_bbox(point1, point2, club_width, img_width, img_height)
            result['bbox'] = bbox
        
        return result
    
    def _save_annotation(self, image_path, annotation_data, output_dir):
        """保存标注结果"""
        if self.output_format == 'yolo':
            self._save_yolo_format(image_path, annotation_data, output_dir)
        elif self.output_format == 'coco':
            self._save_coco_format(image_path, annotation_data, output_dir)
    
    def _save_yolo_format(self, image_path, annotation_data, output_dir):
        """保存YOLO格式标注"""
        output_file = output_dir / f"{image_path.stem}.txt"
        
        if 'bbox' in annotation_data:
            x, y, w, h = annotation_data['bbox']
            img_width, img_height = annotation_data['image_size']
            
            # 转换为YOLO格式 (归一化的中心点坐标和宽高)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            # 类别ID (球杆为0)
            class_id = 0
            
            with open(output_file, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        # 同时保存详细信息到JSON文件（可选）
        json_file = output_dir / f"{image_path.stem}_detail.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
    
    def _save_coco_format(self, image_path, annotation_data, output_dir):
        """保存COCO格式标注"""
        # 这里可以实现COCO格式保存
        pass
    
    def _resize_for_display(self, image):
        """将图像缩放到适合显示的尺寸"""
        height, width = image.shape[:2]
        scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized, scale
        else:
            return image.copy(), 1.0
    
    def _scale_point_to_original(self, point):
        """将显示坐标转换为原始图像坐标"""
        x, y = point
        return (int(x / self.scale_factor), int(y / self.scale_factor))
    
    def _scale_point_to_display(self, point):
        """将原始图像坐标转换为显示坐标"""
        x, y = point
        return (int(x * self.scale_factor), int(y * self.scale_factor))
    
    def _calculate_adaptive_width(self, img_width, img_height, club_length):
        """计算自适应球杆宽度"""
        base_width = img_width * MIN_CLUB_WIDTH_RATIO
        max_width = img_width * MAX_CLUB_WIDTH_RATIO
        
        # 根据球杆长度调整
        if club_length < img_width * 0.1:
            width_factor = 0.8
        elif club_length < img_width * 0.3:
            width_factor = 1.0
        else:
            width_factor = 1.2
        
        adaptive_width = base_width * width_factor
        return max(min(adaptive_width, max_width), 3)
    
    def _calculate_rotated_bbox(self, point1, point2, width):
        """计算旋转边界框的四个角点"""
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        direction = point2 - point1
        length = np.linalg.norm(direction)
        
        if length == 0:
            return None
        
        direction_norm = direction / length
        perpendicular = np.array([-direction_norm[1], direction_norm[0]])
        half_width = width / 2
        
        corners = [
            point1 + perpendicular * half_width,  # 左上
            point2 + perpendicular * half_width,  # 右上
            point2 - perpendicular * half_width,  # 右下
            point1 - perpendicular * half_width   # 左下
        ]
        
        return [corner.tolist() for corner in corners]
    
    def _calculate_axis_aligned_bbox(self, point1, point2, width, img_width, img_height):
        """计算轴对齐边界框"""
        corners = self._calculate_rotated_bbox(point1, point2, width)
        if not corners:
            return None
        
        corners_array = np.array(corners)
        x_min = max(0, np.min(corners_array[:, 0]))
        x_max = min(img_width, np.max(corners_array[:, 0]))
        y_min = max(0, np.min(corners_array[:, 1]))
        y_max = min(img_height, np.max(corners_array[:, 1]))
        
        return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    
    def _calculate_polygon_area(self, corners):
        """计算多边形面积"""
        if not corners or len(corners) < 3:
            return 0
        
        corners_array = np.array(corners)
        x = corners_array[:, 0]
        y = corners_array[:, 1]
        
        return 0.5 * abs(sum(x[i] * y[(i + 1) % len(corners)] - x[(i + 1) % len(corners)] * y[i] 
                            for i in range(len(corners))))
    
    def _show_instructions(self):
        """显示总体操作说明"""
        print("\n" + "="*60)
        print("🎯 交互式标注操作说明")
        print("="*60)
        print("📝 标注方法:")
        print("  1. 用鼠标左键点击球杆的两个端点")
        print("  2. 第二个点击后自动完成当前图像标注")
        print("  3. 按ESC键跳过当前图像或提前结束标注")
        print("\n🎨 标注模式:")
        for mode, desc in ANNOTATION_MODES.items():
            marker = "👉" if mode == self.mode else "  "
            print(f"  {marker} {mode}: {desc}")
        print("\n📄 输出格式: " + self.output_format.upper())
        print("="*60)
    
    def _show_image_instructions(self, image_path):
        """显示当前图像的操作说明"""
        print(f"\n📸 正在标注: {image_path.name}")
        print("🖱️  左键点击球杆两端 | ESC键跳过图像")
        if self.scale_factor < 1.0:
            print(f"🔍 图像已缩放到 {self.scale_factor:.1%} 以适应屏幕")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高尔夫球杆检测交互式标注工具")
    parser.add_argument("--output_dir", "-o", help="输出标注目录（可选，默认在固定输入目录下创建annotations子目录）")
    parser.add_argument("--mode", "-m", choices=list(ANNOTATION_MODES.keys()), 
                       default="rotated_bbox", help="标注模式")
    parser.add_argument("--format", "-f", choices=["yolo", "coco"], 
                       default="yolo", help="输出格式")
    
    args = parser.parse_args()
    
    try:
        # 创建标注器
        annotator = InteractiveAnnotator(mode=args.mode, output_format=args.format)
        
        # 开始标注（使用固定输入目录）
        annotator.annotate_dataset(args.output_dir)
        
    except Exception as e:
        print(f"❌ 标注过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 