#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高尔夫球杆检测视频标注系统
从视频中提取帧并进行交互式标注，输出YOLO格式数据集
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import shutil

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

# 固定路径配置
VIDEO_INPUT_DIR = Path(r"C:\Users\Administrator\Desktop\AIGolf\videos")  # 视频输入目录
DATASET_OUTPUT_DIR = Path(r"C:\Users\Administrator\Desktop\AIGolf\dataset")  # 数据集输出目录（训练脚本的输入）

class VideoAnnotationSystem:
    """视频标注系统核心类"""
    
    def __init__(self, mode='rotated_bbox', frame_interval=10, max_frames_per_video=50):
        """
        初始化视频标注系统
        
        Args:
            mode: 标注模式 ('bbox', 'rotated_bbox', 'line', 'polygon')
            frame_interval: 帧间隔（每隔多少帧提取一帧）
            max_frames_per_video: 每个视频最大提取帧数
        """
        self.mode = mode
        self.frame_interval = frame_interval
        self.max_frames_per_video = max_frames_per_video
        
        # 路径配置
        self.video_input_dir = VIDEO_INPUT_DIR
        self.dataset_output_dir = DATASET_OUTPUT_DIR
        
        # 确保输出目录存在
        self._ensure_output_directories()
        
        # 标注状态
        self.current_image = None
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        
        # 标注数据
        self.club_points = []
        self.current_frame_info = None
        
        # 控制标志
        self.drawing = False
        self.finish_annotation = False
        self.skip_image = False
        
        print(f"🎯 视频标注系统已初始化")
        print(f"📝 标注模式: {ANNOTATION_MODES.get(mode, mode)}")
        print(f"📁 视频输入目录: {self.video_input_dir.absolute()}")
        print(f"📁 数据集输出目录: {self.dataset_output_dir.absolute()}")
        print(f"⚙️ 帧间隔: {frame_interval}, 最大帧数: {max_frames_per_video}")
    
    def _ensure_output_directories(self):
        """确保输出目录结构存在"""
        directories = [
            self.dataset_output_dir / "images",
            self.dataset_output_dir / "annotations",
            self.dataset_output_dir / "processed_videos"  # 记录已处理的视频
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_videos(self):
        """处理视频目录中的所有视频文件"""
        if not self.video_input_dir.exists():
            print(f"❌ 视频输入目录不存在: {self.video_input_dir}")
            print("请确保目录存在并包含视频文件")
            return
        
        # 获取所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(self.video_input_dir.glob(f"*{ext}")))
            video_files.extend(list(self.video_input_dir.glob(f"*{ext.upper()}")))
        
        if not video_files:
            print(f"❌ 在目录 {self.video_input_dir} 中未找到视频文件")
            print("支持的视频格式: .mp4, .avi, .mov, .mkv, .wmv")
            return
        
        print(f"📁 找到 {len(video_files)} 个视频文件")
        
        # 检查已处理的视频
        processed_videos = self._get_processed_videos()
        
        total_frames_extracted = 0
        total_frames_annotated = 0
        
        for i, video_path in enumerate(video_files):
            print(f"\n📹 处理视频 {i+1}/{len(video_files)}: {video_path.name}")
            
            # 检查是否已处理过
            video_hash = self._calculate_video_hash(video_path)
            if video_hash in processed_videos:
                print(f"⏭️ 视频已处理过，跳过: {video_path.name}")
                continue
            
            # 提取帧并标注
            extracted, annotated = self._process_single_video(video_path)
            total_frames_extracted += extracted
            total_frames_annotated += annotated
            
            # 记录已处理的视频
            self._mark_video_as_processed(video_path, video_hash, extracted, annotated)
        
        print(f"\n🎉 视频处理完成!")
        print(f"📊 总计提取帧数: {total_frames_extracted}")
        print(f"📊 总计标注帧数: {total_frames_annotated}")
        print(f"📁 数据集保存在: {self.dataset_output_dir}")
    
    def _process_single_video(self, video_path):
        """处理单个视频文件"""
        print(f"🔄 正在处理视频: {video_path.name}")
        
        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return 0, 0
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 视频信息: {total_frames} 帧, {fps:.2f} FPS")
        
        # 计算要提取的帧
        frame_indices = self._calculate_frame_indices(total_frames)
        print(f"📋 将提取 {len(frame_indices)} 帧进行标注")
        
        extracted_count = 0
        annotated_count = 0
        
        # 显示操作说明
        self._show_video_instructions(video_path)
        
        for frame_idx in frame_indices:
            # 跳转到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️ 无法读取第 {frame_idx} 帧")
                continue
            
            extracted_count += 1
            
            # 准备帧信息
            self.current_frame_info = {
                'video_name': video_path.stem,
                'frame_index': frame_idx,
                'timestamp': frame_idx / fps if fps > 0 else 0,
                'video_path': str(video_path)
            }
            
            # 标注当前帧
            result = self._annotate_frame(frame, frame_idx, len(frame_indices))
            
            if result:
                # 保存帧和标注
                self._save_frame_and_annotation(frame, result)
                annotated_count += 1
                print(f"✅ 已标注帧 {frame_idx}")
            else:
                print(f"⏭️ 跳过帧 {frame_idx}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ 视频处理完成: 提取 {extracted_count} 帧, 标注 {annotated_count} 帧")
        return extracted_count, annotated_count
    
    def _calculate_frame_indices(self, total_frames):
        """计算要提取的帧索引"""
        # 根据帧间隔计算
        frame_indices = list(range(0, total_frames, self.frame_interval))
        
        # 限制最大帧数
        if len(frame_indices) > self.max_frames_per_video:
            # 均匀分布选择帧
            step = len(frame_indices) / self.max_frames_per_video
            frame_indices = [frame_indices[int(i * step)] for i in range(self.max_frames_per_video)]
        
        return frame_indices
    
    def _annotate_frame(self, frame, frame_idx, total_frames):
        """标注单个帧"""
        # 重置标注状态
        self.club_points = []
        self.finish_annotation = False
        self.skip_image = False
        
        # 设置当前图像
        self.original_image = frame.copy()
        self.display_image, self.scale_factor = self._resize_for_display(frame)
        self.current_image = self.display_image.copy()
        
        # 创建窗口和设置鼠标回调
        window_name = f'标注帧 {frame_idx} ({total_frames} 帧总计)'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # 显示当前帧信息
        self._show_frame_info(frame_idx, total_frames)
        
        # 标注循环
        while True:
            cv2.imshow(window_name, self.display_image)
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
        
        cv2.destroyWindow(window_name)
        
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
            f"视频: {self.current_frame_info['video_name']}",
            f"帧: {self.current_frame_info['frame_index']}",
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
            'image_size': (img_width, img_height),
            'frame_info': self.current_frame_info.copy()
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
    
    def _save_frame_and_annotation(self, frame, annotation_data):
        """保存帧图像和标注数据"""
        # 生成唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        video_name = self.current_frame_info['video_name']
        frame_idx = self.current_frame_info['frame_index']
        
        filename = f"{video_name}_frame_{frame_idx:06d}_{timestamp}"
        
        # 保存图像
        image_path = self.dataset_output_dir / "images" / f"{filename}.jpg"
        cv2.imwrite(str(image_path), frame)
        
        # 保存YOLO格式标注
        if 'bbox' in annotation_data:
            self._save_yolo_annotation(filename, annotation_data)
        
        # 保存详细标注信息（JSON格式）
        self._save_detailed_annotation(filename, annotation_data)
    
    def _save_yolo_annotation(self, filename, annotation_data):
        """保存YOLO格式标注"""
        annotation_path = self.dataset_output_dir / "annotations" / f"{filename}.txt"
        
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
            
            with open(annotation_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    def _save_detailed_annotation(self, filename, annotation_data):
        """保存详细标注信息"""
        detail_path = self.dataset_output_dir / "annotations" / f"{filename}_detail.json"
        
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
    
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
    
    def _calculate_video_hash(self, video_path):
        """计算视频文件的哈希值"""
        hash_md5 = hashlib.md5()
        with open(video_path, "rb") as f:
            # 只读取文件的开头和结尾部分来计算哈希（提高速度）
            chunk = f.read(8192)
            hash_md5.update(chunk)
            
            # 跳到文件中间
            f.seek(f.seek(0, 2) // 2)  # 文件大小的一半
            chunk = f.read(8192)
            hash_md5.update(chunk)
            
            # 跳到文件末尾
            f.seek(-8192, 2)
            chunk = f.read(8192)
            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_processed_videos(self):
        """获取已处理的视频列表"""
        processed_file = self.dataset_output_dir / "processed_videos" / "processed_list.json"
        
        if processed_file.exists():
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('processed_hashes', []))
        
        return set()
    
    def _mark_video_as_processed(self, video_path, video_hash, extracted_count, annotated_count):
        """标记视频为已处理"""
        processed_file = self.dataset_output_dir / "processed_videos" / "processed_list.json"
        
        # 读取现有数据
        if processed_file.exists():
            with open(processed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'processed_videos': [], 'processed_hashes': []}
        
        # 添加新记录
        video_record = {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'video_hash': video_hash,
            'processed_time': datetime.now().isoformat(),
            'extracted_frames': extracted_count,
            'annotated_frames': annotated_count,
            'annotation_mode': self.mode
        }
        
        data['processed_videos'].append(video_record)
        data['processed_hashes'].append(video_hash)
        
        # 保存更新后的数据
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _show_video_instructions(self, video_path):
        """显示视频处理说明"""
        print(f"\n📹 开始处理视频: {video_path.name}")
        print("🖱️  操作说明:")
        print("   - 左键点击球杆两端进行标注")
        print("   - ESC键跳过当前帧")
        print("   - 标注完成后自动进入下一帧")
    
    def _show_frame_info(self, frame_idx, total_frames):
        """显示当前帧信息"""
        print(f"📸 标注帧 {frame_idx} (共 {total_frames} 帧)")
        print(f"🎬 视频: {self.current_frame_info['video_name']}")
        print(f"⏱️  时间戳: {self.current_frame_info['timestamp']:.2f}s")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高尔夫球杆检测视频标注系统")
    parser.add_argument("--mode", "-m", choices=list(ANNOTATION_MODES.keys()), 
                       default="rotated_bbox", help="标注模式")
    parser.add_argument("--frame_interval", "-i", type=int, default=10, 
                       help="帧间隔（每隔多少帧提取一帧）")
    parser.add_argument("--max_frames", "-f", type=int, default=50, 
                       help="每个视频最大提取帧数")
    
    args = parser.parse_args()
    
    try:
        # 创建视频标注系统
        system = VideoAnnotationSystem(
            mode=args.mode,
            frame_interval=args.frame_interval,
            max_frames_per_video=args.max_frames
        )
        
        # 开始处理视频
        system.process_videos()
        
    except Exception as e:
        print(f"❌ 视频标注过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 