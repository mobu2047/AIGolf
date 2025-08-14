#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高尔夫球杆检测器模块
用于检测视频中的球杆位置并转换为关键点格式
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GolfClubDetector:
    """高尔夫球杆检测器"""
    
    def __init__(self, model_path=None, confidence=0.2):
        """
        初始化球杆检测器
        
        Args:
            model_path: YOLO模型路径，如果为None则使用默认路径
            confidence: 检测置信度阈值
        """
        if model_path is None:
            model_path = "yolo_dataset_full/runs/detect/golf_club_yolo/weights/best.pt"
        
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        
        # 检查模型文件是否存在
        if not Path(model_path).exists():
            logger.warning(f"YOLO模型文件不存在: {model_path}")
            return
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"成功加载YOLO模型: {model_path}")
        except Exception as e:
            logger.error(f"加载YOLO模型失败: {e}")
            self.model = None
    
    def detect_in_frame(self, frame):
        """
        在单帧中检测球杆
        
        Args:
            frame: 输入图像帧
            
        Returns:
            dict: 检测结果 {'bbox': [x1, y1, x2, y2], 'confidence': float}
            如果未检测到则返回None
        """
        if self.model is None:
            return None
        
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # 选择置信度最高的检测
                    best_idx = boxes.conf.argmax()
                    bbox = boxes.xyxy[best_idx].cpu().numpy()
                    conf = boxes.conf[best_idx].cpu().numpy()
                    
                    return {
                        'bbox': bbox,  # [x1, y1, x2, y2]
                        'confidence': float(conf)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"球杆检测失败: {e}")
            return None
    
    def bbox_to_endpoints(self, bbox, frame_shape):
        """
        将边界框转换为球杆端点
        
        Args:
            bbox: [x1, y1, x2, y2] 边界框坐标
            frame_shape: (height, width) 图像尺寸
            
        Returns:
            tuple: (grip_point, head_point) 球杆握把端和球杆头端的归一化坐标
        """
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # 计算边界框中心和尺寸
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # 判断球杆方向（水平还是垂直）
        if bbox_width > bbox_height:
            # 水平方向的球杆
            grip_point = (x1, center_y)  # 左端为握把
            head_point = (x2, center_y)  # 右端为球杆头
        else:
            # 垂直方向的球杆
            grip_point = (center_x, y1)  # 上端为握把
            head_point = (center_x, y2)  # 下端为球杆头
        
        # 转换为归一化坐标 [0, 1]
        grip_normalized = (grip_point[0] / width, grip_point[1] / height)
        head_normalized = (head_point[0] / width, head_point[1] / height)
        
        return grip_normalized, head_normalized
    
    def detect_in_video(self, video_path):
        """
        检测视频中的球杆，返回关键点格式数据
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            np.ndarray: 球杆关键点数据，形状为 (帧数, 2, 3)
            每帧包含2个点：[握把端, 球杆头端]，每个点3个值：[x, y, visibility]
        """
        if self.model is None:
            logger.warning("YOLO模型未加载，返回空的球杆关键点数据")
            return np.array([])
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return np.array([])
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        golf_club_keypoints = []
        
        logger.info(f"开始检测视频中的球杆: {video_path}")
        logger.info(f"总帧数: {total_frames}")
        
        frame_count = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 检测当前帧中的球杆
                detection = self.detect_in_frame(frame)
                
                if detection is not None:
                    # 转换为端点坐标
                    grip_point, head_point = self.bbox_to_endpoints(
                        detection['bbox'], frame.shape
                    )
                    
                    # 构造关键点数据：[x, y, visibility]
                    # visibility使用检测置信度
                    visibility = detection['confidence']
                    
                    frame_keypoints = np.array([
                        [grip_point[0], grip_point[1], visibility],  # 球杆握把端
                        [head_point[0], head_point[1], visibility]   # 球杆头端
                    ])
                    
                    detection_count += 1
                else:
                    # 未检测到球杆，设置为不可见
                    frame_keypoints = np.array([
                        [0.0, 0.0, 0.0],  # 球杆握把端（不可见）
                        [0.0, 0.0, 0.0]   # 球杆头端（不可见）
                    ])
                
                golf_club_keypoints.append(frame_keypoints)
                
                # 显示进度
                if frame_count % 10 == 0:
                    logger.info(f"处理进度: {frame_count}/{total_frames} 帧，检测到球杆: {detection_count} 次")
        
        finally:
            cap.release()
        
        golf_club_keypoints = np.array(golf_club_keypoints)
        
        logger.info(f"球杆检测完成!")
        logger.info(f"  处理帧数: {frame_count}")
        logger.info(f"  检测成功: {detection_count} 次")
        logger.info(f"  检测率: {detection_count/frame_count*100:.1f}%")
        logger.info(f"  输出形状: {golf_club_keypoints.shape}")
        
        return golf_club_keypoints

def add_golf_club_to_keypoints(existing_keypoints, video_path, model_path=None, confidence=0.2):
    """
    在现有关键点数据基础上添加球杆检测信息
    
    Args:
        existing_keypoints: 现有关键点数据，形状为 (帧数, 33, 3)
        video_path: 视频文件路径
        model_path: YOLO模型路径
        confidence: 检测置信度阈值
        
    Returns:
        np.ndarray: 增强后的关键点数据，形状为 (帧数, 35, 3)
        前33个点为人体关键点，后2个点为球杆关键点
    """
    logger.info("开始添加球杆检测信息到关键点数据")
    
    try:
        # 创建球杆检测器
        detector = GolfClubDetector(model_path, confidence)
        
        # 检测视频中的球杆
        golf_club_keypoints = detector.detect_in_video(video_path)
        
        # 检查是否检测成功
        if golf_club_keypoints.size == 0:
            logger.warning("球杆检测失败，返回原始关键点数据")
            return existing_keypoints
        
        # 检查帧数是否匹配
        if golf_club_keypoints.shape[0] != existing_keypoints.shape[0]:
            logger.warning(f"帧数不匹配: 关键点{existing_keypoints.shape[0]}帧, 球杆检测{golf_club_keypoints.shape[0]}帧")
            # 取较小的帧数
            min_frames = min(golf_club_keypoints.shape[0], existing_keypoints.shape[0])
            existing_keypoints = existing_keypoints[:min_frames]
            golf_club_keypoints = golf_club_keypoints[:min_frames]
        
        # 合并关键点数据
        enhanced_keypoints = np.concatenate([existing_keypoints, golf_club_keypoints], axis=1)
        
        logger.info(f"成功添加球杆检测信息:")
        logger.info(f"  原始关键点形状: {existing_keypoints.shape}")
        logger.info(f"  球杆关键点形状: {golf_club_keypoints.shape}")
        logger.info(f"  增强后形状: {enhanced_keypoints.shape}")
        
        return enhanced_keypoints
        
    except Exception as e:
        logger.error(f"添加球杆检测信息时出错: {e}")
        logger.warning("返回原始关键点数据以确保系统正常运行")
        return existing_keypoints

# 关键点索引定义
KEYPOINT_CONFIG = {
    'pose_points': 33,           # MediaPipe人体关键点数量
    'golf_club_points': 2,       # 球杆关键点数量
    'total_points': 35,          # 总关键点数量
    'golf_club_indices': [33, 34],  # 球杆关键点索引
    'golf_club_names': ['golf_club_grip', 'golf_club_head']  # 球杆关键点名称
}

def get_golf_club_keypoints(enhanced_keypoints):
    """
    从增强的关键点数据中提取球杆关键点
    
    Args:
        enhanced_keypoints: 增强后的关键点数据，形状为 (帧数, 35, 3)
        
    Returns:
        np.ndarray: 球杆关键点数据，形状为 (帧数, 2, 3)
    """
    if enhanced_keypoints.shape[1] < KEYPOINT_CONFIG['total_points']:
        logger.warning("关键点数据不包含球杆信息")
        return np.array([])
    
    return enhanced_keypoints[:, KEYPOINT_CONFIG['golf_club_indices'], :]

if __name__ == "__main__":
    # 测试代码
    detector = GolfClubDetector()
    test_video = "test/labeled/1.mp4"
    
    if Path(test_video).exists():
        golf_keypoints = detector.detect_in_video(test_video)
        print(f"测试完成，球杆关键点形状: {golf_keypoints.shape}")
    else:
        print(f"测试视频不存在: {test_video}") 