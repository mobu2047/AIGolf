# keypoint_3d_processor.py
# ---------------------------------------------------
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple

class Keypoint3DProcessor:
    """处理3D姿态估计的类，从2D姿态关键点生成3D姿态和相关特征"""
    
    @staticmethod
    def estimate_3d_from_2d(keypoints_2d: np.ndarray, image_size: Tuple[int, int] = None) -> np.ndarray:
        """
        使用简化的算法从2D关键点估计3D姿态
        
        Args:
            keypoints_2d: 形状为(33, 2)的2D关键点数组
            image_size: 原始图像尺寸(高度, 宽度)，用于归一化，如果已经归一化则为None
            
        Returns:
            形状为(33, 3)的3D关键点数组
        """
        # 如果提供了图像尺寸，则归一化2D关键点
        if image_size:
            height, width = image_size
            normalized_kps = keypoints_2d.copy()
            normalized_kps[:, 0] /= width
            normalized_kps[:, 1] /= height
        else:
            normalized_kps = keypoints_2d
        
        # 这是一个简化的3D估计方法，实际应用中应使用更复杂的技术
        keypoints_3d = np.zeros((normalized_kps.shape[0], 3), dtype=np.float32)
        keypoints_3d[:, :2] = normalized_kps  # 复制x和y坐标
        
        # 基于人体骨骼知识估计z坐标
        # 使用中心点作为参考
        hip_center = (normalized_kps[23] + normalized_kps[24]) / 2  # 髋部中心
        shoulder_center = (normalized_kps[11] + normalized_kps[12]) / 2  # 肩部中心
        
        # 计算躯干长度作为深度缩放因子
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        # 根据关节位置估计相对深度
        for i in range(normalized_kps.shape[0]):
            # 计算到中心的距离，用作深度估计的依据
            dist_to_hip = np.linalg.norm(normalized_kps[i] - hip_center)
            
            # 为不同关节分配不同的深度值
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # 头部和脸部
                keypoints_3d[i, 2] = 0.2 * torso_length
            elif i in [11, 12]:  # 肩膀
                keypoints_3d[i, 2] = 0.1 * torso_length
            elif i in [13, 14]:  # 肘部
                keypoints_3d[i, 2] = 0.0
            elif i in [15, 16, 17, 18, 19, 20, 21, 22]:  # 手部
                keypoints_3d[i, 2] = -0.1 * torso_length
            elif i in [23, 24]:  # 髋部
                keypoints_3d[i, 2] = 0.0
            elif i in [25, 26]:  # 膝盖
                keypoints_3d[i, 2] = -0.1 * torso_length
            else:  # 脚部
                keypoints_3d[i, 2] = -0.2 * torso_length
                
        return keypoints_3d
    
    @staticmethod
    def compute_joint_velocities(keypoints_sequence: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """
        计算关节速度
        
        Args:
            keypoints_sequence: 形状为(frames, 33, 3)的3D关键点序列
            fps: 视频帧率
            
        Returns:
            形状为(frames-1, 33, 3)的速度数组
        """
        time_delta = 1.0 / fps
        velocities = np.diff(keypoints_sequence, axis=0) / time_delta
        return velocities
    
    @staticmethod
    def compute_joint_accelerations(velocities: np.ndarray, fps: float = 30.0) -> np.ndarray:
        """
        计算关节加速度
        
        Args:
            velocities: 形状为(frames, 33, 3)的速度数组
            fps: 视频帧率
            
        Returns:
            形状为(frames-1, 33, 3)的加速度数组
        """
        time_delta = 1.0 / fps
        accelerations = np.diff(velocities, axis=0) / time_delta
        return accelerations
    
    @staticmethod
    def compute_angular_velocities(keypoints_sequence: np.ndarray, angle_pairs: Dict[str, Tuple[int, int, int]], fps: float = 30.0) -> Dict[str, np.ndarray]:
        """
        计算关节角速度
        
        Args:
            keypoints_sequence: 形状为(frames, 33, 3)的3D关键点序列
            angle_pairs: 角度对字典，键为关节名称，值为三个关键点索引
            fps: 视频帧率
            
        Returns:
            关节角速度字典，键为关节名称，值为角速度数组
        """
        frames = keypoints_sequence.shape[0]
        angles = {}
        
        # 计算每一帧的关节角度
        for joint_name, (idx1, idx2, idx3) in angle_pairs.items():
            joint_angles = np.zeros(frames)
            for i in range(frames):
                p1 = keypoints_sequence[i, idx1]
                p2 = keypoints_sequence[i, idx2]
                p3 = keypoints_sequence[i, idx3]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                # 计算两个向量的夹角
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
                joint_angles[i] = np.degrees(np.arccos(cos_angle))
            
            angles[joint_name] = joint_angles
        
        # 计算角速度
        angular_velocities = {}
        for joint_name, joint_angles in angles.items():
            angular_velocities[joint_name] = np.diff(joint_angles) / (1.0 / fps)
            
        return angular_velocities
    
    @staticmethod
    def extract_dynamic_features(keypoints_2d_sequence: np.ndarray, fps: float = 30.0) -> Dict[str, np.ndarray]:
        """
        从2D关键点序列提取动态特征
        
        Args:
            keypoints_2d_sequence: 形状为(frames, 33, 2)的2D关键点序列
            fps: 视频帧率
            
        Returns:
            包含各种动态特征的字典
        """
        frames = keypoints_2d_sequence.shape[0]
        
        # 推算3D姿态
        keypoints_3d_sequence = np.zeros((frames, 33, 3), dtype=np.float32)
        for i in range(frames):
            keypoints_3d_sequence[i] = Keypoint3DProcessor.estimate_3d_from_2d(keypoints_2d_sequence[i])
        
        # 计算速度、加速度和角速度
        velocities = Keypoint3DProcessor.compute_joint_velocities(keypoints_3d_sequence, fps)
        accelerations = Keypoint3DProcessor.compute_joint_accelerations(velocities, fps)
        
        # 定义关节角度对
        from error_detection import angle_pairs
        angular_velocities = Keypoint3DProcessor.compute_angular_velocities(keypoints_3d_sequence, angle_pairs, fps)
        
        # 计算速度和加速度大小（模长）
        velocity_magnitudes = np.linalg.norm(velocities, axis=2)
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=2)
        
        # 返回所有动态特征
        return {
            "keypoints_3d": keypoints_3d_sequence,
            "velocities": velocities,
            "accelerations": accelerations,
            "velocity_magnitudes": velocity_magnitudes,
            "acceleration_magnitudes": acceleration_magnitudes,
            "angular_velocities": angular_velocities
        } 