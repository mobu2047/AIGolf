# keypoint_processor.py
# -------------------
# 存放 KeypointProcessor 类。

import math
import torch
import numpy as np
from typing import Dict
from utils import angle_difference  # 若需 angle_difference, ...
# 也可不 import, 看情况

class KeypointProcessor:
    """关键点处理工具类"""

    NOSE = 0
    NECK = 1
    R_SHOULDER = 12
    L_SHOULDER = 11
    R_HIP = 24
    L_HIP = 23
    R_ELBOW = 14
    L_ELBOW = 13
    R_WRIST = 16
    L_WRIST = 15
    R_KNEE = 26
    L_KNEE = 25
    R_ANKLE = 28
    L_ANKLE = 27

    @staticmethod
    def get_region_masks() -> Dict[str, np.ndarray]:
        return {
            'shoulders': np.array([11, 12, 13, 14]),
            'arms': np.array([13, 14, 15, 16]),
            'hips': np.array([23, 24]),
            'legs': np.array([25, 26, 27, 28, 29, 30, 31, 32])
        }

    @staticmethod
    def preprocess(frame: torch.Tensor) -> torch.Tensor:
        """
        改进的归一化处理（以髋部中心为基准 + 肩宽），
        并对人物左右进行旋转对齐。
        """
        L_HIP, R_HIP = KeypointProcessor.L_HIP, KeypointProcessor.R_HIP
        hip_center = (frame[L_HIP, :2] + frame[R_HIP, :2]) / 2.0
        neck = frame[KeypointProcessor.NECK, :2]

        shoulder_width = torch.norm(
            frame[KeypointProcessor.R_SHOULDER, :2] -
            frame[KeypointProcessor.L_SHOULDER, :2]
        )
        torso_length = torch.norm(neck - hip_center) + 1e-8
        scale_factor = (torso_length + shoulder_width) / 2

        normalized_points = (frame[:, :2] - hip_center) / scale_factor

        # 人物左右髋对齐
        dx = frame[R_HIP, 0].item() - frame[L_HIP, 0].item()
        dy = frame[R_HIP, 1].item() - frame[L_HIP, 1].item()
        rotation_angle = math.atan2(dy, dx)
        rot_matrix = torch.tensor([
            [ math.cos(-rotation_angle), -math.sin(-rotation_angle)],
            [ math.sin(-rotation_angle),  math.cos(-rotation_angle)]
        ])
        rotated_points = normalized_points @ rot_matrix.T

        return rotated_points
