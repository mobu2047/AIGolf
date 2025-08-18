# assistant_transparent_lines.py
# ---------------------------------------------------------
# 实现需求：
#   生成透明背景的辅助线PNG图片
#   1) 从 stage_indices 和 keypoint_data 中提取信息
#   2) 生成仅包含辅助线的透明PNG图片
#   3) 输出到 resultData/img/站姿/ 目录
#   4) 模块化设计，方便后续添加更多辅助线类型
# ---------------------------------------------------------

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image  # 使用PIL来保存透明PNG
import matplotlib.pyplot as plt
from utils import calculate_line_thickness
from multiprocessing import Pool
import sys
import json
import glob
import platform  # 添加platform模块导入
from typing import Dict, List, Tuple
import gc

# 并行处理配置
ENABLE_PARALLEL = True  # 并行处理开关
MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1  # 自动设置进程数

# 添加内存优化配置
USE_MEMORY_OPTIMIZATION = True  # 内存优化开关
PREALLOC_BUFFER_SIZE = (1920, 1080, 4)  # 预分配缓冲区大小(高, 宽, 通道数)

# 添加图像处理优化
USE_IMAGE_CACHE = True  # 图像缓存开关
IMAGE_CACHE_SIZE = 100  # 图像缓存大小（帧数）
BATCH_SAVE_SIZE = 20    # 批量保存大小

# 添加进程通信优化配置
USE_SHARED_MEMORY = True    # 共享内存开关
USE_PICKLE_PROTOCOL_HIGHEST = True  # 使用最高级别的Pickle协议
MAX_SHARED_MEMORY_SIZE = 100 * 1024 * 1024  # 100MB

def set_parallel_enabled(enabled: bool):
    """设置是否启用并行处理"""
    global ENABLE_PARALLEL
    ENABLE_PARALLEL = enabled

def set_memory_optimization(enabled: bool):
    """设置是否启用内存优化"""
    global USE_MEMORY_OPTIMIZATION
    USE_MEMORY_OPTIMIZATION = enabled

def set_image_cache_enabled(enabled: bool):
    """设置是否启用图像缓存"""
    global USE_IMAGE_CACHE
    USE_IMAGE_CACHE = enabled

def set_shared_memory_enabled(enabled: bool):
    """设置是否启用共享内存"""
    global USE_SHARED_MEMORY
    USE_SHARED_MEMORY = enabled

class AssistantLineGenerator:
    """
    辅助线生成器基类，提供通用方法和接口
    """
    def __init__(self, line_color=(0, 0, 255), line_thickness=9, video_width=3840, video_height=2160):
        """
        初始化辅助线生成器
        
        参数:
            line_color: BGR颜色元组，默认为红色(0,0,255)
            line_thickness: 线条粗细，默认为9
            video_width: 视频宽度，默认为3840(4K)
            video_height: 视频高度，默认为2160(4K)
        """
        self.line_color = line_color
        self.line_thickness = calculate_line_thickness(line_thickness, video_width, video_height)
        
    def calculate_lines_info(self, frame_size, keypoint):
        """
        计算辅助线信息，需要子类实现
        
        参数:
            frame_size: (height, width)元组
            keypoint: 关键点数据，(33,2)张量
            
        返回:
            字典，包含辅助线绘制所需信息
        """
        raise NotImplementedError("子类需要实现此方法")
    
    def draw_lines(self, canvas, lines_info):
        """
        在画布上绘制辅助线，需要子类实现
        
        参数:
            canvas: 要绘制的画布
            lines_info: 由calculate_lines_info计算出的辅助线信息
        """
        raise NotImplementedError("子类需要实现此方法")
    
    def generate_transparent_image(self, frame_size, lines_info):
        """
        生成透明背景的辅助线图片
        
        参数:
            frame_size: (height, width)元组
            lines_info: 辅助线信息
            
        返回:
            包含辅助线的透明背景图片
        """
        # 使用缓冲区池获取预分配的画布，而不是每次创建新的
        canvas = buffer_pool.get_buffer((frame_size[0], frame_size[1], 4), dtype=np.uint8)
        
        # 确保画布是干净的（全透明）
        canvas.fill(0)
        
        # 在透明画布上绘制辅助线
        self.draw_lines(canvas, lines_info)
        
        return canvas


class StanceAssistantLine(AssistantLineGenerator):
    """
    站姿辅助线生成器，第一种辅助线
    """
    def __init__(self, line_color=(0, 0, 255), line_thickness=6, video_width=1920, video_height=1080):
        """初始化为红色线条，粗细为6"""
        super().__init__(line_color, line_thickness, video_width, video_height)

    def calculate_lines_info(self, frame_size, kpt_33x2):
        """
        计算站姿辅助线信息
        
        参数:
            frame_size: (height, width)元组
            kpt_33x2: 关键点数据，(33,2)张量
            
        返回:
            字典，包含站姿辅助线绘制所需信息
        """
        h, w = frame_size
        # 将相对坐标 -> 像素坐标
        coords = kpt_33x2.clone()
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.int().numpy()  # => (33,2)

        # 头部圆环(增大)
        head_indices = [0, 7, 8, 9, 10]
        hx_list = [coords[i][0] for i in head_indices]
        hy_list = [coords[i][1] for i in head_indices]
        min_x, max_x = min(hx_list), max(hx_list)
        min_y, max_y = min(hy_list), max(hy_list)
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        diag = int(np.hypot(max_x - min_x, max_y - min_y))
        radius = (diag // 2) + 40

        # 肩膀连线(1.3倍延长)
        left_shoulder  = (coords[11][0], coords[11][1])
        right_shoulder = (coords[12][0], coords[12][1])
        extended_shoulder_line = self.compute_extended_line_points(
            left_shoulder, right_shoulder, factor=1.3
        )

        # 肩膀->脚踝
        left_ankle  = (coords[27][0], coords[27][1])
        right_ankle = (coords[28][0], coords[28][1])

        # 肩膀->手
        left_hand  = (coords[15][0], coords[15][1])
        right_hand = (coords[16][0], coords[16][1])

        # 鼻子->双手中点 => 延长
        nose = (coords[0][0], coords[0][1])
        hands_intersect = ((left_hand[0] + right_hand[0]) // 2,
                          (left_hand[1] + right_hand[1]) // 2)
        nose2hands_line = self.compute_extended_line_points(nose, hands_intersect, factor=2.0)

        return {
            "head_center": (center_x, center_y),
            "head_radius": radius,
            "shoulder_line_points": extended_shoulder_line,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_ankle": left_ankle,
            "right_ankle": right_ankle,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "nose2hands_line": nose2hands_line
        }
    
    def draw_lines(self, canvas, lines_info):
        """
        在画布上绘制站姿辅助线
        
        参数:
            canvas: 要绘制的画布
            lines_info: 辅助线信息
        """
        # 线条颜色，带有Alpha通道(完全不透明)
        color = (*self.line_color, 255)  # (B,G,R,A)
        thickness = self.line_thickness
        
        # 1) 头部圆环
        cv2.circle(canvas, lines_info["head_center"], lines_info["head_radius"], color, thickness)

        # 2) 肩膀连线(延长)
        p1, p2 = lines_info["shoulder_line_points"]
        cv2.line(canvas, p1, p2, color, thickness)

        # 3) 肩膀->脚踝
        cv2.line(canvas, lines_info["left_shoulder"], lines_info["left_ankle"], color, thickness)
        cv2.line(canvas, lines_info["right_shoulder"], lines_info["right_ankle"], color, thickness)

        # 4) 肩膀->手
        cv2.line(canvas, lines_info["left_shoulder"], lines_info["left_hand"], color, thickness)
        cv2.line(canvas, lines_info["right_shoulder"], lines_info["right_hand"], color, thickness)

        # 5) 鼻子->双手中点(延长)
        cv2.line(canvas, lines_info["nose2hands_line"][0], lines_info["nose2hands_line"][1], color, thickness)
    
    def compute_extended_line_points(self, pt1, pt2, factor=2.0):
        """
        计算延长线的端点
        
        参数:
            pt1, pt2: 原始线段的两个端点
            factor: 延长倍数
            
        返回:
            延长后的线段端点
        """
        x1, y1 = pt1
        x2, y2 = pt2
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dx = (x2 - x1) / 2
        dy = (y2 - y1) / 2
        half_len = np.hypot(dx, dy)
        if half_len < 1e-5:
            return pt1, pt2

        new_half_len = half_len * factor
        ux = dx / half_len
        uy = dy / half_len
        x1_new = int(cx - ux * new_half_len)
        y1_new = int(cy - uy * new_half_len)
        x2_new = int(cx + ux * new_half_len)
        y2_new = int(cy + uy * new_half_len)

        return (x1_new, y1_new), (x2_new, y2_new)


class HeightAssistantLine(AssistantLineGenerator):
    """
    头顶高度线生成器，K线图（第二种辅助线）
    在人物头顶显示一条水平白线，用于展示人物高度
    """
    def __init__(self, line_color=(255, 255, 255), line_thickness=4, video_width=1920, video_height=1080):
        """初始化为白色线条，粗细为4"""
        super().__init__(line_color, line_thickness, video_width, video_height)
        
    def calculate_lines_info(self, frame_size, kpt_33x2):
        """
        计算头顶高度线信息，参考assistant_lines2.py中的estimate_head_top函数
        
        参数:
            frame_size: (height, width)元组
            kpt_33x2: 关键点数据，(33,2)张量
            
        返回:
            字典，包含头顶高度线绘制所需信息
        """
        h, w = frame_size
        # 将相对坐标 -> 像素坐标
        coords = kpt_33x2.clone()
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.int().numpy()  # => (33,2)

        # 使用面部关键点索引（Mediapipe Pose 关键点：0~8）
        face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 面部关键点
        face_points = coords[face_indices]  # 提取面部关键点坐标
        
        # 过滤掉无效的点（坐标为0的点）
        valid_face_points = []
        for point in face_points:
            if point[0] > 0 and point[1] > 0:
                valid_face_points.append(point)
        
        if valid_face_points:
            # 转换为numpy数组以便使用numpy函数
            valid_face_points = np.array(valid_face_points)
            
            # 计算面部边界
            min_y = np.min(valid_face_points[:, 1])
            max_y = np.max(valid_face_points[:, 1])
            face_height = max_y - min_y
            
            # 横向中心：取面部关键点的平均x值
            face_center_x = int(np.mean(valid_face_points[:, 0]))
            
            # 根据经验，将头顶估计为：最小y值向上延伸2.3倍face_height
            # 这个系数参考自assistant_lines2.py中的estimate_head_top函数
            offset = 2.3 * face_height
            head_top_y = int(min_y - offset)
            
            # 防止越出画面
            head_top_y = max(0, head_top_y)
            
            print(f"[DEBUG] 面部高度: {face_height:.1f}, 偏移量: {offset:.1f}")
            print(f"[DEBUG] 最高点y值: {min_y}, 计算的头顶位置y值: {head_top_y}")
        else:
            # 如果没有有效的面部关键点，使用默认值
            head_top_y = int(h * 0.15)  # 默认在画面15%处
            face_center_x = w // 2
            print(f"[DEBUG] 无有效面部关键点，使用默认头顶位置y值: {head_top_y}")
        
        # 计算线条宽度：不贯穿整个图片，只显示在头顶附近
        line_width = w // 3  # 线条长度为画面宽度的1/3
        
        # 线条起点和终点（居中显示在头顶）
        head_center_x = face_center_x  # 使用面部中心x坐标
        line_start = (head_center_x - line_width // 2, head_top_y)
        line_end = (head_center_x + line_width // 2, head_top_y)
        
        # 文本位置，在线上方
        text_position = (head_center_x, head_top_y - 15)
        
        # 调试信息
        print(f"[DEBUG] 最终头顶线位置: Y={head_top_y}, X={head_center_x}")
        
        return {
            "line_start": line_start,
            "line_end": line_end,
            "text_position": text_position,
            "head_top_y": head_top_y,
            "head_center_x": head_center_x
        }
    
    def draw_lines(self, canvas, lines_info):
        """
        在画布上绘制头顶高度线
        
        参数:
            canvas: 要绘制的画布
            lines_info: 辅助线信息
        """
        # 线条颜色，带有Alpha通道(完全不透明)
        color = (*self.line_color, 255)  # (B,G,R,A)
        thickness = self.line_thickness
        
        # 绘制水平线
        cv2.line(canvas, lines_info["line_start"], lines_info["line_end"], color, thickness)
        
        # 在线条端点添加短的垂直线段（装饰）
        line_start, line_end = lines_info["line_start"], lines_info["line_end"]
        vert_line_length = 15  # 垂直线长度
        
        # 左侧垂直线
        cv2.line(canvas, 
                 (line_start[0], line_start[1]), 
                 (line_start[0], line_start[1] + vert_line_length), 
                 color, thickness)
        
        # 右侧垂直线
        cv2.line(canvas, 
                 (line_end[0], line_end[1]), 
                 (line_end[0], line_end[1] + vert_line_length), 
                 color, thickness)
        
        # 添加中央垂直指示线，从线的中心向上延伸
        center_x = lines_info["head_center_x"]
        center_y = lines_info["head_top_y"]
        
        # 绘制中央垂直线（向上）
        cv2.line(canvas,
                (center_x, center_y),
                (center_x, center_y - 20),  # 向上延伸20像素
                color, thickness)


class BodyAssistantLine(AssistantLineGenerator):
    """
    躯干框辅助线，第三种辅助线
    绘制人物躯干框（半透明填充+边框+十字线）
    """
    def __init__(self, line_color=(255, 255, 255), line_thickness=4, video_width=1920, video_height=1080):
        """初始化为白色线条，粗细为4"""
        super().__init__(line_color, line_thickness, video_width, video_height)
        self.fill_color = (0, 255, 0)  # 填充颜色为绿色
        self.alpha = 0.18  # 填充透明度
        
    def calculate_lines_info(self, frame_size, kpt_33x2):
        """
        计算躯干框信息，参考assistant_lines2.py中的compute_lines_info_for_second函数
        
        参数:
            frame_size: (height, width)元组
            kpt_33x2: 关键点数据，(33,2)张量
            
        返回:
            字典，包含躯干框绘制所需信息
        """
        h, w = frame_size
        # 判断是否需要旋转
        need_rotate = (w > h)
        
        # 将相对坐标 -> 像素坐标
        coords = kpt_33x2.clone()
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.int().numpy()  # => (33,2)
        
        # 提取肩膀和脚踝
        L_shoulder = (coords[11][0], coords[11][1])
        R_shoulder = (coords[12][0], coords[12][1])
        L_ankle = (coords[27][0], coords[27][1])
        R_ankle = (coords[28][0], coords[28][1])

        print(f"[DEBUG] body assistant: left_shoulder={L_shoulder}, right_shoulder={R_shoulder}, "
              f"left_ankle={L_ankle}, right_ankle={R_ankle}")

        if not need_rotate:
            # 水平范围：仅用左右肩
            trunk_left = min(L_shoulder[0], R_shoulder[0])
            trunk_right = max(L_shoulder[0], R_shoulder[0])
            # 垂直范围：用肩部上沿到脚踝下沿
            trunk_top = min(L_shoulder[1], R_shoulder[1])
            trunk_bottom = max(L_ankle[1], R_ankle[1])
        else:
            trunk_left = min(L_shoulder[0], R_shoulder[0])
            trunk_right = max(L_ankle[0], R_ankle[0])
            trunk_top = min(L_shoulder[1], L_ankle[1])
            trunk_bottom = max(R_shoulder[1], L_ankle[1])
            
        mid_x = (trunk_left + trunk_right) // 2
        mid_y = (trunk_top + trunk_bottom) // 2

        return {
            "trunk_left": int(trunk_left),
            "trunk_right": int(trunk_right),
            "trunk_top": int(trunk_top),
            "trunk_bottom": int(trunk_bottom),
            "mid_x": int(mid_x),
            "mid_y": int(mid_y),
            "need_rotate": need_rotate
        }
    
    def draw_lines(self, canvas, lines_info):
        """
        在画布上绘制躯干框
        
        参数:
            canvas: 要绘制的画布
            lines_info: 辅助线信息
        """
        # 线条颜色，带有Alpha通道(完全不透明)
        color = (*self.line_color, 255)  # (B,G,R,A)
        fill_color = (*self.fill_color, int(self.alpha * 255))  # 半透明填充颜色
        thickness = self.line_thickness
        
        # 提取必要信息
        tleft = lines_info["trunk_left"]
        tright = lines_info["trunk_right"]
        ttop = lines_info["trunk_top"]
        tbottom = lines_info["trunk_bottom"]
        midx = lines_info["mid_x"]
        midy = lines_info["mid_y"]
        
        # 绘制躯干框（半透明填充）
        cv2.rectangle(canvas, (tleft, ttop), (tright, tbottom), fill_color, -1)
        
        # 绘制躯干框边框
        cv2.rectangle(canvas, (tleft, ttop), (tright, tbottom), color, thickness)
        
        # 绘制十字线
        cv2.line(canvas, (tleft, midy), (tright, midy), color, thickness)
        cv2.line(canvas, (midx, ttop), (midx, tbottom), color, thickness)


class SkeletonAssistantLine(AssistantLineGenerator):
    """
    骨线图辅助线，第四种辅助线
    绘制人体骨架连线，显示关键点之间的连接关系
    根据人物实时运动变化更新线条位置
    """
    def __init__(self, line_color=(0, 0, 255), line_thickness=4, video_width=1920, video_height=1080):
        """初始化为红色线条，粗细为4"""
        super().__init__(line_color, line_thickness, video_width, video_height)
        # 定义骨架连接关系（关键点对）
        self.skeleton_connections = [
            # 躯干
            (11, 12),  # 左肩 -> 右肩
            (11, 23),  # 左肩 -> 左臀
            (12, 24),  # 右肩 -> 右臀
            (23, 24),  # 左臀 -> 右臀
            
            # 头部和颈部
            (0, 1),    # 鼻子 -> 左眼内角
            (1, 2),    # 左眼内角 -> 左眼
            (2, 3),    # 左眼 -> 左眼外角
            (0, 4),    # 鼻子 -> 右眼内角
            (4, 5),    # 右眼内角 -> 右眼
            (5, 6),    # 右眼 -> 右眼外角
            (0, 9),    # 鼻子 -> 嘴左角
            (9, 10),   # 嘴左角 -> 嘴右角
            (0, 10),   # 鼻子 -> 嘴右角
            
            # 鼻子到脖子中点
            (0, 33),   # 鼻子 -> 脖子中点（虚拟点）
            
            # 左臂
            (11, 13),  # 左肩 -> 左肘
            (13, 15),  # 左肘 -> 左手腕
            (15, 17),  # 左手腕 -> 左手
            (15, 19),  # 左手腕 -> 左拇指
            (17, 19),  # 左手 -> 左拇指
            (17, 21),  # 左手 -> 左小指
            (19, 21),  # 左拇指 -> 左小指
            
            # 右臂
            (12, 14),  # 右肩 -> 右肘
            (14, 16),  # 右肘 -> 右手腕
            (16, 18),  # 右手腕 -> 右手
            (16, 20),  # 右手腕 -> 右拇指
            (18, 20),  # 右手 -> 右拇指
            (18, 22),  # 右手 -> 右小指
            (20, 22),  # 右拇指 -> 右小指
            
            # 左腿
            (23, 25),  # 左臀 -> 左膝
            (25, 27),  # 左膝 -> 左踝
            (27, 29),  # 左踝 -> 左脚
            (27, 31),  # 左踝 -> 左脚趾
            (29, 31),  # 左脚 -> 左脚趾
            
            # 右腿
            (24, 26),  # 右臀 -> 右膝
            (26, 28),  # 右膝 -> 右踝
            (28, 30),  # 右踝 -> 右脚
            (28, 32),  # 右踝 -> 右脚趾
            (30, 32),  # 右脚 -> 右脚趾
        ]
        
    def calculate_lines_info(self, frame_size, kpt_33x2):
        """
        计算骨架线信息
        
        参数:
            frame_size: (height, width)元组
            kpt_33x2: 关键点数据，(33,2)张量
            
        返回:
            字典，包含骨架线绘制所需信息
        """
        h, w = frame_size
        # 将相对坐标 -> 像素坐标
        coords = kpt_33x2.clone()
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.int().numpy()  # => (33,2)
        
        # 创建虚拟点 - 脖子中点 (index 33)
        neck_point = np.mean([coords[11], coords[12]], axis=0).astype(int)
        coords = np.vstack([coords, [neck_point]])
        
        # 判断是否需要旋转
        need_rotate = (w > h)
        
        # 计算关键点可见性（简单判断坐标是否为0）
        visible_points = np.all(coords > 0, axis=1)
        
        # 提取有效的连接线（两端点都可见）
        valid_connections = []
        for conn in self.skeleton_connections:
            p1_idx, p2_idx = conn
            if p1_idx < len(visible_points) and p2_idx < len(visible_points):
                if visible_points[p1_idx] and visible_points[p2_idx]:
                    p1 = (int(coords[p1_idx][0]), int(coords[p1_idx][1]))
                    p2 = (int(coords[p2_idx][0]), int(coords[p2_idx][1]))
                    valid_connections.append((p1, p2))
        
        return {
            "valid_connections": valid_connections,
            "keypoints": coords,
            "visible_points": visible_points,
            "need_rotate": need_rotate
        }
    
    def draw_lines(self, canvas, lines_info):
        """
        在画布上绘制骨架线
        
        参数:
            canvas: 要绘制的画布
            lines_info: 辅助线信息
        """
        # 线条颜色，带有Alpha通道(完全不透明)
        color = (*self.line_color, 255)  # (B,G,R,A)
        thickness = self.line_thickness
        
        # 绘制骨架连接线
        for connection in lines_info["valid_connections"]:
            p1, p2 = connection
            cv2.line(canvas, p1, p2, color, thickness)
        
        # 在关键点处绘制小圆点
        keypoints = lines_info["keypoints"]
        visible = lines_info["visible_points"]
        
        # 只绘制可见的关键点
        for i in range(len(keypoints)):
            if i < len(visible) and visible[i]:
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                # 绘制关键点小圆圈
                cv2.circle(canvas, (x, y), 3, color, -1)


class FishboneAssistantLine(AssistantLineGenerator):
    """
    鱼骨线辅助线，第五种辅助线
    绘制一根竖线和三根横线：
    - 竖线为人物正中分割线，连接头部和腹部中心
    - 第一根横线是两个肩膀的连线
    - 第二根横线是髋部连线
    - 第三根横线是膝盖连线
    根据人物实时运动变化更新线条位置
    """
    def __init__(self, line_color=(0, 165, 255), line_thickness=3, video_width=1920, video_height=1080):
        """初始化为橙色线条，粗细为3"""
        super().__init__(line_color, line_thickness, video_width, video_height)
        
    def calculate_lines_info(self, frame_size, kpt_33x2):
        """
        计算鱼骨线信息
        
        参数:
            frame_size: (height, width)元组
            kpt_33x2: 关键点数据，(33,2)张量
            
        返回:
            字典，包含鱼骨线绘制所需信息
        """
        h, w = frame_size
        # 将相对坐标 -> 像素坐标
        coords = kpt_33x2.clone()
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.int().numpy()  # => (33,2)
        
        # 判断是否需要旋转
        need_rotate = (w > h)
        
        # 提取关键点坐标
        # 肩膀关键点 (11: 左肩, 12: 右肩)
        left_shoulder = coords[11]
        right_shoulder = coords[12]
        
        # 髋部关键点 (23: 左髋, 24: 右髋)
        left_hip = coords[23]
        right_hip = coords[24]
        
        # 膝盖关键点 (25: 左膝, 26: 右膝)
        left_knee = coords[25]
        right_knee = coords[26]
        
        # 头部关键点 (0: 鼻子, 中心点可以用鼻子代替)
        nose = coords[0]
        
        # 计算各个横线的中心点
        shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                          (left_shoulder[1] + right_shoulder[1]) // 2)
        
        hip_center = ((left_hip[0] + right_hip[0]) // 2, 
                     (left_hip[1] + right_hip[1]) // 2)
        
        knee_center = ((left_knee[0] + right_knee[0]) // 2, 
                      (left_knee[1] + right_knee[1]) // 2)
        
        # 计算竖线起点和终点（头顶上方延伸 + 膝盖下方延伸）
        # 注意：y坐标小的在上方，大的在下方
        
        # 估计头顶位置（鼻子上方一定距离）
        face_height = int(shoulder_center[1] - nose[1])  # 面部高度估算
        vertical_top_y = max(0, nose[1] - int(face_height * 1.0))  # 头顶上方延伸
        vertical_top = (nose[0], vertical_top_y)
        
        # 竖线终点延伸到膝盖下方更远距离
        extend_factor = 1.5  # 延伸因子增大到1.5（原为1.2）
        knee_to_hip_height = knee_center[1] - hip_center[1]
        vertical_bottom_y = int(knee_center[1] + knee_to_hip_height * (extend_factor - 1))
        vertical_bottom = (knee_center[0], vertical_bottom_y)
        
        # 调整横线宽度（大幅延伸）
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        line_extension = int(shoulder_width * 1.2)  # 延伸120%（大幅增加）
        
        # 延伸肩膀线 - 取消边界保护
        shoulder_left_x = int(left_shoulder[0] - line_extension)
        shoulder_right_x = int(right_shoulder[0] + line_extension)
        shoulder_left = (shoulder_left_x, left_shoulder[1])
        shoulder_right = (shoulder_right_x, right_shoulder[1])
        
        # 延伸髋部线 - 取消边界保护
        hip_left_x = int(left_hip[0] - line_extension)
        hip_right_x = int(right_hip[0] + line_extension)
        hip_left = (hip_left_x, left_hip[1])
        hip_right = (hip_right_x, right_hip[1])
        
        # 延伸膝盖线 - 取消边界保护
        knee_left_x = int(left_knee[0] - line_extension)
        knee_right_x = int(right_knee[0] + line_extension)
        knee_left = (knee_left_x, left_knee[1])
        knee_right = (knee_right_x, right_knee[1])
        
        return {
            "vertical_top": vertical_top,
            "vertical_bottom": vertical_bottom,
            "shoulder_left": shoulder_left,
            "shoulder_right": shoulder_right,
            "hip_left": hip_left,
            "hip_right": hip_right,
            "knee_left": knee_left,
            "knee_right": knee_right,
            "need_rotate": need_rotate
        }
    
    def draw_lines(self, canvas, lines_info):
        """
        在画布上绘制鱼骨线
        
        参数:
            canvas: 要绘制的画布
            lines_info: 辅助线信息
        """
        # 线条颜色，带有Alpha通道(完全不透明)
        color = (*self.line_color, 255)  # (B,G,R,A)
        thickness = self.line_thickness
        
        # 绘制竖线（头部到膝盖下方）
        cv2.line(canvas, 
                lines_info["vertical_top"], 
                lines_info["vertical_bottom"], 
                color, thickness)
        
        # 绘制肩膀横线
        cv2.line(canvas, 
                lines_info["shoulder_left"], 
                lines_info["shoulder_right"], 
                color, thickness)
        
        # 绘制髋部横线
        cv2.line(canvas, 
                lines_info["hip_left"], 
                lines_info["hip_right"], 
                color, thickness)
        
        # 绘制膝盖横线
        cv2.line(canvas, 
                lines_info["knee_left"], 
                lines_info["knee_right"], 
                color, thickness)


class ShoulderRotationAssistantLine(AssistantLineGenerator):
    """
    肩膀旋转角度辅助线，用于显示肩膀旋转角度
    绘制一条肩膀连线，并计算其相对于参考帧的旋转角度
    角度值显示为角度值，正值表示顺时针旋转，负值表示逆时针旋转
    """
    def __init__(self, line_color=(255, 128, 0), line_thickness=4, video_width=1920, video_height=1080):
        """初始化为橙色线条，粗细为4"""
        super().__init__(line_color, line_thickness, video_width, video_height)
        self.text_color = (0, 0, 255)  # 红色文字 (BGR)
        self.text_size = 0.8  # 文字大小
        self.text_thickness = 2  # 文字粗细
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.reference_shoulder_width = None  # 参考肩宽（第一帧）
        
        # 增加前后朝向检测所需属性
        self.facing_front = True  # 默认是面向前方
        self.prev_angles = []  # 存储最近的角度值，用于检测转向
        self.angle_history_size = 5  # 历史记录大小
        self.front_back_transition_threshold = 85  # 接近90度时考虑前后转向
        self.is_rotating = False  # 是否处于旋转状态
        self.rotation_count = 0  # 旋转计数
        
        # 角度平滑处理相关属性
        self.smoothed_angles = []  # 平滑后的角度历史
        self.smooth_window_size = 3  # 平滑窗口大小（越大越平滑，但延迟也越大）
        self.max_angle_change = 50  # 单帧最大角度变化（度）
        self.last_smoothed_angle = None  # 上一帧的平滑角度
        self.trend_direction = 0  # 0表示未确定，1表示增加，-1表示减少
        
    def smooth_angle(self, new_angle):
        """
        对角度进行平滑处理，防止剧烈抖动并保持单向变化
        
        参数:
            new_angle: 当前帧计算出的角度
            
        返回:
            平滑后的角度
        """
        # 如果是第一次平滑，直接返回当前角度
        if self.last_smoothed_angle is None:
            self.last_smoothed_angle = new_angle
            # 初始化趋势方向
            self.trend_direction = 0  # 0表示未确定，1表示增加，-1表示减少
            return new_angle
        
        # 检测角度变化方向
        angle_diff = new_angle - self.last_smoothed_angle
        
        # 如果方向尚未确定且变化明显，则确定方向
        if self.trend_direction == 0 and abs(angle_diff) > 3:
            self.trend_direction = 1 if angle_diff > 0 else -1
            print(f"[INFO] 确定旋转方向: {'增加' if self.trend_direction > 0 else '减少'}")
        
        # 根据当前趋势方向过滤角度变化
        if self.trend_direction != 0:
            # 如果角度变化与趋势方向相反且变化不大，则忽略变化，保持单向运动
            if (self.trend_direction > 0 and angle_diff < 0) or (self.trend_direction < 0 and angle_diff > 0):
                # 如果反向变化幅度超过阈值，则认为是到达极值点，改变趋势方向
                if abs(angle_diff) > self.max_angle_change:
                    # 到达极值点，改变趋势方向
                    self.trend_direction = -self.trend_direction
                    print(f"[INFO] 检测到极值点，改变旋转方向: {'增加' if self.trend_direction > 0 else '减少'}")
                else:
                    # 变化不大，则保持原方向，忽略小的反向变化
                    new_angle = self.last_smoothed_angle
        
        # 限制单帧角度变化幅度，但保持变化方向
        if abs(angle_diff) > self.max_angle_change:
            # 限制变化幅度，保持方向
            new_angle = self.last_smoothed_angle + (self.max_angle_change if angle_diff > 0 else -self.max_angle_change)
        
        # 添加到平滑窗口
        self.smoothed_angles.append(new_angle)
        
        # 保持窗口大小
        if len(self.smoothed_angles) > self.smooth_window_size:
            self.smoothed_angles.pop(0)
        
        # 计算平滑角度（加权平均，最近的角度权重高）
        if len(self.smoothed_angles) > 2:
            weights = np.linspace(0.5, 1.0, len(self.smoothed_angles))
            weighted_sum = sum(a * w for a, w in zip(self.smoothed_angles, weights))
            weighted_average = weighted_sum / sum(weights)
            smoothed_angle = int(weighted_average)
        else:
            # 样本太少时直接使用最新值
            smoothed_angle = new_angle
        
        # 更新上一帧的平滑角度
        self.last_smoothed_angle = smoothed_angle
        
        return smoothed_angle
        
    def calculate_lines_info(self, frame_size, kpt_33x2):
        """
        计算肩膀旋转角度信息，基于肩膀连线在X轴的投影
        
        参数:
            frame_size: (height, width)元组
            kpt_33x2: 关键点数据，(33,2)张量
            
        返回:
            字典，包含肩膀连线和旋转角度信息
        """
        h, w = frame_size
        
        # 将相对坐标 -> 像素坐标
        coords = kpt_33x2.clone()
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.int().numpy()
        
        # 提取肩膀关键点（Mediapipe Pose关键点：11=左肩，12=右肩）
        left_shoulder = tuple(coords[11])
        right_shoulder = tuple(coords[12])
        
        # 提取髋部关键点（用于辅助判断朝向）
        left_hip = tuple(coords[23])
        right_hip = tuple(coords[24])
        
        # 计算当前帧肩膀连线在X轴上的投影长度（水平投影）
        projection_length = abs(right_shoulder[0] - left_shoulder[0])
        
        # 计算肩宽（两肩点之间的直线距离）
        current_shoulder_width = np.sqrt(
            (right_shoulder[0] - left_shoulder[0])**2 + 
            (right_shoulder[1] - left_shoulder[1])**2
        )
        
        # 如果是第一帧或参考肩宽未设置，则使用当前帧作为参考
        if self.reference_shoulder_width is None:
            self.reference_shoulder_width = projection_length
            print(f"[INFO] 设置肩膀旋转角度参考宽度: {self.reference_shoulder_width:.1f}像素")
        
        # 计算投影比例（当前投影/参考投影）
        projection_ratio = projection_length / self.reference_shoulder_width if self.reference_shoulder_width > 0 else 1.0
        
        # 限制投影比例在[0,1]范围内
        projection_ratio = max(0.001, min(1, projection_ratio))  # 避免除零错误
        
        # 计算原始旋转角度（基本角度）：arccos(投影比例)，转换为角度
        # 当投影比例=1时，角度=0度（身体正对前方）
        # 当投影比例接近0时，角度接近90度（身体侧向）
        raw_angle = np.degrees(np.arccos(projection_ratio))
        
        # 确定基本旋转方向
        if right_shoulder[1] < left_shoulder[1]:
            # 如果右肩在左肩左侧，则是向左转（逆时针）
            raw_angle = -raw_angle
            
        # 检测前后朝向变化的逻辑
        if len(self.prev_angles) > 0:
            last_angle = self.prev_angles[-1]
            
            # 当接近90度时检测朝向变化
            if abs(raw_angle) > self.front_back_transition_threshold:
                # 如果角度开始减小（朝向侧面的角度变小）说明可能在改变朝向
                if abs(raw_angle) < abs(last_angle) and not self.is_rotating:
                    self.is_rotating = True
                    print(f"[INFO] 检测到可能的朝向变化，当前角度: {raw_angle:.1f}, 上一个角度: {last_angle:.1f}")
                    
            # 如果之前检测到可能在旋转，且角度变化不大，说明旋转完成
            elif self.is_rotating and abs(raw_angle - last_angle) < 10:
                self.is_rotating = False
                self.facing_front = not self.facing_front
                self.rotation_count += 1
                print(f"[INFO] 朝向变化完成: {'正面' if self.facing_front else '背面'}, 旋转次数: {self.rotation_count}")
        
        # 更新角度历史记录
        self.prev_angles.append(raw_angle)
        if len(self.prev_angles) > self.angle_history_size:
            self.prev_angles.pop(0)
            
        # 根据当前朝向确定最终角度
        final_angle = raw_angle
        if not self.facing_front:
            # 背面朝向时，角度应该继续增加
            if final_angle > 0:
                final_angle = 180 - final_angle
            else:
                final_angle = -180 - final_angle
        
        # 取整数部分
        raw_rotation_angle = int(final_angle)
        
        # 应用平滑处理
        rotation_angle = self.smooth_angle(raw_rotation_angle)
        
        # 计算文本位置（肩膀连线中点下方，更靠左）
        mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
        mid_y = (left_shoulder[1] + right_shoulder[1]) // 2
        text_position = (mid_x - 100, mid_y + 30)  # 偏左下方更多
        
        # 创建肩膀连线的起点和终点
        # 在视觉上延长肩膀连线，使其更加明显
        extension_factor = 0.3  # 延长30%
        shoulder_vector = (right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1])
        vector_length = np.sqrt(shoulder_vector[0]**2 + shoulder_vector[1]**2)
        
        if vector_length > 0:
            unit_vector = (shoulder_vector[0]/vector_length, shoulder_vector[1]/vector_length)
            extension = vector_length * extension_factor
            
            extended_left = (
                int(left_shoulder[0] - unit_vector[0] * extension),
                int(left_shoulder[1] - unit_vector[1] * extension)
            )
            
            extended_right = (
                int(right_shoulder[0] + unit_vector[0] * extension),
                int(right_shoulder[1] + unit_vector[1] * extension)
            )
        else:
            # 如果肩膀重合，使用默认扩展
            extended_left = left_shoulder
            extended_right = right_shoulder
        
        # 返回当前帧的所有计算信息
        return {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "extended_left": extended_left,
            "extended_right": extended_right,
            "mid_point": (mid_x, mid_y),
            "rotation_angle": rotation_angle,
            "raw_angle": raw_angle,  # 原始角度，用于调试
            "projection_ratio": projection_ratio,
            "projection_length": projection_length,
            "current_width": current_shoulder_width,
            "text_position": text_position,
            "facing_front": self.facing_front  # 当前朝向
        }
    
    def draw_lines(self, canvas, lines_info):
        """
        在画布上绘制肩膀连线和旋转角度
        
        参数:
            canvas: 要绘制的画布
            lines_info: 辅助线信息
        """
        # 线条颜色，带有Alpha通道(完全不透明)
        color = (*self.line_color, 255)  # (B,G,R,A)
        text_color = (*self.text_color, 255)  # (B,G,R,A)
        thickness = self.line_thickness
        
        # 绘制延长的肩膀连线
        cv2.line(canvas, 
                lines_info["extended_left"], 
                lines_info["extended_right"], 
                color, thickness)
        
        # 在肩膀点上画小圆点
        cv2.circle(canvas, lines_info["left_shoulder"], radius=5, color=color, thickness=-1)
        cv2.circle(canvas, lines_info["right_shoulder"], radius=5, color=color, thickness=-1)
        
        # 显示旋转角度 - 移除正负号，只显示绝对值
        angle_value = lines_info['rotation_angle']
        angle_text = f"{abs(angle_value)}"  # 仅显示数字，不添加任何符号或朝向信息
        
        cv2.putText(canvas, 
                   angle_text, 
                   lines_info["text_position"], 
                   self.font_face, 
                   self.text_size, 
                   text_color, 
                   self.text_thickness)


class HipRotationAssistantLine(AssistantLineGenerator):
    """
    髋部旋转角度辅助线，用于显示髋部旋转角度
    绘制一条髋部连线，并计算其相对于参考帧的旋转角度
    角度值显示为角度值，正值表示顺时针旋转，负值表示逆时针旋转
    """
    def __init__(self, line_color=(0, 128, 255), line_thickness=4, video_width=1920, video_height=1080):
        """初始化为橙色线条，粗细为4"""
        super().__init__(line_color, line_thickness, video_width, video_height)
        self.text_color = (0, 0, 255)  # 红色文字 (BGR)
        self.text_size = 0.8  # 文字大小
        self.text_thickness = 2  # 文字粗细
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.reference_hip_width = None  # 参考髋宽（第一帧）
        
        # 增加前后朝向检测所需属性
        self.facing_front = True  # 默认是面向前方
        self.prev_angles = []  # 存储最近的角度值，用于检测转向
        self.angle_history_size = 5  # 历史记录大小
        self.front_back_transition_threshold = 85  # 接近90度时考虑前后转向
        self.is_rotating = False  # 是否处于旋转状态
        self.rotation_count = 0  # 旋转计数
        
        # 角度平滑处理相关属性
        self.smoothed_angles = []  # 平滑后的角度历史
        self.smooth_window_size = 3  # 平滑窗口大小（越大越平滑，但延迟也越大）
        self.max_angle_change = 50  # 单帧最大角度变化（度）
        self.last_smoothed_angle = None  # 上一帧的平滑角度
        self.trend_direction = 0  # 0表示未确定，1表示增加，-1表示减少
        
    def smooth_angle(self, new_angle):
        """
        对角度进行平滑处理，防止剧烈抖动并保持单向变化
        
        参数:
            new_angle: 当前帧计算出的角度
            
        返回:
            平滑后的角度
        """
        # 如果是第一次平滑，直接返回当前角度
        if self.last_smoothed_angle is None:
            self.last_smoothed_angle = new_angle
            # 初始化趋势方向
            self.trend_direction = 0  # 0表示未确定，1表示增加，-1表示减少
            return new_angle
        
        # 检测角度变化方向
        angle_diff = new_angle - self.last_smoothed_angle
        
        # 如果方向尚未确定且变化明显，则确定方向
        if self.trend_direction == 0 and abs(angle_diff) > 3:
            self.trend_direction = 1 if angle_diff > 0 else -1
            print(f"[INFO] 确定旋转方向: {'增加' if self.trend_direction > 0 else '减少'}")
        
        # 根据当前趋势方向过滤角度变化
        if self.trend_direction != 0:
            # 如果角度变化与趋势方向相反且变化不大，则忽略变化，保持单向运动
            if (self.trend_direction > 0 and angle_diff < 0) or (self.trend_direction < 0 and angle_diff > 0):
                # 如果反向变化幅度超过阈值，则认为是到达极值点，改变趋势方向
                if abs(angle_diff) > self.max_angle_change:
                    # 到达极值点，改变趋势方向
                    self.trend_direction = -self.trend_direction
                    print(f"[INFO] 检测到极值点，改变旋转方向: {'增加' if self.trend_direction > 0 else '减少'}")
                else:
                    # 变化不大，则保持原方向，忽略小的反向变化
                    new_angle = self.last_smoothed_angle
        
        # 限制单帧角度变化幅度，但保持变化方向
        if abs(angle_diff) > self.max_angle_change:
            # 限制变化幅度，保持方向
            new_angle = self.last_smoothed_angle + (self.max_angle_change if angle_diff > 0 else -self.max_angle_change)
        
        # 添加到平滑窗口
        self.smoothed_angles.append(new_angle)
        
        # 保持窗口大小
        if len(self.smoothed_angles) > self.smooth_window_size:
            self.smoothed_angles.pop(0)
        
        # 计算平滑角度（加权平均，最近的角度权重高）
        if len(self.smoothed_angles) > 2:
            weights = np.linspace(0.5, 1.0, len(self.smoothed_angles))
            weighted_sum = sum(a * w for a, w in zip(self.smoothed_angles, weights))
            weighted_average = weighted_sum / sum(weights)
            smoothed_angle = int(weighted_average)
        else:
            # 样本太少时直接使用最新值
            smoothed_angle = new_angle
        
        # 更新上一帧的平滑角度
        self.last_smoothed_angle = smoothed_angle
        
        return smoothed_angle
        
    def calculate_lines_info(self, frame_size, kpt_33x2):
        """
        计算髋部旋转角度信息，基于髋部连线在X轴的投影
        
        参数:
            frame_size: (height, width)元组
            kpt_33x2: 关键点数据，(33,2)张量
            
        返回:
            字典，包含髋部连线和旋转角度信息
        """
        h, w = frame_size
        
        # 将相对坐标 -> 像素坐标
        coords = kpt_33x2.clone()
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.int().numpy()
        
        # 提取髋部关键点（Mediapipe Pose关键点：23=左髋，24=右髋）
        left_hip = tuple(coords[23])
        right_hip = tuple(coords[24])
        
        # 计算当前帧髋部连线在X轴上的投影长度（水平投影）
        projection_length = abs(right_hip[0] - left_hip[0])
        
        # 计算髋宽（两髋点之间的直线距离）
        current_hip_width = np.sqrt(
            (right_hip[0] - left_hip[0])**2 + 
            (right_hip[1] - left_hip[1])**2
        )
        
        # 如果是第一帧或参考髋宽未设置，则使用当前帧作为参考
        if self.reference_hip_width is None:
            self.reference_hip_width = projection_length
            print(f"[INFO] 设置髋部旋转角度参考宽度: {self.reference_hip_width:.1f}像素")
        
        # 计算投影比例（当前投影/参考投影）
        projection_ratio = projection_length / self.reference_hip_width if self.reference_hip_width > 0 else 1.0
        
        # 限制投影比例在[0,1]范围内
        projection_ratio = max(0.001, min(1, projection_ratio))  # 避免除零错误
        
        # 计算原始旋转角度（基本角度）：arccos(投影比例)，转换为角度
        # 当投影比例=1时，角度=0度（身体正对前方）
        # 当投影比例接近0时，角度接近90度（身体侧向）
        raw_angle = np.degrees(np.arccos(projection_ratio))
        
        # 确定基本旋转方向
        if right_hip[1] < left_hip[1]:
            # 如果右髋在左髋左侧，则是向左转（逆时针）
            raw_angle = -raw_angle
            
        # 检测前后朝向变化的逻辑
        if len(self.prev_angles) > 0:
            last_angle = self.prev_angles[-1]
            
            # 当接近90度时检测朝向变化
            if abs(raw_angle) > self.front_back_transition_threshold:
                # 如果角度开始减小（朝向侧面的角度变小）说明可能在改变朝向
                if abs(raw_angle) < abs(last_angle) and not self.is_rotating:
                    self.is_rotating = True
                    print(f"[INFO] 检测到可能的朝向变化，当前角度: {raw_angle:.1f}, 上一个角度: {last_angle:.1f}")
                    
            # 如果之前检测到可能在旋转，且角度变化不大，说明旋转完成
            elif self.is_rotating and abs(raw_angle - last_angle) < 10:
                self.is_rotating = False
                self.facing_front = not self.facing_front
                self.rotation_count += 1
                print(f"[INFO] 朝向变化完成: {'正面' if self.facing_front else '背面'}, 旋转次数: {self.rotation_count}")
        
        # 更新角度历史记录
        self.prev_angles.append(raw_angle)
        if len(self.prev_angles) > self.angle_history_size:
            self.prev_angles.pop(0)
            
        # 根据当前朝向确定最终角度
        final_angle = raw_angle
        if not self.facing_front:
            # 背面朝向时，角度应该继续增加
            if final_angle > 0:
                final_angle = 180 - final_angle
            else:
                final_angle = -180 - final_angle
        
        # 取整数部分
        raw_rotation_angle = int(final_angle)
        
        # 应用平滑处理
        rotation_angle = self.smooth_angle(raw_rotation_angle)
        
        # 计算文本位置（髋部连线中点下方，更靠左）
        mid_x = (left_hip[0] + right_hip[0]) // 2
        mid_y = (left_hip[1] + right_hip[1]) // 2
        text_position = (mid_x - 100, mid_y + 30)  # 偏左下方更多
        
        # 创建髋部连线的起点和终点
        # 在视觉上延长髋部连线，使其更加明显
        extension_factor = 0.3  # 延长30%
        hip_vector = (right_hip[0] - left_hip[0], right_hip[1] - left_hip[1])
        vector_length = np.sqrt(hip_vector[0]**2 + hip_vector[1]**2)
        
        if vector_length > 0:
            unit_vector = (hip_vector[0]/vector_length, hip_vector[1]/vector_length)
            extension = vector_length * extension_factor
            
            extended_left = (
                int(left_hip[0] - unit_vector[0] * extension),
                int(left_hip[1] - unit_vector[1] * extension)
            )
            
            extended_right = (
                int(right_hip[0] + unit_vector[0] * extension),
                int(right_hip[1] + unit_vector[1] * extension)
            )
        else:
            # 如果髋部重合，使用默认扩展
            extended_left = left_hip
            extended_right = right_hip
        
        # 返回当前帧的所有计算信息
        return {
            "left_hip": left_hip,
            "right_hip": right_hip,
            "extended_left": extended_left,
            "extended_right": extended_right,
            "mid_point": (mid_x, mid_y),
            "rotation_angle": rotation_angle,
            "raw_angle": raw_angle,  # 原始角度，用于调试
            "projection_ratio": projection_ratio,
            "projection_length": projection_length,
            "current_width": current_hip_width,
            "text_position": text_position,
            "facing_front": self.facing_front  # 当前朝向
        }
    
    def draw_lines(self, canvas, lines_info):
        """
        在画布上绘制髋部连线和旋转角度
        
        参数:
            canvas: 要绘制的画布
            lines_info: 辅助线信息
        """
        # 线条颜色，带有Alpha通道(完全不透明)
        color = (*self.line_color, 255)  # (B,G,R,A)
        text_color = (*self.text_color, 255)  # (B,G,R,A)
        thickness = self.line_thickness
        
        # 绘制延长的髋部连线
        cv2.line(canvas, 
                lines_info["extended_left"], 
                lines_info["extended_right"], 
                color, thickness)
        
        # 在髋部点上画小圆点
        cv2.circle(canvas, lines_info["left_hip"], radius=5, color=color, thickness=-1)
        cv2.circle(canvas, lines_info["right_hip"], radius=5, color=color, thickness=-1)
        
        # 显示旋转角度 - 移除正负号，只显示绝对值
        angle_value = lines_info['rotation_angle']
        angle_text = f"{abs(angle_value)}"  # 仅显示数字，不添加任何符号或朝向信息
        
        cv2.putText(canvas, 
                   angle_text, 
                   lines_info["text_position"], 
                   self.font_face, 
                   self.text_size, 
                   text_color, 
                   self.text_thickness)


def generate_transparent_assistant_lines(
    video_path: str,
    keypoint_data: torch.Tensor,
    output_base_folder: str,
    line_type: str = "stance",
    need_rotate: bool = None,
    stage_indices: dict = None,
    stage_name_for_ready: str = "0",
    video_width: int = None,
    video_height: int = None
):
    """
    生成透明辅助线PNG图片
    
    参数:
        video_path: 原视频路径
        keypoint_data: 关键点数据，(N,33,2)张量
        output_base_folder: 输出基础目录，将在其下创建img/站姿等子目录
        line_type: 辅助线类型，默认为"stance"(站姿)
        need_rotate: 是否需要旋转，None表示自动检测
        stage_indices: 阶段索引字典，如 {"0": [10, 20], "1": [30, 40]}
        stage_name_for_ready: 准备阶段的索引名，默认为"0"
        video_width: 视频宽度，默认为None（会从视频中读取）
        video_height: 视频高度，默认为None（会从视频中读取）
        
    返回:
        生成图片的路径列表
    """
    # 启用性能计时
    import time
    start_time = time.time()
    
    # 仅当未提供宽高参数时才读取视频获取尺寸
    if video_width is None or video_height is None:
        # 读取视频确定尺寸
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERR] 无法打开视频: {video_path}")
            return []
        
        # 获取视频信息
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        # 使用传入的宽高参数
        print(f"[INFO] 使用传入的视频尺寸: {video_width}x{video_height}")
        total_frames = len(keypoint_data)
    
    # 从 manifest 读取 rotation_type → 标准化 need_rotate
    # 文件路径: <output_base_folder>/clip_manifest.json
    manifest_path = os.path.join(output_base_folder, "clip_manifest.json")
    rotation_type = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as mf:
                manifest = json.load(mf)
                rotation_type = manifest.get('rotation_type')
        except Exception as e:
            print(f"[WARN] 读取clip_manifest失败: {str(e)}")
    
    # 以 rotation_type 为准，不再用宽高比推断
    if need_rotate is None:
        if rotation_type in ("clockwise", "counterclockwise", "rotate180"):
            need_rotate = True
        else:
            need_rotate = False
    print(f"[INFO] 辅助线方向: rotation_type={rotation_type}, need_rotate={need_rotate}")
    
    # 创建正确的辅助线生成器
    if line_type == "stance":
        line_generator = StanceAssistantLine(video_width=video_width, video_height=video_height)
        output_subfolder = "stand"
    elif line_type == "height":
        line_generator = HeightAssistantLine(video_width=video_width, video_height=video_height)
        output_subfolder = "k"
    elif line_type == "body":
        line_generator = BodyAssistantLine(video_width=video_width, video_height=video_height)
        output_subfolder = "body"
    elif line_type == "skeleton":
        line_generator = SkeletonAssistantLine(video_width=video_width, video_height=video_height)
        output_subfolder = "skeleton"
    elif line_type == "fishbone":
        line_generator = FishboneAssistantLine(video_width=video_width, video_height=video_height)
        output_subfolder = "fishbone"
    elif line_type == "shoulder_rotation":
        line_generator = ShoulderRotationAssistantLine(video_width=video_width, video_height=video_height)
        output_subfolder = "shoulder_rotation"
    elif line_type == "hip_rotation":
        line_generator = HipRotationAssistantLine(video_width=video_width, video_height=video_height)
        output_subfolder = "hip_rotation"
    else:
        print(f"[WARN] 未知的辅助线类型: {line_type}")
        return []
    
    # 创建输出目录
    output_folder = os.path.join(output_base_folder, "img", output_subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    # 确保keypoint_data长度与视频帧数匹配
    if len(keypoint_data) < total_frames:
        print(f"[WARN] 关键点数据长度({len(keypoint_data)})小于视频帧数({total_frames})")
        total_frames = len(keypoint_data)
    
    # 找到"准备阶段"的第一个关键帧
    first_key_frame = 0
    
    # 获取帧尺寸
    frame_size = (video_height, video_width)
    
    # 生成每一帧的透明辅助线图片
    generated_paths = []
    batch_images = []
    batch_paths = []
    
    # 对于需要实时更新的线型，预计算所有关键帧的线条信息
    if line_type == "skeleton" or line_type == "fishbone" or line_type == "shoulder_rotation" or line_type == "hip_rotation":
        # 不需要特殊预计算，每帧都会重新计算
        lines_info_cache = None
    else:
        # 对于静态线型，只计算一次线条信息
        cache_key = f"{line_type}_{first_key_frame}"
        lines_info_cache = image_cache.get(cache_key)
        
        if lines_info_cache is None:
            # 如果缓存中没有，计算线条信息并缓存
            lines_info_cache = line_generator.calculate_lines_info(frame_size, keypoint_data[first_key_frame])
            image_cache.set(cache_key, lines_info_cache)
            print(f"[INFO] 已为第 {first_key_frame} 帧计算辅助线信息，所有帧将使用相同的辅助线")
    
    # 使用批处理优化I/O操作
    process_time = 0
    save_time = 0
    
    for frame_idx in range(total_frames):
        # 计时开始 - 处理
        proc_start = time.time()
        
        # 如果是实时更新线型，则使用当前帧的关键点数据（实时变化）
        if line_type == "skeleton" or line_type == "fishbone" or line_type == "shoulder_rotation" or line_type == "hip_rotation":
            # 检查缓存
            cache_key = f"{line_type}_{frame_idx}"
            lines_info = image_cache.get(cache_key)
            
            if lines_info is None:
                # 为每一帧计算新的线条信息
                lines_info = line_generator.calculate_lines_info(frame_size, keypoint_data[frame_idx])
                
                # 缓存计算结果
                if USE_IMAGE_CACHE:
                    image_cache.set(cache_key, lines_info)
                    
            if frame_idx % 100 == 0:
                if line_type == "skeleton":
                    print(f"\r[INFO] 生成骨架线帧 {frame_idx+1}/{total_frames}", end="")
                elif line_type == "fishbone":
                    print(f"\r[INFO] 生成鱼骨线帧 {frame_idx+1}/{total_frames}", end="")
                elif line_type == "shoulder_rotation":
                    print(f"\r[INFO] 生成肩膀旋转角度线帧 {frame_idx+1}/{total_frames}", end="")
                elif line_type == "hip_rotation":
                    print(f"\r[INFO] 生成髋部旋转角度线帧 {frame_idx+1}/{total_frames}", end="")
        else:
            # 静态线型使用缓存的线条信息
            lines_info = lines_info_cache
        
        # 生成透明图片
        transparent_img = line_generator.generate_transparent_image(frame_size, lines_info)
        
        # 计时结束 - 处理
        process_time += time.time() - proc_start
        
        # 计时开始 - 保存
        save_start = time.time()
        
        # 如果需要旋转
        if need_rotate:
            transparent_img = cv2.rotate(transparent_img, cv2.ROTATE_90_CLOCKWISE)
        
        # 保存PNG图片
        out_name = f"assist_{frame_idx:04d}.png"
        out_path = os.path.join(output_folder, out_name)
        
        # 添加到批处理
        batch_images.append(transparent_img)
        batch_paths.append(out_path)
        generated_paths.append(out_path)
        
        # 当达到批处理大小或最后一帧时保存
        if len(batch_images) >= BATCH_SAVE_SIZE or frame_idx == total_frames - 1:
            try:
                # 批量保存图像
                batch_save_transparent_pngs(batch_images, batch_paths)
                
                # 检查文件是否成功写入
                for path in batch_paths:
                    if not os.path.exists(path):
                        print(f"[WARN] 文件未写入: {path}")
                        if path in generated_paths:
                            generated_paths.remove(path)
                    elif os.path.getsize(path) == 0:
                        print(f"[WARN] 文件大小为0: {path}")
                        os.remove(path)  # 删除空文件
                        if path in generated_paths:
                            generated_paths.remove(path)
                
                # 清空批处理
                batch_images = []
                batch_paths = []
            except Exception as e:
                print(f"[WARN] 保存PNG图片时出错: {str(e)}")
        
        # 计时结束 - 保存
        save_time += time.time() - save_start
        
        # 每100帧打印一次进度（非骨架线和鱼骨线类型）
        if line_type != "skeleton" and line_type != "fishbone" and frame_idx % 100 == 0 and frame_idx > 0:
            print(f"[INFO] 已生成 {frame_idx}/{total_frames} 帧透明辅助线图片")
    
    if line_type == "skeleton" or line_type == "fishbone":
        print("")  # 换行，完成进度显示
        
    # 打印性能统计
    total_time = time.time() - start_time
    print(f"[INFO] 完成{line_type}辅助线透明图片生成 => {output_folder}, 共 {len(generated_paths)} 帧")
    print(f"[INFO] 性能统计: 总耗时 {total_time:.2f}秒, 处理耗时 {process_time:.2f}秒, 保存耗时 {save_time:.2f}秒")
    print(f"[INFO] 每帧平均: 处理 {process_time/total_frames*1000:.2f}毫秒, 保存 {save_time/total_frames*1000:.2f}毫秒")
    
    if len(generated_paths) == 0:
        print(f"[WARN] 未生成任何PNG图片，检查目录权限和空间: {output_folder}")
        # 尝试写入一个测试文件
        test_path = os.path.join(output_folder, "test.txt")
        try:
            with open(test_path, "w") as f:
                f.write("Test write permission")
            print(f"[INFO] 测试文件写入成功: {test_path}")
            os.remove(test_path)
        except Exception as test_err:
            print(f"[WARN] 测试文件写入失败: {str(test_err)}")
    
    return generated_paths


# 将嵌套函数移到全局作用域
def clean_old_files(session_folder, line_type, folder_name):
    """清理旧的辅助线PNG文件"""
    folder_path = os.path.join(session_folder, "img", folder_name)
    if os.path.exists(folder_path):
        try:
            old_pngs = glob.glob(os.path.join(folder_path, "*.png"))
            for png in old_pngs:
                os.remove(png)
            print(f"[INFO] 已删除 {len(old_pngs)} 个旧的{line_type}PNG文件")
        except Exception as e:
            print(f"[WARN] 删除{line_type}旧文件时出错: {str(e)}")

# 修改并行处理函数，为Linux环境优化
def generate_line_for_mp(task_params):
    """
    为多进程设计的辅助线生成函数，接收任务参数元组，返回(线类型, 生成路径列表)元组
    """
    # 解包任务参数
    line_type = task_params["line_type"]
    line_folder = task_params["line_folder"]
    video_path = task_params["video_path"]
    keypoint_data_path = task_params["keypoint_data_path"]
    video_width = task_params["video_width"] 
    video_height = task_params["video_height"]
    
    try:
        # 加载关键点数据
        keypoint_data = torch.load(keypoint_data_path)
        
        # 生成透明辅助线
        print(f"[INFO] 开始生成{line_type}辅助线...")
        output_base_folder = os.path.dirname(os.path.dirname(line_folder))
        
        # 调用生成函数
        generated_paths = generate_transparent_assistant_lines(
            video_path=video_path,
            keypoint_data=keypoint_data,
            output_base_folder=output_base_folder,
            line_type=line_type,
            video_width=video_width,
            video_height=video_height
        )
        
        print(f"[INFO] 已生成{line_type}辅助线: {len(generated_paths)}帧")
        return (line_type, generated_paths)
    except Exception as e:
        print(f"[ERROR] 生成{line_type}辅助线时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return (line_type, [])

def is_linux_platform():
    """检查是否为Linux平台"""
    import platform
    return platform.system() == "Linux"

def get_optimal_workers():
    """获取最优的工作进程数"""
    cpu_count = os.cpu_count() or 4
    if is_linux_platform():
        # Linux下尽量利用多核心
        return max(1, cpu_count - 1)
    else:
        # Windows下保守设置
        return min(4, max(1, cpu_count // 2))

def generate_all_assistant_lines(session_id: str, base_folder: str = "./resultData"):
    """
    为指定会话生成所有类型的辅助线透明图片
    """
    # 启用性能计时
    import time
    total_start_time = time.time()
    
    # 设置最优并行工作进程数
    global MAX_WORKERS
    original_workers = MAX_WORKERS
    MAX_WORKERS = get_optimal_workers()
    
    # 根据平台动态调整并行策略
    global ENABLE_PARALLEL
    original_parallel = ENABLE_PARALLEL
    ENABLE_PARALLEL = ENABLE_PARALLEL and (is_linux_platform() or platform.system() != "Windows")
    
    print(f"[INFO] 平台: {platform.system()}, 启用并行: {ENABLE_PARALLEL}, 工作进程数: {MAX_WORKERS}")
    
    try:
        # 设置路径
        session_folder = os.path.join(base_folder, session_id)
        video_folder = os.path.join(session_folder, "video")
        original_video_path = os.path.join(video_folder, "original.mp4")
        
        if not os.path.exists(original_video_path):
            print(f"[ERR] 找不到原始视频文件: {original_video_path}")
            return {"error": f"找不到原始视频文件: {original_video_path}"}    
                # 加载关键点数据
        keypoint_data_path = os.path.join(session_folder, "keypoints.pt")
        if not os.path.exists(keypoint_data_path):
            print(f"[ERR] 找不到关键点数据: {keypoint_data_path}")
            return {"error": f"找不到关键点数据: {keypoint_data_path}"}     
        keypoint_load_start = time.time()
        keypoint_data = torch.load(keypoint_data_path)
        keypoint_load_time = time.time() - keypoint_load_start
        print(f"[INFO] 加载关键点数据耗时: {keypoint_load_time:.2f}秒，关键点数量: {keypoint_data.shape[0]}")
        # 加载阶段索引数据
        stage_indices_path = os.path.join(session_folder, "stage_indices.json")
        stage_indices = None
        if os.path.exists(stage_indices_path):
            try:
                with open(stage_indices_path, 'r') as f:
                    stage_indices = json.load(f)
                print(f"[INFO] 已加载阶段索引数据: {stage_indices_path}")
            except Exception as e:
                print(f"[WARN] 无法加载阶段索引数据: {str(e)}")
        else:
            print(f"[WARN] 找不到阶段索引数据: {stage_indices_path}，将使用第一帧作为参考")
    
        # 获取原始视频图像数量，确保辅助线数量匹配
        img_all_dir = os.path.join(session_folder, "img", "all")
        actual_frame_count = 0
        if os.path.exists(img_all_dir):
            image_files = [f for f in os.listdir(img_all_dir) if f.endswith('.jpg')]
            
            # 对图像文件名进行解析并排序
            try:
                # 解析帧号并排序
                frame_numbers = [int(''.join(filter(str.isdigit, os.path.basename(f)))) for f in image_files]
                actual_frame_count = len(image_files)
                max_frame_number = max(frame_numbers) if frame_numbers else 0
                
                print(f"[INFO] 检测到实际处理的视频帧数: {actual_frame_count}，最大帧号: {max_frame_number}")
            except Exception as e:
                print(f"[WARN] 解析帧号出错: {str(e)}，使用文件数量作为实际帧数")
                actual_frame_count = len(image_files)
                print(f"[INFO] 检测到实际处理的视频帧数: {actual_frame_count}")
            
            # 如果关键点数据比实际帧数多，截取关键点数据
            if keypoint_data.shape[0] > actual_frame_count:
                keypoint_data = keypoint_data[:actual_frame_count]
                print(f"[INFO] 截取关键点数据以匹配实际帧数: {keypoint_data.shape[0]}")
                
            # 打印剪切帧信息
            if actual_frame_count < keypoint_data.shape[0]:
                print(f"[INFO] 辅助线生成将使用图像帧数量 {actual_frame_count} 而不是关键点数量 {keypoint_data.shape[0]}")
    
        # 构建线类型配置
        line_configs = {
            "stance": {"class": StanceAssistantLine, "folder": os.path.join("img", "stand"), "desc": "站姿线"},
            "height": {"class": HeightAssistantLine, "folder": os.path.join("img", "k"), "desc": "高度线"},
            "body": {"class": BodyAssistantLine, "folder": os.path.join("img", "body"), "desc": "躯干线"},
            "skeleton": {"class": SkeletonAssistantLine, "folder": os.path.join("img", "skeleton"), "desc": "骨架线"},
            "fishbone": {"class": FishboneAssistantLine, "folder": os.path.join("img", "fishbone"), "desc": "鱼骨线"},
            "shoulder_rotation": {"class": ShoulderRotationAssistantLine, "folder": os.path.join("img", "shoulder_rotation"), "desc": "肩膀旋转角度线"},
            "hip_rotation": {"class": HipRotationAssistantLine, "folder": os.path.join("img", "hip_rotation"), "desc": "髋部旋转角度线"}
        }
        
        # 选择要生成的线类型
        # line_types_to_generate = ["stance"]  # 默认只生成站姿线
        line_types_to_generate = ["stance", "height", "body", "skeleton", "fishbone", "shoulder_rotation", "hip_rotation"]  # 生成所有类型的辅助线
        
        # 清理所有类型的旧文件
        for line_type, config in line_configs.items():
            folder_path = os.path.join(session_folder, config["folder"])
            if os.path.exists(folder_path):
                try:
                    files = glob.glob(os.path.join(folder_path, "*.png"))
                    for f in files:
                        os.remove(f)
                    print(f"[INFO] 已清理旧的{config['desc']}文件: {len(files)}个")
                except Exception as e:
                            print(f"[WARN] 清理{config['desc']}文件夹时出错: {str(e)}")
            else:
                # 创建文件夹
                os.makedirs(folder_path, exist_ok=True)
                print(f"[INFO] 已创建{config['desc']}文件夹: {folder_path}")
        
        # 读取原视频获取尺寸参数
        cap = cv2.VideoCapture(original_video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"[INFO] 视频尺寸: {video_width}x{video_height}")
        
        # 准备任务参数
        tasks = []
        # 只处理指定的线类型
        for line_type in line_types_to_generate:
            if line_type in line_configs:
                config = line_configs[line_type]
                task_params = {
        "session_id": session_id,
                    "base_folder": base_folder,
                    "line_type": line_type,
                    "line_folder": os.path.join(session_folder, config["folder"]),
                    "video_path": original_video_path,
                    "keypoint_data_path": keypoint_data_path,
                    "video_width": video_width,
                    "video_height": video_height
                }
                tasks.append(task_params)
            else:
                print(f"[WARN] 未知的辅助线类型: {line_type}")
        
        # 设置返回结果字典
        result_dict = {}
        
        # 根据配置选择并行或顺序处理
        if ENABLE_PARALLEL and len(tasks) > 1:
            print(f"[INFO] 使用{MAX_WORKERS}个进程并行生成辅助线")
            
            # Linux环境，使用fork模式
            if is_linux_platform():
                import multiprocessing as mp
                ctx = mp.get_context("fork")
                
                # 优化共享内存使用，仅适用于Linux
                global USE_SHARED_MEMORY
                original_shared_memory = USE_SHARED_MEMORY
                USE_SHARED_MEMORY = True
                
                try:
                    # 使用Linux优化的进程池实现
                    with ctx.Pool(MAX_WORKERS) as pool:
                        # 使用chunksize=1确保每个任务分配到一个进程
                        proc_results = pool.map(generate_line_for_mp, tasks, chunksize=1)
                        
                        # 收集处理结果
                        for line_type, paths in proc_results:
                            folder_key = f"{line_type}_line_folder"
                            count_key = f"{line_type}_line_count"
                            result_dict[folder_key] = os.path.join(session_folder, line_configs[line_type]["folder"])
                            result_dict[count_key] = len(paths)
                            
                except Exception as e:
                    print(f"[WARN] Linux并行处理失败，切换到顺序处理: {str(e)}")
                    # 如果并行失败，尝试顺序处理
                    for task in tasks:
                        line_type, paths = generate_line_for_mp(task)
                        folder_key = f"{line_type}_line_folder"
                        count_key = f"{line_type}_line_count"
                        result_dict[folder_key] = os.path.join(session_folder, line_configs[line_type]["folder"])
                        result_dict[count_key] = len(paths)
                
                # 恢复设置
                USE_SHARED_MEMORY = original_shared_memory
            
            # 非Linux环境，使用标准Pool
            else:
                # 优化序列化
                global USE_PICKLE_PROTOCOL_HIGHEST
                original_pickle = USE_PICKLE_PROTOCOL_HIGHEST
                USE_PICKLE_PROTOCOL_HIGHEST = True
                
                try:
                    with Pool(MAX_WORKERS) as pool:
                        proc_results = pool.map(generate_line_for_mp, tasks, chunksize=1)
                        
                        # 收集处理结果
                        for line_type, paths in proc_results:
                            folder_key = f"{line_type}_line_folder"
                            count_key = f"{line_type}_line_count"
                            result_dict[folder_key] = os.path.join(session_folder, line_configs[line_type]["folder"])
                            result_dict[count_key] = len(paths)
                except Exception as e:
                    print(f"[WARN] 并行处理失败，切换到顺序处理: {str(e)}")
                    # 如果并行失败，尝试顺序处理
                    for task in tasks:
                        line_type, paths = generate_line_for_mp(task)
                        folder_key = f"{line_type}_line_folder"
                        count_key = f"{line_type}_line_count"
                        result_dict[folder_key] = os.path.join(session_folder, line_configs[line_type]["folder"])
                        result_dict[count_key] = len(paths)
                
                # 恢复设置
                USE_PICKLE_PROTOCOL_HIGHEST = original_pickle
        else:
            # 顺序处理
            print("[INFO] 使用单进程顺序处理所有辅助线")
            for task in tasks:
                line_type, paths = generate_line_for_mp(task)
                folder_key = f"{line_type}_line_folder"
                count_key = f"{line_type}_line_count"
                result_dict[folder_key] = os.path.join(session_folder, line_configs[line_type]["folder"])
                result_dict[count_key] = len(paths)
        
        # 添加总处理时间
        total_time = time.time() - total_start_time
        result_dict["total_time"] = f"{total_time:.2f}秒"
        
        # 恢复全局设置
        MAX_WORKERS = original_workers
        ENABLE_PARALLEL = original_parallel
        
        # 执行垃圾回收
        gc.collect()
        
        print(f"[INFO] 所有辅助线生成完成，总耗时: {total_time:.2f}秒")
        return result_dict
    
    except Exception as e:
        print(f"[ERR] 生成辅助线失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 恢复全局设置
        MAX_WORKERS = original_workers
        ENABLE_PARALLEL = original_parallel
        
        # 执行垃圾回收
        gc.collect()
        
        return {"error": f"生成辅助线失败: {str(e)}"}

# 预分配缓冲区类，用于复用内存
class BufferPool:
    """内存缓冲区池，用于减少内存分配"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BufferPool, cls).__new__(cls)
            cls._instance._buffers = {}
        return cls._instance
    
    def get_buffer(self, shape, dtype=np.uint8):
        """获取指定大小的缓冲区"""
        if not USE_MEMORY_OPTIMIZATION:
            return np.zeros(shape, dtype=dtype)
            
        key = (shape, dtype)
        if key not in self._buffers:
            self._buffers[key] = np.zeros(shape, dtype=dtype)
        return self._buffers[key].copy()
    
    def clear(self):
        """清空缓冲区池"""
        self._buffers.clear()

# 创建全局缓冲区实例
buffer_pool = BufferPool()

# 图像缓存类，用于减少重复渲染
class ImageCache:
    """图像缓存，用于存储常用图像"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageCache, cls).__new__(cls)
            cls._instance.cache = {}
            cls._instance.max_size = IMAGE_CACHE_SIZE
        return cls._instance
    
    def get(self, key):
        """获取缓存的图像"""
        if not USE_IMAGE_CACHE:
            return None
        return self.cache.get(key)
    
    def set(self, key, image):
        """设置缓存图像"""
        if not USE_IMAGE_CACHE:
            return
            
        # 如果缓存已满，删除最早的条目
        if len(self.cache) >= self.max_size:
            # 删除第一个键（最早加入的）
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        # 添加新图像到缓存
        self.cache[key] = image
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()

# 创建全局图像缓存实例
image_cache = ImageCache()

# 批量保存PNG图像
def batch_save_transparent_pngs(image_batch, path_batch):
    """
    批量保存透明PNG图像，减少I/O操作
    
    参数:
        image_batch: BGRA图像列表
        path_batch: 对应的保存路径列表
    """
    if not image_batch:
        return
        
    for img, path in zip(image_batch, path_batch):
        # 使用PIL保存透明PNG图片
        # OpenCV使用BGRA格式，而PIL使用RGBA，需要转换
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        pil_img.save(path, format='PNG')
        
    # 设置文件权限
    try:
        import stat
        for path in path_batch:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    except Exception as e:
        print(f"[WARN] 设置文件权限时出错: {str(e)}")

# 创建共享内存数组（Linux平台优化）
def create_shared_array(shape, dtype=np.float32):
    """
    创建共享内存数组，提高进程间通信效率
    
    参数:
        shape: 数组形状
        dtype: 数据类型
        
    返回:
        共享内存数组
    """
    if not USE_SHARED_MEMORY or not is_linux_platform():
        return np.zeros(shape, dtype=dtype)
        
    try:
        # Linux环境下使用SharedMemory
        import multiprocessing as mp
        from multiprocessing import shared_memory
        
        # 计算数组大小
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        
        # 检查是否超过最大限制
        if nbytes > MAX_SHARED_MEMORY_SIZE:
            print(f"[WARN] 共享内存大小({nbytes/1024/1024:.1f}MB)超过限制({MAX_SHARED_MEMORY_SIZE/1024/1024:.1f}MB)，使用普通数组")
            return np.zeros(shape, dtype=dtype)
            
        # 创建共享内存
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        
        # 创建使用共享内存的ndarray
        shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        
        # 清零
        shared_array.fill(0)
        
        # 返回数组和共享内存对象
        return shared_array, shm
    except ImportError:
        print("[WARN] 共享内存模块不可用，使用普通数组")
        return np.zeros(shape, dtype=dtype)
    except Exception as e:
        print(f"[WARN] 创建共享内存失败: {str(e)}, 使用普通数组")
        return np.zeros(shape, dtype=dtype)

# 自定义序列化和反序列化函数（优化进程间通信）
def serialize_data(data):
    """高效序列化数据"""
    import pickle
    protocol = pickle.HIGHEST_PROTOCOL if USE_PICKLE_PROTOCOL_HIGHEST else pickle.DEFAULT_PROTOCOL
    return pickle.dumps(data, protocol=protocol)

def deserialize_data(serialized_data):
    """高效反序列化数据"""
    import pickle
    return pickle.loads(serialized_data)

def test_assistant_lines():
    """
    测试辅助线生成功能
    """
    import time
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="辅助线生成测试工具")
    parser.add_argument("--session_id", type=str, default="WeChat_20250214160750", help="会话ID，默认为WeChat_20250214160750")
    parser.add_argument("--base_folder", type=str, default="./resultData", help="基础数据目录")
    parser.add_argument("--line_type", type=str, default="stance", 
                       choices=["stance", "height", "body", "skeleton", "fishbone", "shoulder_rotation", "hip_rotation", "all"], 
                       help="要测试的辅助线类型")
    args = parser.parse_args()
    
    # 启用内存优化
    set_memory_optimization(True)
    
    # 设置图像缓存
    set_image_cache_enabled(True)
    
    # 辅助线类型 -> 文件夹名称映射
    folder_mapping = {
        "stance": "stand",
        "height": "k",
        "body": "body",
        "skeleton": "skeleton",
        "fishbone": "fishbone",
        "shoulder_rotation": "shoulder_rotation",
        "hip_rotation": "hip_rotation"
    }
    
    print(f"[INFO] 开始测试{args.line_type}辅助线生成")
    
    if args.line_type == "all":
        # 测试生成所有类型辅助线
        start_time = time.time()
        result = generate_all_assistant_lines(args.session_id, args.base_folder)
        total_time = time.time() - start_time
        
        print(f"[INFO] 生成所有辅助线完成，总耗时: {total_time:.2f}秒")
        print(f"[INFO] 结果: {result}")
    else:
        # 测试生成单一类型辅助线
        session_folder = os.path.join(args.base_folder, args.session_id)
        video_folder = os.path.join(session_folder, "video")
        video_path = os.path.join(video_folder, "original.mp4")
        keypoint_path = os.path.join(session_folder, "keypoints.pt")
        
        if not os.path.exists(video_path):
            print(f"[ERR] 找不到视频文件: {video_path}")
            return
        if not os.path.exists(keypoint_path):
            print(f"[ERR] 找不到关键点数据: {keypoint_path}")
            return
        
        # 加载关键点数据
        print(f"[INFO] 加载关键点数据: {keypoint_path}")
        keypoint_data = torch.load(keypoint_path)
        
        # 清理旧文件
        output_folder = os.path.join(session_folder, "img", folder_mapping[args.line_type])
        clean_old_files(session_folder, args.line_type, folder_mapping[args.line_type])
        
        print(f"[INFO] 开始生成{args.line_type}辅助线到: {output_folder}")
        
        start_time = time.time()
        paths = generate_transparent_assistant_lines(
            line_type=args.line_type,
            video_path=video_path,
        keypoint_data=keypoint_data,
            output_base_folder=session_folder
        )
        total_time = time.time() - start_time
        
        print(f"[INFO] 生成{args.line_type}辅助线完成，共{len(paths)}帧，耗时: {total_time:.2f}秒")
    
    # 清理缓存
    buffer_pool.clear()
    image_cache.clear()
    gc.collect()
    
    print("[INFO] 测试完成")


if __name__ == "__main__":
    test_assistant_lines()

