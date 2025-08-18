# swing_analyzer.py
# -------------------
# 存放 SwingAnalyzer 类及其相关逻辑。

import math
import torch
import numpy as np
from typing import Dict, List, Tuple
from config import KEYPOINT_WEIGHTS, STAGE_MAP
from keypoint_processor import KeypointProcessor
from utils import angle_difference

class SwingAnalyzer:
    """高尔夫挥杆分析器（专业教练定制的阶段识别与对比逻辑）"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def _get_stage_weights(self, stage_name: str) -> dict:
        base = KEYPOINT_WEIGHTS.get('default', {}).copy()
        if stage_name in KEYPOINT_WEIGHTS:
            base.update(KEYPOINT_WEIGHTS[stage_name])
        return base

    def _create_weight_matrix(self, stage: int) -> torch.Tensor:
        stage_name = STAGE_MAP[stage]
        weights_config = self._get_stage_weights(stage_name)
        masks = KeypointProcessor.get_region_masks()

        weight_matrix = torch.ones(33, 2)
        for region, mask in masks.items():
            if region in weights_config:
                weight_matrix[mask] *= weights_config[region]
        return weight_matrix

    @staticmethod
    def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cosine_angle))
        cross = np.cross(ba, bc)
        return angle if cross > 0 else -angle

    @staticmethod
    def compute_pelvic_rotation(frame_np: np.ndarray, need_rotate: str = "none") -> float:
        L_HIP = KeypointProcessor.L_HIP
        R_HIP = KeypointProcessor.R_HIP
        L_SHOULDER = KeypointProcessor.L_SHOULDER
        R_SHOULDER = KeypointProcessor.R_SHOULDER
        
        # 如果需要旋转坐标系
        if need_rotate != "none":
            # 创建旋转后的帧数据
            rotated_frame = frame_np.copy()
            # 交换x和y坐标
            rotated_frame[:, 0], rotated_frame[:, 1] = frame_np[:, 1].copy(), frame_np[:, 0].copy()
            
            if need_rotate == "clockwise":
                # 顺时针旋转，翻转y坐标
                rotated_frame[:, 1] = 1.0 - rotated_frame[:, 1]
            else:  # counterclockwise
                # 逆时针旋转，翻转x坐标
                rotated_frame[:, 0] = 1.0 - rotated_frame[:, 0]
                
            frame_np = rotated_frame
        
        hip_center = (frame_np[L_HIP] + frame_np[R_HIP]) / 2.0
        shoulder_center = (frame_np[L_SHOULDER] + frame_np[R_SHOULDER]) / 2.0
        torso_vector = shoulder_center - hip_center
        left_proj = np.dot(frame_np[L_HIP] - hip_center, torso_vector)
        right_proj = np.dot(frame_np[R_HIP] - hip_center, torso_vector)
        return math.degrees(math.atan2(right_proj - left_proj, np.linalg.norm(torso_vector)))

    @staticmethod
    def compute_shoulder_tilt(frame_np: np.ndarray, need_rotate: str = "none") -> float:
        R_SHOULDER = KeypointProcessor.R_SHOULDER
        L_SHOULDER = KeypointProcessor.L_SHOULDER
        
        # 如果需要旋转坐标系
        if need_rotate != "none":
            # 创建旋转后的帧数据
            rotated_frame = frame_np.copy()
            # 交换x和y坐标
            rotated_frame[:, 0], rotated_frame[:, 1] = frame_np[:, 1].copy(), frame_np[:, 0].copy()
            
            if need_rotate == "clockwise":
                # 顺时针旋转，翻转y坐标
                rotated_frame[:, 1] = 1.0 - rotated_frame[:, 1]
            else:  # counterclockwise
                # 逆时针旋转，翻转x坐标
                rotated_frame[:, 0] = 1.0 - rotated_frame[:, 0]
                
            frame_np = rotated_frame
            
        return frame_np[R_SHOULDER][1] - frame_np[L_SHOULDER][1]

    @staticmethod
    def compute_spine_angle(frame_np: np.ndarray, need_rotate: str = "none") -> float:
        L_HIP = KeypointProcessor.L_HIP
        R_HIP = KeypointProcessor.R_HIP
        NECK = KeypointProcessor.NECK
        
        # 如果需要旋转坐标系
        if need_rotate != "none":
            # 创建旋转后的帧数据
            rotated_frame = frame_np.copy()
            # 交换x和y坐标
            rotated_frame[:, 0], rotated_frame[:, 1] = frame_np[:, 1].copy(), frame_np[:, 0].copy()
            
            if need_rotate == "clockwise":
                # 顺时针旋转，翻转y坐标
                rotated_frame[:, 1] = 1.0 - rotated_frame[:, 1]
            else:  # counterclockwise
                # 逆时针旋转，翻转x坐标
                rotated_frame[:, 0] = 1.0 - rotated_frame[:, 0]
                
            frame_np = rotated_frame
            
        hip_center = (frame_np[L_HIP] + frame_np[R_HIP]) / 2.0
        mid_spine = (frame_np[L_HIP] + frame_np[R_HIP] + frame_np[NECK]) / 3.0
        
        # 直接在这里计算角度，避免循环导入
        ba = hip_center - mid_spine
        bc = frame_np[NECK] - mid_spine
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cosine_angle))
        cross = np.cross(ba, bc)
        return angle if cross > 0 else -angle

    def compute_detailed_errors(self,
                                frame: torch.Tensor,
                                standard_pose: torch.Tensor,
                                standard_std: torch.Tensor,
                                standard_angle_stats: dict,
                                target_stage: int,
                                rotation_type: str = "none") -> dict:
        stage_name = STAGE_MAP[target_stage]
        weights_config = self._get_stage_weights(stage_name)
        angle_config = weights_config.get('angle_config', {})

        position_weight = weights_config.get('position_weight', 0.5)
        angle_weight = weights_config.get('angle_weight', 0.5)

        from keypoint_processor import KeypointProcessor
        masks = KeypointProcessor.get_region_masks()
        def mean_xy(tensor2d, indices):
            sub = tensor2d[indices]
            mx = float(sub[:,0].mean())
            my = float(sub[:,1].mean())
            return (mx, my)

        regions_info = {}
        frame_np = frame.cpu().numpy()
        std_np   = standard_pose.cpu().numpy()

        for region_name, region_mask in masks.items():
            video_pos = mean_xy(frame, region_mask)
            std_pos   = mean_xy(standard_pose, region_mask)
            sub_std   = standard_std[region_mask]
            var_val   = float((sub_std[:,0].mean() + sub_std[:,1].mean())/2.0)
            ex = video_pos[0] - std_pos[0]
            ey = video_pos[1] - std_pos[1]
            region_error = math.sqrt(ex*ex + ey*ey)

            regions_info[region_name] = {
                "video_position": video_pos,
                "std_position": std_pos,
                "error": region_error,
                "std_variance": var_val
            }

        angles_map = {}
        L_SHOULDER, L_ELBOW = KeypointProcessor.L_SHOULDER, KeypointProcessor.L_ELBOW
        R_SHOULDER, R_ELBOW = KeypointProcessor.R_SHOULDER, KeypointProcessor.R_ELBOW

        video_angles = {}
        video_angles["left_elbow"] = self.compute_angle(frame_np[L_SHOULDER], frame_np[L_ELBOW], frame_np[15])
        video_angles["right_elbow"] = self.compute_angle(frame_np[R_SHOULDER], frame_np[R_ELBOW], frame_np[16])
        video_angles["left_shoulder"] = self.compute_angle(frame_np[L_ELBOW], frame_np[L_SHOULDER], frame_np[KeypointProcessor.NECK])
        video_angles["right_shoulder"] = self.compute_angle(frame_np[R_ELBOW], frame_np[R_SHOULDER], frame_np[KeypointProcessor.NECK])
        video_angles["left_knee"] = self.compute_angle(frame_np[23], frame_np[25], frame_np[27])
        video_angles["right_knee"] = self.compute_angle(frame_np[24], frame_np[26], frame_np[28])
        video_angles["spine_angle"] = self.compute_spine_angle(frame_np, rotation_type)
        video_angles["pelvic_rotation"] = self.compute_pelvic_rotation(frame_np, rotation_type)
        video_angles["shoulder_tilt"] = self.compute_shoulder_tilt(frame_np, rotation_type)

        std_np_pose = standard_pose.cpu().numpy()
        std_angles = {}
        std_angles["left_elbow"] = self.compute_angle(std_np_pose[L_SHOULDER], std_np_pose[L_ELBOW], std_np_pose[15])
        std_angles["right_elbow"] = self.compute_angle(std_np_pose[R_SHOULDER], std_np_pose[R_ELBOW], std_np_pose[16])
        std_angles["left_shoulder"] = self.compute_angle(std_np_pose[L_ELBOW], std_np_pose[L_SHOULDER], std_np_pose[KeypointProcessor.NECK])
        std_angles["right_shoulder"] = self.compute_angle(std_np_pose[R_ELBOW], std_np_pose[R_SHOULDER], std_np_pose[KeypointProcessor.NECK])
        std_angles["left_knee"] = self.compute_angle(std_np_pose[23], std_np_pose[25], std_np_pose[27])
        std_angles["right_knee"] = self.compute_angle(std_np_pose[24], std_np_pose[26], std_np_pose[28])
        std_angles["spine_angle"] = self.compute_spine_angle(std_np_pose)
        std_angles["pelvic_rotation"] = self.compute_pelvic_rotation(std_np_pose)
        std_angles["shoulder_tilt"] = self.compute_shoulder_tilt(std_np_pose)

        angle_info_dict = {}
        for joint in video_angles.keys():
            vid_a = video_angles[joint]
            std_a = std_angles[joint]
            if joint in standard_angle_stats:
                var_val = standard_angle_stats[joint]["std"] or 1.0
            else:
                var_val = 1.0
            err_val = abs(angle_difference(vid_a, std_a))
            angle_info_dict[joint] = {
                "video_angle": vid_a,
                "std_angle": std_a,
                "error": err_val,
                "variance": var_val
            }

        return {
            "regions": regions_info,
            "position_weight": position_weight,
            "angle_weight": angle_weight,
            "angles": angle_info_dict
        }

    

    
    def _direct_selection(self, similarity_matrix, top_k, frame_count):
        """直接根据相似度选择每个阶段的top_k帧"""
        stages = {}
        for stage in STAGE_MAP.keys():
            # 获取与该阶段最相似的top_k帧
            stage_similarities = similarity_matrix[:, stage]
            if torch.max(stage_similarities) == -float('inf'):
                continue
                
            _, top_indices = torch.topk(stage_similarities, min(top_k, frame_count))
            top_indices = top_indices.tolist()
            
            if top_indices:
                stages[STAGE_MAP[stage]] = sorted(top_indices)
        
        # 应用一些简单的时序约束：确保阶段的顺序
        last_frame = -1
        for stage in sorted(STAGE_MAP.keys()):
            stage_key = STAGE_MAP[stage]
            if stage_key in stages and stages[stage_key]:
                # 确保当前阶段的帧都在上一阶段之后
                stages[stage_key] = [f for f in stages[stage_key] if f > last_frame]
                if stages[stage_key]:
                    last_frame = max(stages[stage_key])
        
        return stages


    def identify_swing_stages(self, X: torch.Tensor, top_k: int = 1) -> Tuple[Dict[str, List[int]], str]:
        """
        按照标准挥杆物理特性的阶段分类算法 - 从关键特征点开始识别
        
        Args:
            X: 关键点数据，形状为 (N, 66) 或 (N, 33, 2)
            top_k: 每个阶段返回的候选帧数量
            
        Returns:
            Tuple[Dict[str, List[int]], str]: 
                - 第一项：每个阶段的候选帧索引
                - 第二项：旋转类型 ("none", "clockwise" 或 "counterclockwise")
        """
        frame_count = X.shape[0]
        
        # 确保X的形状为 (N, 33, 2)
        if X.dim() == 2:
            X = X.view(frame_count, 33, 2)
        
        # 步骤0: 检测是否需要旋转关键点坐标（判断横屏还是竖屏）
        # 取第一帧的关键点范围判断视频方向
        first_frame = X[0]
        x_min, _ = torch.min(first_frame[:, 0], dim=0)
        x_max, _ = torch.max(first_frame[:, 0], dim=0)
        y_min, _ = torch.min(first_frame[:, 1], dim=0)
        y_max, _ = torch.max(first_frame[:, 1], dim=0)
        
        width = x_max - x_min
        height = y_max - y_min
        need_rotate = (width > height)
        rotation_type = "none"  # 默认不旋转
        
        # 如果需要旋转（横屏视频），智能判断人体朝向并选择正确的旋转方式
        if need_rotate:
            print(f"[INFO] 检测到横屏视频，对关键点坐标系进行旋转以正确识别挥杆阶段")
            
            # 智能判断人体朝向
            # 定义头部和脚部关键点索引
            head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 鼻子、眼、耳、嘴等
            foot_indices = [27, 28, 29, 30, 31, 32]  # 脚踝、脚跟、脚趾等
            
            # 计算头部和脚部的平均x坐标
            head_x_avg = torch.mean(first_frame[head_indices, 0])
            foot_x_avg = torch.mean(first_frame[foot_indices, 0])
            
            # 根据头部和脚部在x轴的相对位置判断朝向
            head_at_right = head_x_avg > foot_x_avg
            
            rotated_X = X.clone()
            # 交换x和y坐标
            rotated_X[:, :, 0], rotated_X[:, :, 1] = X[:, :, 1].clone(), X[:, :, 0].clone()
            
            if head_at_right:
                # 头部在右侧，需要顺时针旋转，翻转y坐标
                print(f"[INFO] 检测到头部在右侧，应用顺时针旋转")
                rotated_X[:, :, 0] = 1.0 - rotated_X[:, :, 0]
                rotation_type = "clockwise"
            else:
                # 头部在左侧，需要逆时针旋转，翻转x坐标
                print(f"[INFO] 检测到头部在左侧，应用逆时针旋转")
                rotated_X[:, :, 1] = 1.0 - rotated_X[:, :, 1]
                rotation_type = "counterclockwise"
            
            # 使用旋转后的坐标进行后续处理
            X = rotated_X
        
        # 步骤1: 对关键点数据进行平滑处理，减少噪点影响
        # 使用更小的窗口以避免过度平滑
        smoothed_X = temporal_smoothing(X, window_size=5)
        
        # 步骤2: 提取关键关节点索引
        LEFT_WRIST = KeypointProcessor.L_WRIST      # 左手腕
        RIGHT_WRIST = KeypointProcessor.R_WRIST     # 右手腕
        LEFT_SHOULDER = KeypointProcessor.L_SHOULDER  # 左肩
        RIGHT_SHOULDER = KeypointProcessor.R_SHOULDER  # 右肩
        LEFT_ELBOW = KeypointProcessor.L_ELBOW      # 左肘
        RIGHT_ELBOW = KeypointProcessor.R_ELBOW     # 右肘
        LEFT_HIP = KeypointProcessor.L_HIP          # 左髋
        RIGHT_HIP = KeypointProcessor.R_HIP         # 右髋
        NECK = KeypointProcessor.NECK               # 颈部
        
        # 步骤3: 计算关键特征序列
        # 手腕高度 (Y值越小表示位置越高)
        wrists_y = (smoothed_X[:, LEFT_WRIST, 1] + smoothed_X[:, RIGHT_WRIST, 1]) / 2
        
        # 减少对手腕Y坐标的额外平滑，使用较小的窗口和更保守的参数
        wrists_y_np = wrists_y.cpu().numpy()
        try:
            import scipy.signal
            # 减小窗口大小和多项式阶数，减少过度平滑
            wrists_y_filtered = torch.tensor(scipy.signal.savgol_filter(wrists_y_np, 7, 2), 
                                            device=wrists_y.device, dtype=wrists_y.dtype)
        except ImportError:
            # 同样减小窗口大小
            wrists_y_filtered = torch.tensor(savgol_filter1d(wrists_y_np, 7, 2), 
                                            device=wrists_y.device, dtype=wrists_y.dtype)
        
        # 为了避免过度平滑，对滤波结果和原始数据进行加权平均
        wrists_y = 0.7 * wrists_y_filtered + 0.3 * wrists_y
        
        # 手腕X坐标 (表示左右位置)
        wrists_x = (smoothed_X[:, LEFT_WRIST, 0] + smoothed_X[:, RIGHT_WRIST, 0]) / 2
        
        
        # 身体中心X坐标 (用于归一化，避免相机位置影响)
        # 结合髋部、肩膀、脚和鼻子一起进行加权计算，提高定位准确性
        NOSE_IDX = 0  # 鼻子关键点索引
        FOOT_INDICES = [27, 28, 29, 30]  # 脚踝和脚部关键点索引
        
        # 计算髋部中心
        hip_center_x = (smoothed_X[:, LEFT_HIP, 0] + smoothed_X[:, RIGHT_HIP, 0]) / 2
        
        # 计算肩膀中心
        shoulder_center_x = (smoothed_X[:, LEFT_SHOULDER, 0] + smoothed_X[:, RIGHT_SHOULDER, 0]) / 2
        
        # 计算脚部中心
        foot_x = torch.zeros_like(hip_center_x)
        for foot_idx in FOOT_INDICES:
            foot_x += smoothed_X[:, foot_idx, 0]
        foot_center_x = foot_x / len(FOOT_INDICES)
        
        # 鼻子x坐标
        nose_x = smoothed_X[:, NOSE_IDX, 0]
        
        # 加权计算身体中心x坐标
        # 髋部(0.4) + 肩膀(0.3) + 脚(0.2) + 鼻子(0.1)
        body_center_x = (0.4 * hip_center_x + 
                        0.3 * shoulder_center_x + 
                        0.2 * foot_center_x + 
                        0.1 * nose_x)
        
        # 归一化手腕水平位置 (相对于身体中心)
        norm_wrists_x = wrists_x - body_center_x
        
        # 计算手腕高度变化率 (正值表示下降，负值表示上升)
        # 减小计算变化率的间隔，从5减到3，以便更精确地捕捉动作变化
        wrist_y_change = torch.zeros(frame_count)
        for i in range(3, frame_count):
            wrist_y_change[i] = wrists_y[i] - wrists_y[i-3]
        
        # 计算手臂平行度 (下方手臂的水平程度)
        arm_parallel = torch.zeros(frame_count)
        for i in range(frame_count):
            # 确定哪条是下方的手臂
            left_wrist_y = smoothed_X[i, LEFT_WRIST, 1].item()
            right_wrist_y = smoothed_X[i, RIGHT_WRIST, 1].item()
            
            if left_wrist_y > right_wrist_y:  # Y坐标更大表示位置更低
                # 左手臂是下方手臂
                elbow_x = smoothed_X[i, LEFT_ELBOW, 0].item()
                elbow_y = smoothed_X[i, LEFT_ELBOW, 1].item()
                wrist_x = smoothed_X[i, LEFT_WRIST, 0].item()
                wrist_y = smoothed_X[i, LEFT_WRIST, 1].item()
            else:
                # 右手臂是下方手臂
                elbow_x = smoothed_X[i, RIGHT_ELBOW, 0].item()
                elbow_y = smoothed_X[i, RIGHT_ELBOW, 1].item()
                wrist_x = smoothed_X[i, RIGHT_WRIST, 0].item()
                wrist_y = smoothed_X[i, RIGHT_WRIST, 1].item()
            
            # 计算与水平线的夹角，90度是垂直，0度是水平
            angle = abs(math.degrees(math.atan2(abs(wrist_y - elbow_y), abs(wrist_x - elbow_x))))
            # 转换为水平程度得分 (0最水平，90最垂直)
            arm_parallel[i] = angle
        
        # 初始化存储各阶段帧的字典
        stage_indices = {str(i): [] for i in range(9)}
        
        # 步骤4: 识别随杆阶段 (阶段6) - 手部与肩膀平齐，位于最右侧，手臂接近平行
        # 只在视频的后半部分搜索随杆阶段
        search_start = 0
        search_end = frame_count
        
        # 获取最右侧的前10个候选帧
        right_values, right_indices = torch.topk(norm_wrists_x[search_start:search_end], min(10, search_end-search_start))
        # 调整索引到原始帧序列
        right_indices = right_indices + search_start
        
        # 计算每个候选帧的得分，结合手部位置和手臂水平程度，但手部位置占主要比重
        right_candidates_scores = []
        for idx in right_indices:
            idx = idx.item()
            # 手部右侧位置得分（越大越好，需要取负数使得值越小越好）
            position_score = -norm_wrists_x[idx].item()  # 取负使得位置越右分数越低
            # 归一化位置分数（基于当前选择的候选帧）
            min_pos = -torch.max(norm_wrists_x[right_indices]).item()
            max_pos = -torch.min(norm_wrists_x[right_indices]).item()
            if max_pos > min_pos:
                norm_position_score = (position_score - min_pos) / (max_pos - min_pos)
            else:
                norm_position_score = 0
            
            # 手臂水平程度得分（越小越好）
            arm_horizontal_score = arm_parallel[idx]
            # 归一化手臂水平分数（0-90度）
            norm_arm_score = arm_horizontal_score / 90.0
            
            # 综合得分：位置占70%，手臂水平度占30%
            total_score = 0.7 * norm_position_score + 0.3 * norm_arm_score
            right_candidates_scores.append((idx, total_score))
            
        # 按得分排序（得分越低越好）
        right_candidates_scores.sort(key=lambda x: x[1])
        
        # 选择最好的候选帧作为随杆阶段
        right_most_candidates = [idx for idx, _ in right_candidates_scores]
        
        follow_through_idx = None
        if right_most_candidates:
            # 在得分最好的帧中，选择最右的那一个
            follow_through_idx = max(right_most_candidates[:3], key=lambda i: norm_wrists_x[i].item())
            
            # 直接使用按得分排序的前top_k个帧
            if top_k > 1:
                # 确保follow_through_idx在列表中的第一位
                candidates = [idx for idx, _ in right_candidates_scores]
                if follow_through_idx in candidates:
                    candidates.remove(follow_through_idx)
                selected_frames = [follow_through_idx] + candidates[:top_k-1]
                stage_indices["6"] = sorted(selected_frames)
            else:
                stage_indices["6"] = [follow_through_idx]
        else:
            # 如果找不到符合所有条件的帧，直接使用手腕最右的点
            follow_through_idx = right_indices[0].item()  # 取topk的第一个
            stage_indices["6"] = [follow_through_idx]
        
        # 步骤5: 识别收杆阶段 (阶段7) - 随杆之后手部高度最高的帧
        if follow_through_idx is not None and follow_through_idx < frame_count - 1:
            search_start = follow_through_idx + 1
            search_end = frame_count
            
            # 找到手腕高度最高的帧 (Y值最小)
            finish_idx = search_start + torch.argmin(wrists_y[search_start:search_end])
            stage_indices["7"] = [finish_idx.item()]
            
            # 添加相似帧
            if top_k > 1 and search_end > search_start + 1:
                window = min(10, (search_end - search_start) // 2)
                start = max(finish_idx.item() - window, search_start)
                end = min(finish_idx.item() + window, search_end)
                
                similar_frames = []
                for i in range(start, end):
                    if i == finish_idx.item():
                        continue
                    # 计算与收杆帧的相似度
                    similarity = abs(wrists_y[i] - wrists_y[finish_idx])
                    similar_frames.append((i, similarity.item()))
                
                # 选择相似度最高的帧
                similar_frames.sort(key=lambda x: x[1])
                additional_frames = [idx for idx, _ in similar_frames[:top_k-1]]
                stage_indices["7"].extend(additional_frames)
                stage_indices["7"].sort()
        else:
            # 如果找不到随杆阶段，使用视频末尾帧
            stage_indices["7"] = [max(0, frame_count - 5)]
        
        # 步骤6: 识别上杆和下杆阶段 - 手部与肩膀平齐，位于最左侧，根据手臂运动方向区分
        search_start = 0
        search_end = follow_through_idx if follow_through_idx is not None else int(frame_count * 0.7)
        
        # 获取最左侧的前10个候选帧
        left_values, left_indices = torch.topk(-norm_wrists_x[search_start:search_end], min(30, search_end-search_start))
        # 调整索引到原始帧序列
        left_indices = left_indices + search_start
        
        # 计算每个候选帧的得分，结合手部位置和手臂水平程度，但手部位置占主要比重
        left_candidates_scores = []
        for idx in left_indices:
            idx = idx.item()
            # 手部左侧位置得分（越小越好，已经是负数所以值越大越好，需再次取负）
            position_score = norm_wrists_x[idx].item()  # 值本身已经是负数，越小越左
            # 归一化位置分数（基于当前选择的候选帧）
            min_pos = torch.min(norm_wrists_x[left_indices]).item()
            max_pos = torch.max(norm_wrists_x[left_indices]).item()
            if max_pos > min_pos:
                norm_position_score = (position_score - min_pos) / (max_pos - min_pos)
            else:
                norm_position_score = 0
            
            # 手臂水平程度得分（越小越好）
            arm_horizontal_score = arm_parallel[idx]
            # 归一化手臂水平分数（0-90度）
            norm_arm_score = arm_horizontal_score / 90.0
            
            # 综合得分：位置占70%，手臂水平度占30%
            total_score = 0.7 * norm_position_score + 0.3 * norm_arm_score
            left_candidates_scores.append((idx, total_score))
            
        # 按得分排序（得分越低越好）
        left_candidates_scores.sort(key=lambda x: x[1])
        
        # 选择最好的候选帧
        left_best_candidates = [idx for idx, _ in left_candidates_scores]
        
        # 初始化上杆和下杆阶段帧列表
        backswing_frames = []
        downswing_frames = []
        
        # 在得分最好的候选帧中，判断每一帧的运动趋势，使用窗口平均而不是单点
        for idx in left_best_candidates:
            if idx > 5 and idx < frame_count - 5:
                # 使用窗口平均计算前后趋势
                window_size = 3  # 使用小窗口计算平均趋势
                before_window = wrists_y[max(0, idx - 5 - window_size + 1):idx - 5 + 1]
                current_window = wrists_y[max(0, idx - window_size + 1):idx + 1]
                after_window = wrists_y[idx + 5:min(frame_count, idx + 5 + window_size)]
                
                # 确保窗口不为空
                if len(before_window) > 0 and len(after_window) > 0:
                    before_avg = torch.mean(before_window)
                    current_avg = torch.mean(current_window)
                    after_avg = torch.mean(after_window)
                    
                    before_trend = before_avg - current_avg
                    after_trend = after_avg - current_avg
                    
                    # 放宽判断条件，增加判断力度门槛
                    threshold = 0.005  # 可调整的阈值，避免微小波动的影响
                    
                    # 上杆阶段 - 手部持续上升过程（Y值持续减小）
                    if before_trend > threshold and after_trend < -threshold:
                        backswing_frames.append(idx)
                    
                    # 下杆阶段 - 手部持续下降过程（Y值持续增大）
                    elif before_trend < -threshold and after_trend > threshold:
                        downswing_frames.append(idx)
                    
                    # 处理中间情况 - 当运动趋势不明显但位置最左
                    elif abs(before_trend) < threshold and abs(after_trend) < threshold:
                        # 选择最左的帧作为候选
                        if norm_wrists_x[idx].item() < -0.2:  # 可调整的阈值，选择明显偏左的帧
                            if len(backswing_frames) < 3:  # 限制添加的数量
                                backswing_frames.append(idx)
        
        # 如果某个阶段没有找到帧，扩大搜索范围
        search_more = False
        if not backswing_frames or not downswing_frames:
            search_more = True
            # 扩大搜索范围至20个候选帧
            more_values, more_indices = torch.topk(-norm_wrists_x[search_start:search_end], 
                                                  min(20, search_end-search_start))
            more_indices = more_indices + search_start
            
            # 只检查之前未检查过的帧
            extra_indices = [idx.item() for idx in more_indices if idx.item() not in left_best_candidates]
            
            for idx in extra_indices:
                if idx > 5 and idx < frame_count - 5:
                    # 使用窗口平均计算前后趋势
                    window_size = 3  # 使用小窗口计算平均趋势
                    before_window = wrists_y[max(0, idx - 5 - window_size + 1):idx - 5 + 1]
                    current_window = wrists_y[max(0, idx - window_size + 1):idx + 1]
                    after_window = wrists_y[idx + 5:min(frame_count, idx + 5 + window_size)]
                    
                    # 确保窗口不为空
                    if len(before_window) > 0 and len(after_window) > 0:
                        before_avg = torch.mean(before_window)
                        current_avg = torch.mean(current_window)
                        after_avg = torch.mean(after_window)
                        
                        before_trend = before_avg - current_avg
                        after_trend = after_avg - current_avg
                        
                        # 放宽判断条件，增加判断力度门槛
                        threshold = 0.005  # 可调整的阈值，避免微小波动的影响
                        
                        # 上杆阶段 - 手部持续上升过程（Y值持续减小）
                        if before_trend > threshold and after_trend < -threshold:
                            backswing_frames.append(idx)
                        # 下杆阶段 - 手部持续下降过程（Y值持续增大）
                        elif before_trend < -threshold and after_trend > threshold:
                            downswing_frames.append(idx)
                        
                        # 如果已经找到两种阶段，可以提前结束
                        if backswing_frames and downswing_frames:
                            break
        
        # 如果仍然找不到上杆或下杆帧，继续扩大搜索范围
        if search_more and (not backswing_frames or not downswing_frames):
            # 扩大搜索范围至40个候选帧
            more_values, more_indices = torch.topk(-norm_wrists_x[search_start:search_end], 
                                                 min(40, search_end-search_start))
            more_indices = more_indices + search_start
            
            # 只检查之前未检查过的帧
            checked_indices = left_best_candidates + [idx.item() for idx in more_indices[:20] if idx.item() not in left_best_candidates]
            extra_indices = [idx.item() for idx in more_indices if idx.item() not in checked_indices]
            
            for idx in extra_indices:
                if idx > 5 and idx < frame_count - 5:
                    # 使用窗口平均计算前后趋势
                    window_size = 3  # 使用小窗口计算平均趋势
                    before_window = wrists_y[max(0, idx - 5 - window_size + 1):idx - 5 + 1]
                    current_window = wrists_y[max(0, idx - window_size + 1):idx + 1]
                    after_window = wrists_y[idx + 5:min(frame_count, idx + 5 + window_size)]
                    
                    # 确保窗口不为空
                    if len(before_window) > 0 and len(after_window) > 0:
                        before_avg = torch.mean(before_window)
                        current_avg = torch.mean(current_window)
                        after_avg = torch.mean(after_window)
                        
                        before_trend = before_avg - current_avg
                        after_trend = after_avg - current_avg
                        
                        # 放宽判断条件，增加判断力度门槛
                        threshold = 0.005  # 可调整的阈值，避免微小波动的影响
                        
                        # 上杆阶段 - 手部持续上升过程（Y值持续减小）
                        if before_trend > threshold and after_trend < -threshold:
                            backswing_frames.append(idx)
                        # 下杆阶段 - 手部持续下降过程（Y值持续增大）
                        elif before_trend < -threshold and after_trend > threshold:
                            downswing_frames.append(idx)
                        
                        # 如果已经找到两种阶段，可以提前结束
                        if backswing_frames and downswing_frames:
                            break
        
        # 如果仍然找不到，搜索整个范围
        if search_more and (not backswing_frames or not downswing_frames):
            # 搜索整个区域内的所有帧
            remaining_indices = list(range(search_start, search_end))
            
            # 按照X坐标从左到右排序
            remaining_indices.sort(key=lambda i: norm_wrists_x[i].item())
            
            # 只检查之前未检查过的帧
            checked_indices = left_best_candidates + [idx.item() for idx in more_indices if idx.item() not in left_best_candidates]
            extra_indices = [idx for idx in remaining_indices if idx not in checked_indices]
            
            for idx in extra_indices:
                if idx > 5 and idx < frame_count - 5:
                    # 使用窗口平均计算前后趋势
                    window_size = 3  # 使用小窗口计算平均趋势
                    before_window = wrists_y[max(0, idx - 5 - window_size + 1):idx - 5 + 1]
                    current_window = wrists_y[max(0, idx - window_size + 1):idx + 1]
                    after_window = wrists_y[idx + 5:min(frame_count, idx + 5 + window_size)]
                    
                    # 确保窗口不为空
                    if len(before_window) > 0 and len(after_window) > 0:
                        before_avg = torch.mean(before_window)
                        current_avg = torch.mean(current_window)
                        after_avg = torch.mean(after_window)
                        
                        before_trend = before_avg - current_avg
                        after_trend = after_avg - current_avg
                        
                        # 放宽判断条件，增加判断力度门槛
                        threshold = 0.005  # 可调整的阈值，避免微小波动的影响
                        
                        # 上杆阶段 - 手部持续上升过程（Y值持续减小）
                        if before_trend > threshold and after_trend < -threshold:
                            backswing_frames.append(idx)
                        # 下杆阶段 - 手部持续下降过程（Y值持续增大）
                        elif before_trend < -threshold and after_trend > threshold:
                            downswing_frames.append(idx)
                        
                        # 如果已经找到两种阶段，可以提前结束
                        if backswing_frames and downswing_frames:
                            break
        
        # 如果仍然找不到上杆帧，按位置强制选择
        if not backswing_frames:
            mid_point = (search_start + search_end) // 2
            backswing_idx = search_start + (mid_point - search_start) // 2
            backswing_frames.append(backswing_idx)
        
        # 如果仍然没有找到下杆帧，按位置强制选择
        if not downswing_frames:
            mid_point = (search_start + search_end) // 2
            downswing_idx = mid_point + (search_end - mid_point) // 2
            downswing_frames.append(downswing_idx)
        
        # 为上杆阶段选择最好的帧
        # 优先选择运动趋势正确的帧，按照得分排序
        if len(backswing_frames) > 0:
            # 按得分排序
            backswing_with_scores = [(idx, next((score for i, score in left_candidates_scores if i == idx), float('inf'))) 
                                     for idx in backswing_frames]
            backswing_with_scores.sort(key=lambda x: x[1])  # 按得分排序
            
            # 选择得分最好的帧作为主帧
            backswing_idx = backswing_with_scores[0][0]
            stage_indices["2"] = [backswing_idx]  # 修正为阶段2（上杆阶段）
            
            # 如果需要多个帧
            if top_k > 1 and len(backswing_frames) > 1:
                additional_frames = [idx for idx, _ in backswing_with_scores[1:min(top_k, len(backswing_with_scores))]]
                stage_indices["2"].extend(additional_frames)
                stage_indices["2"].sort()
        
        # 为下杆阶段选择最好的帧
        if len(downswing_frames) > 0:
            # 按得分排序
            downswing_with_scores = [(idx, next((score for i, score in left_candidates_scores if i == idx), float('inf'))) 
                                     for idx in downswing_frames]
            downswing_with_scores.sort(key=lambda x: x[1])  # 按得分排序
            
            # 选择得分最好的帧作为主帧
            downswing_idx = downswing_with_scores[0][0]
            stage_indices["4"] = [downswing_idx]  # 下杆阶段改为阶段4
            
            # 如果需要多个帧
            if top_k > 1 and len(downswing_frames) > 1:
                additional_frames = [idx for idx, _ in downswing_with_scores[1:min(top_k, len(downswing_with_scores))]]
                stage_indices["4"].extend(additional_frames)
                stage_indices["4"].sort()
        
        # 步骤7: 识别击球阶段 (阶段5) - 下杆和随杆之间手部坐标最低点
        if stage_indices["4"] and stage_indices["6"]:
            downswing_idx = max(stage_indices["4"])
            follow_through_idx = min(stage_indices["6"])
            
            if follow_through_idx > downswing_idx:
                search_start = downswing_idx + 1
                search_end = follow_through_idx
                
                if search_end > search_start:
                    # 找到手腕高度最低点 (Y值最大)
                    impact_candidates = []
                    
                    # 计算每个候选帧的得分，结合手腕高度和与身体中心的水平对齐程度
                    for i in range(search_start, search_end):
                        # 手腕高度得分 (Y值越大越好)
                        height_score = wrists_y[i].item()
                        
                        # 双手平均x坐标与身体中心的距离 (越小越好)
                        wrists_center_x = (smoothed_X[i, LEFT_WRIST, 0] + smoothed_X[i, RIGHT_WRIST, 0]) / 2
                        
                        # 计算加权身体中心x坐标
                        hip_center_x = (smoothed_X[i, LEFT_HIP, 0] + smoothed_X[i, RIGHT_HIP, 0]) / 2
                        shoulder_center_x = (smoothed_X[i, LEFT_SHOULDER, 0] + smoothed_X[i, RIGHT_SHOULDER, 0]) / 2
                        
                        # 计算脚部中心
                        foot_x = 0
                        for foot_idx in [27, 28, 29, 30]:  # 脚踝和脚部关键点
                            foot_x += smoothed_X[i, foot_idx, 0].item()
                        foot_center_x = foot_x / 4
                        
                        # 鼻子x坐标
                        nose_x = smoothed_X[i, 0, 0].item()  # 鼻子关键点索引为0
                        
                        # 加权计算身体中心x坐标
                        body_center_x_frame = (0.4 * hip_center_x + 
                                            0.3 * shoulder_center_x + 
                                            0.2 * foot_center_x + 
                                            0.1 * nose_x).item()
                        
                        center_alignment = abs(wrists_center_x - body_center_x_frame).item()
                        
                        # 归一化两个分数
                        max_height = torch.max(wrists_y[search_start:search_end]).item()
                        min_height = torch.min(wrists_y[search_start:search_end]).item()
                        height_range = max_height - min_height
                        if height_range > 0:
                            norm_height_score = (height_score - min_height) / height_range
                        else:
                            norm_height_score = 1.0
                        
                        # 归一化中心对齐分数 (越小越好)
                        max_align_threshold = 0.15  # 设置一个合理的阈值
                        norm_center_score = 1.0 - min(center_alignment / max_align_threshold, 1.0)
                        
                        # 综合得分: 高度占70%，中心对齐占30%
                        total_score = 0.7 * norm_height_score + 0.3 * norm_center_score
                        
                        impact_candidates.append((i, total_score))
                    
                    # 按得分排序（得分越高越好）
                    impact_candidates.sort(key=lambda x: x[1], reverse=True)
                    
                    if impact_candidates:
                        # 选择得分最高的帧作为击球阶段
                        impact_idx = impact_candidates[0][0]
                        stage_indices["5"] = [impact_idx]
                        
                        # 添加相似帧
                        if top_k > 1 and len(impact_candidates) > 1:
                            additional_frames = [idx for idx, _ in impact_candidates[1:top_k]]
                            stage_indices["5"].extend(additional_frames)
                            stage_indices["5"].sort()
                    else:
                        # 如果没有候选帧，使用最低点
                        impact_idx = search_start + torch.argmax(wrists_y[search_start:search_end])
                        stage_indices["5"] = [impact_idx.item()]
        
        # 如果没找到击球阶段，在中间部分寻找手腕最低点并考虑身体中心对齐
        if not stage_indices["5"]:  # 击球阶段改为阶段5
            search_start = int(frame_count * 0.3)
            search_end = int(frame_count * 0.7)
            
            if search_end > search_start:
                # 计算每个候选帧的得分，结合手腕高度和与身体中心的水平对齐程度
                impact_candidates = []
                for i in range(search_start, search_end):
                    # 手腕高度得分 (Y值越大越好)
                    height_score = wrists_y[i].item()
                    
                    # 双手平均x坐标与身体中心的距离 (越小越好)
                    wrists_center_x = (smoothed_X[i, LEFT_WRIST, 0] + smoothed_X[i, RIGHT_WRIST, 0]) / 2
                    
                    # 计算加权身体中心x坐标
                    hip_center_x = (smoothed_X[i, LEFT_HIP, 0] + smoothed_X[i, RIGHT_HIP, 0]) / 2
                    shoulder_center_x = (smoothed_X[i, LEFT_SHOULDER, 0] + smoothed_X[i, RIGHT_SHOULDER, 0]) / 2
                    
                    # 计算脚部中心
                    foot_x = 0
                    for foot_idx in [27, 28, 29, 30]:  # 脚踝和脚部关键点
                        foot_x += smoothed_X[i, foot_idx, 0].item()
                    foot_center_x = foot_x / 4
                    
                    # 鼻子x坐标
                    nose_x = smoothed_X[i, 0, 0].item()  # 鼻子关键点索引为0
                    
                    # 加权计算身体中心x坐标
                    body_center_x_frame = (0.4 * hip_center_x + 
                                        0.3 * shoulder_center_x + 
                                        0.2 * foot_center_x + 
                                        0.1 * nose_x).item()
                    
                    center_alignment = abs(wrists_center_x - body_center_x_frame).item()
                    
                    # 归一化两个分数
                    max_height = torch.max(wrists_y[search_start:search_end]).item()
                    min_height = torch.min(wrists_y[search_start:search_end]).item()
                    height_range = max_height - min_height
                    if height_range > 0:
                        norm_height_score = (height_score - min_height) / height_range
                    else:
                        norm_height_score = 1.0
                    
                    # 归一化中心对齐分数 (越小越好)
                    max_align_threshold = 0.15  # 设置一个合理的阈值
                    norm_center_score = 1.0 - min(center_alignment / max_align_threshold, 1.0)
                    
                    # 综合得分: 高度占70%，中心对齐占30%
                    total_score = 0.7 * norm_height_score + 0.3 * norm_center_score
                    
                    impact_candidates.append((i, total_score))
                
                # 按得分排序（得分越高越好）
                impact_candidates.sort(key=lambda x: x[1], reverse=True)
                
                if impact_candidates:
                    # 选择得分最高的帧作为击球阶段
                    impact_idx = impact_candidates[0][0]
                    stage_indices["5"] = [impact_idx]
                else:
                    # 如果仍无法确定，使用最低点
                    impact_idx = search_start + torch.argmax(wrists_y[search_start:search_end])
                    stage_indices["5"] = [impact_idx.item()]
        
        # 步骤8: 识别顶点阶段 (阶段3) - 上杆和下杆之间手部坐标最高点
        if stage_indices["2"] and stage_indices["4"]:  # 在上杆和下杆之间查找顶点
            backswing_idx = max(stage_indices["2"])
            downswing_idx = min(stage_indices["4"])
            
            if downswing_idx > backswing_idx:
                search_start = backswing_idx + 1
                search_end = downswing_idx
                
                if search_end > search_start:
                    # 找到手腕高度最高点 (Y值最小)
                    top_idx = search_start + torch.argmin(wrists_y[search_start:search_end])
                    stage_indices["3"] = [top_idx.item()]  # 顶点阶段为阶段3
                    
                    # 添加相似帧
                    if top_k > 1:
                        window = min(5, (search_end - search_start) // 2)
                        start = max(top_idx.item() - window, search_start)
                        end = min(top_idx.item() + window, search_end)
                        
                        similar_frames = []
                        for i in range(start, end):
                            if i == top_idx.item():
                                continue
                            # 计算与顶点帧的相似度
                            similarity = abs(wrists_y[i] - wrists_y[top_idx])
                            similar_frames.append((i, similarity.item()))
                        
                        # 选择相似度最高的帧
                        similar_frames.sort(key=lambda x: x[1])
                        additional_frames = [idx for idx, _ in similar_frames[:top_k-1]]
                        stage_indices["3"].extend(additional_frames)
                        stage_indices["3"].sort()
        elif not stage_indices["3"] and stage_indices["4"]:  # 如果没有上杆阶段，但有下杆阶段
            # 如果没找到上杆但找到下杆，在下杆前寻找顶点
            downswing_idx = min(stage_indices["4"])
            search_start = 0
            search_end = downswing_idx
            
            if search_end > search_start:
                # 找到手腕高度最高点 (Y值最小)
                top_idx = search_start + torch.argmin(wrists_y[search_start:search_end])
                stage_indices["3"] = [top_idx.item()]  # 顶点阶段
        
        # 步骤9: 识别准备摆放阶段 (阶段0) - 视频最开始的部分，手部坐标最低点
        if stage_indices["2"] and stage_indices["4"]:
            # 已经找到上杆和下杆阶段，在上杆之前找准备摆放
            backswing_idx = min(stage_indices["2"])
            search_start = 0
            search_end = backswing_idx
            
            if search_end > search_start:
                # 找到手腕高度最低点 (Y值最大)
                setup_idx = search_start + torch.argmax(wrists_y[search_start:search_end])
                stage_indices["0"] = [setup_idx.item()]
                
                # 添加相似帧
                if top_k > 1:
                    window = min(5, (search_end - search_start) // 2)
                    start = max(setup_idx.item() - window, search_start)
                    end = min(setup_idx.item() + window, search_end)
                    
                    similar_frames = []
                    for i in range(start, end):
                        if i == setup_idx.item():
                            continue
                        # 计算与准备摆放帧的相似度
                        similarity = abs(wrists_y[i] - wrists_y[setup_idx])
                        similar_frames.append((i, similarity.item()))
                    
                    # 选择相似度最高的帧
                    similar_frames.sort(key=lambda x: x[1])
                    additional_frames = [idx for idx, _ in similar_frames[:top_k-1]]
                    stage_indices["0"].extend(additional_frames)
                    stage_indices["0"].sort()
        else:
            # 如果找不到上杆阶段，使用视频最前面的帧
            stage_indices["0"] = [0]

        # 步骤10: 识别引杆阶段 (阶段1) - 在准备摆放和上杆之间
        if stage_indices["0"] and stage_indices["2"]:
            setup_idx = max(stage_indices["0"])
            backswing_idx = min(stage_indices["2"])
            
            if backswing_idx > setup_idx:
                search_start = setup_idx + 1
                search_end = backswing_idx
                
                if search_end > search_start:
                    # 计算准备摆放和上杆阶段的手腕高度平均值
                    avg_height = (wrists_y[setup_idx] + wrists_y[backswing_idx]) / 2
                    
                    # 找到最接近平均高度的帧
                    height_diff = torch.abs(wrists_y[search_start:search_end] - avg_height)
                    address_idx = search_start + torch.argmin(height_diff)
                    stage_indices["1"] = [address_idx.item()]
                    
                    # 添加相似帧
                    if top_k > 1:
                        window = min(5, (search_end - search_start) // 2)
                        start = max(address_idx.item() - window, search_start)
                        end = min(address_idx.item() + window, search_end)
                        
                        similar_frames = []
                        for i in range(start, end):
                            if i == address_idx.item():
                                continue
                            # 计算与引杆帧的相似度
                            similarity = abs(wrists_y[i] - wrists_y[address_idx])
                            similar_frames.append((i, similarity.item()))
                        
                        # 选择相似度最高的帧
                        similar_frames.sort(key=lambda x: x[1])
                        additional_frames = [idx for idx, _ in similar_frames[:top_k-1]]
                        stage_indices["1"].extend(additional_frames)
                        stage_indices["1"].sort()
        elif not stage_indices["1"]:
            # 如果找不到引杆阶段，但有准备摆放阶段
            if stage_indices["0"]:
                setup_idx = max(stage_indices["0"])
                address_idx = min(setup_idx + 10, frame_count - 1)
                stage_indices["1"] = [address_idx]
            else:
                # 如果连准备摆放也没找到，使用视频10%处作为引杆阶段
                stage_indices["1"] = [int(frame_count * 0.1)]
        
        # 步骤11: 识别起杆阶段 (阶段2) - 在引杆和上杆之间，球杆接近水平的帧
        # 需要重新调整阶段编号：原来的阶段2变成阶段3，以此类推
        original_stage_2 = stage_indices["2"].copy() if stage_indices["2"] else []
        original_stage_3 = stage_indices["3"].copy() if stage_indices["3"] else []
        original_stage_4 = stage_indices["4"].copy() if stage_indices["4"] else []
        original_stage_5 = stage_indices["5"].copy() if stage_indices["5"] else []
        original_stage_6 = stage_indices["6"].copy() if stage_indices["6"] else []
        original_stage_7 = stage_indices["7"].copy() if stage_indices["7"] else []
        
        # 重新分配阶段编号
        stage_indices["3"] = original_stage_2  # 原上杆变成阶段3
        stage_indices["4"] = original_stage_3  # 原顶点变成阶段4  
        stage_indices["5"] = original_stage_4  # 原下杆变成阶段5
        stage_indices["6"] = original_stage_5  # 原击球变成阶段6
        stage_indices["7"] = original_stage_6  # 原随杆变成阶段7
        stage_indices["8"] = original_stage_7  # 原收杆变成阶段8
        
        # 检测起杆阶段（阶段2）- 在引杆和上杆之间寻找球杆水平的帧
        stage_indices["2"] = []
        if stage_indices["1"] and stage_indices["3"] and X.shape[1] >= 35:  # 确保有球杆数据
            takeaway_frames = stage_indices["1"]  # 引杆阶段帧
            backswing_frames = stage_indices["3"]  # 上杆阶段帧（重新分配后）
            
            # 确定搜索范围：从引杆结束到上杆开始
            search_start = max(takeaway_frames) if takeaway_frames else 0
            search_end = min(backswing_frames) if backswing_frames else frame_count
            
            # 在搜索范围内寻找球杆水平的帧
            horizontal_frames = []
            
            for frame_idx in range(search_start, min(search_end + 10, frame_count)):
                if frame_idx >= frame_count:
                    break
                    
                frame_keypoints = X[frame_idx]
                
                # 检查是否有球杆数据
                if X.shape[1] < 35:
                    continue
                    
                # 获取球杆关键点（假设球杆关键点在索引33和34）
                club_grip = frame_keypoints[33][:2]  # 球杆握把端
                club_head = frame_keypoints[34][:2]  # 球杆头端
                
                # 检查关键点是否有效
                if (club_grip[0] <= 0 or club_grip[1] <= 0 or 
                    club_head[0] <= 0 or club_head[1] <= 0):
                    continue
                
                # 计算球杆与水平线的夹角
                dx = club_head[0] - club_grip[0]
                dy = club_head[1] - club_grip[1]
                
                if abs(dx) < 1e-6:  # 避免除零错误
                    continue
                    
                # 计算角度（弧度转角度）
                angle = abs(math.degrees(math.atan2(abs(dy), abs(dx))))
                
                # 如果球杆接近水平（±15度），记录该帧
                if angle <= 15.0:
                    horizontal_frames.append(frame_idx)
            
            # 如果找到了水平帧，设置为起杆阶段
            if horizontal_frames:
                stage_indices["2"] = horizontal_frames[:top_k]  # 限制帧数
                print(f"[INFO] 检测到起杆阶段（阶段2）帧数: {len(stage_indices['2'])}帧")
            else:
                # 如果没找到明显的水平帧，在引杆和上杆之间插值
                if search_end > search_start:
                    mid_frame = (search_start + search_end) // 2
                    stage_indices["2"] = [mid_frame]
                    print(f"[INFO] 未检测到明显的起杆阶段，使用插值帧: {mid_frame}")
        
        # 确保所有阶段都有至少一个索引（现在是9个阶段：0-8）
        for stage in ["0", "1", "2", "3", "4", "5", "6", "7", "8"]:
            if not stage_indices[stage]:
                # 对缺失的阶段进行插值
                self._interpolate_missing_stage(stage_indices, stage)
        
        return stage_indices, rotation_type
    
    def _interpolate_missing_stage(self, stage_indices: Dict[str, List[int]], missing_stage: str):
        """对缺失的阶段进行插值"""
        stage_idx = int(missing_stage)
        
        # 找到最近的前后阶段
        prev_stage = None
        for i in range(stage_idx - 1, -1, -1):
            if stage_indices[str(i)]:
                prev_stage = str(i)
                break
                
        next_stage = None
        for i in range(stage_idx + 1, 9):  # 修改为9个阶段（0-8）
            if stage_indices[str(i)]:
                next_stage = str(i)
                break
        
        if prev_stage is not None and next_stage is not None:
            # 在前后阶段之间进行插值
            prev_frame = stage_indices[prev_stage][-1]
            next_frame = stage_indices[next_stage][0]
            
            if next_frame > prev_frame:
                # 按比例插值
                ratio = (stage_idx - int(prev_stage)) / (int(next_stage) - int(prev_stage))
                interp_frame = int(prev_frame + ratio * (next_frame - prev_frame))
                stage_indices[missing_stage] = [interp_frame]
            else:
                # 异常情况，使用前一阶段帧+1
                stage_indices[missing_stage] = [prev_frame + 1]
        elif prev_stage is not None:
            # 只有前一阶段存在
            stage_indices[missing_stage] = [stage_indices[prev_stage][-1] + 1]
        elif next_stage is not None:
            # 只有后一阶段存在
            stage_indices[missing_stage] = [stage_indices[next_stage][0] - 1]
        else:
            # 前后阶段都不存在，使用默认值
            stage_indices[missing_stage] = [0]

    def compute_angle(self, p1, p2, p3):
        """
        计算三个点形成的角度
        
        Args:
            p1: 第一个点坐标 [x, y]
            p2: 中间点坐标 [x, y]（角的顶点）
            p3: 第三个点坐标 [x, y]
            
        Returns:
            角度值（度）
        """
        import numpy as np
        
        # 计算向量
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 计算夹角（弧度）
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在[-1, 1]范围内
        angle = np.arccos(cos_angle)
        
        # 转为角度
        angle_deg = np.degrees(angle)
        return angle_deg

def temporal_smoothing(X, window_size=5):
    """
    对关键点序列进行时序平滑处理
    
    Args:
        X: 关键点数据，形状为 (N, K, 2) 或 (N, K*2)，其中N是帧数，K是关键点数
        window_size: 平滑窗口大小
        
    Returns:
        平滑后的关键点数据，与输入形状相同
    """
    import torch
    
    # 确保X的形状正确
    orig_shape = X.shape
    if X.dim() == 2:
        # 如果是扁平化的数据 (N, K*2)，重塑为 (N, K, 2)
        frame_count = X.shape[0]
        X = X.view(frame_count, -1, 2)
    
    N, K, D = X.shape  # 帧数、关键点数、维度数(通常为2)
    
    # 创建滑动平均核
    kernel_size = min(window_size, N)
    if kernel_size % 2 == 0:  # 确保窗口大小为奇数
        kernel_size -= 1
    if kernel_size < 3:
        return X  # 如果窗口太小，不进行平滑
        
    # 使用更温和的平滑处理
    pad_size = kernel_size // 2
    smoothed_X = X.clone()
    device = X.device  # 获取输入张量的设备
    
    # 创建一个加权窗口，中间权重更高
    window_weights = torch.tensor([1 + abs(i - pad_size) for i in range(kernel_size)], 
                                 dtype=X.dtype, device=device)
    window_weights = window_weights.max() - window_weights + 1  # 反转权重使中间最高
    window_weights = window_weights / window_weights.sum()  # 归一化
    window_weights = window_weights.view(1, 1, -1)
    
    # 对每个关键点进行加权平滑
    for k in range(K):
        for d in range(D):
            # 获取当前关键点的特定维度序列
            signal = X[:, k, d]
            
            # 使用反射填充进行边界处理
            padded_signal = torch.nn.functional.pad(signal.unsqueeze(0).unsqueeze(0), 
                                                   (pad_size, pad_size), 
                                                   mode='reflect')
            
            # 应用加权平滑 - 确保window和padded_signal在同一设备上
            smoothed_signal = torch.nn.functional.conv1d(padded_signal, window_weights).squeeze()
            
            # 避免过度平滑，混合原始信号和平滑后的信号
            alpha = 0.7  # 平滑信号的权重
            smoothed_X[:, k, d] = alpha * smoothed_signal + (1 - alpha) * signal
    
    # 恢复原始形状
    if len(orig_shape) == 2:
        smoothed_X = smoothed_X.view(orig_shape)
        
    return smoothed_X

def savgol_filter1d(x, window_length=7, polyorder=2):
    """
    实现一维Savitzky-Golay滤波器
    
    Args:
        x: 输入信号
        window_length: 窗口长度（必须是奇数）
        polyorder: 多项式阶数
    
    Returns:
        滤波后的信号
    """
    import torch
    import numpy as np

    # 确保窗口长度为奇数
    if window_length % 2 == 0:
        window_length += 1
    
    # 窗口长度不能大于信号长度
    window_length = min(window_length, len(x) - (1 - len(x) % 2))
    
    # 窗口长度必须大于polyorder
    polyorder = min(polyorder, window_length - 1)
    
    # 将PyTorch张量转换为NumPy数组
    is_tensor = isinstance(x, torch.Tensor)
    if is_tensor:
        device = x.device
        x_np = x.cpu().numpy()
    else:
        x_np = x
    
    # 计算Savitzky-Golay系数
    half_window = (window_length - 1) // 2
    b = np.mat([[k**i for i in range(polyorder + 1)] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[0]
    
    # 应用滤波器
    firstvals = x_np[0] - np.abs(x_np[1:half_window+1][::-1] - x_np[0])
    lastvals = x_np[-1] + np.abs(x_np[-half_window-1:-1][::-1] - x_np[-1])
    x_extended = np.concatenate((firstvals, x_np, lastvals))
    
    y = np.convolve(m[::-1], x_extended, mode='valid')
    
    # 为了避免过度平滑，将滤波结果与原始数据混合
    y = 0.7 * y + 0.3 * x_np
    
    # 将NumPy数组转换回PyTorch张量
    if is_tensor:
        y = torch.tensor(y, device=device, dtype=x.dtype)
    
    return y
