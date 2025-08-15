# -*- coding: utf-8 -*-
"""
高尔夫挥杆检查模块
用于根据关键点数据检查高尔夫挥杆动作是否符合特定条件

主要功能：
1. 读取keypoints.pt和stage_indices.json文件
2. 解析检查条件文本，创建挥杆条件检查函数
3. 对每个挥杆阶段应用相应的检查条件
4. 生成报告，标出不符合条件的项目及其对应帧号
"""

import os
import sys
import json
import torch
import numpy as np
import math
from datetime import datetime

# 导入项目相关模块
from config import BODY_POINT_NAMES, STAGE_MAP, CHECK_PARAMS, CONDITION_STAGE_RULES

# 常量定义
DEFAULT_OUTPUT_DIR = "resultData"  # 默认输出目录

# 关键点索引常量，基于MediaPipe的索引
NOSE = 0
RIGHT_EYE = 2  # 添加右眼索引
LEFT_EYE = 5   # 可选：也可以添加左眼索引
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
CLUB_HAND_INDEX = 33  # 来自输入关键点文件的附加点(握把/手)
CLUB_HEAD_INDEX = 34  # 来自输入关键点文件的附加点(杆头)

# 挥杆阶段映射
STAGE_NAMES = {
    "0": "准备姿势",
    "1": "引杆",
    "2": "起杆",      # 新增
    "3": "上杆",      # 原来的"2"
    "4": "上杆顶点",   # 原来的"3"
    "5": "下杆",      # 原来的"4"
    "6": "击球",      # 原来的"5"
    "7": "送杆",      # 原来的"6"
    "8": "收杆"       # 原来的"7"
}

# 硬编码的检查条件字典
HARDCODED_CONDITIONS = {
        "front": [
            "躯干偏移检测",
            "髋部旋转检测",
            "杆身范围检测",

        ],
        "side": [
            "头部K线检测",
            "臀线检测",
            "膝盖弯曲检测",
            "双脚连线水平检测",
            "挥杆轨迹检测"
        ]
    }
    


class SwingChecker:
    """高尔夫挥杆检查类"""
    
    def __init__(self, use_hardcoded=True, keypoints=None):
        """
        初始化挥杆检查器
        
        Args:
            use_hardcoded: 是否使用硬编码的检查条件
            keypoints: 关键点数据，用于预先计算参考尺度
        """
        self.conditions = HARDCODED_CONDITIONS
        self.check_functions = self._register_check_functions()
        
        # 预先计算参考尺度
        self.reference_scale = None
        if keypoints is not None:
            self.calculate_reference_scale(keypoints)
    
    def _register_check_functions(self):
        """
        注册各种检查函数
        
        Returns:
            dict: 检查函数映射
        """
        # 以视角(front/side)和中文检查项名称建立到检测函数的映射
        # 这样可以直接根据 HARDCODED_CONDITIONS 的名称取到对应函数
        check_functions = {
            "front": {
                "躯干偏移检测": self.check_torso_sway_front,
                "髋部旋转检测": self.check_hip_rotation_front,
                "杆身范围检测": self.check_shaft_range_front,
            },
            "side": {
                "头部K线检测": self.check_head_k_line,
                "臀线检测": self.check_hip_line,
                "膝盖弯曲检测": self.check_knee_bend,
                "双脚连线水平检测": self.check_feet_line,
                "挥杆轨迹检测": self.check_swing_path,
            },
        }
        return check_functions
    
    def check_swing(self, keypoints, stage_indices, generate_visualizations=False, vis_dir=None, video_id=None, target_views=None):
        """
        检查挥杆动作
        
        Args:
            keypoints: 关键点数据，形状为(帧数, 关键点数, 坐标维度)的张量
            stage_indices: 各阶段对应的帧索引字典（已包含9个阶段0-8）
            generate_visualizations: 是否生成可视化图像
            vis_dir: 可视化图像保存目录
            
        Returns:
            dict: 检查结果报告
        """
        # 1. 先执行检查逻辑，收集检查结果
        selected_views = set(target_views) if target_views else {"front"}
        results = self._perform_swing_checks(keypoints, stage_indices, selected_views)
        
        # 2. 如果需要生成可视化，使用检查结果生成可视化
        if generate_visualizations and vis_dir:
            pass
            self._generate_visualizations(keypoints, results, stage_indices, vis_dir, video_id)
        
        return results
    def _generate_visualizations(self,keypoints, results, stage_indices, vis_dir, video_id):
        pass
    def _perform_swing_checks(self, keypoints, stage_indices, selected_views=None):
        """
        执行挥杆检查，收集检查结果
        
        Args:
            keypoints: 关键点数据
            stage_indices: 各阶段对应的帧索引字典
            
        Returns:
            dict: 检查结果
        """
        # 结果结构: { stage_key: { name: 阶段名, results: [ {condition, view, check_function, passed, failed_frames, frame_results} ] } }
        results = {}

        if not isinstance(stage_indices, dict):
            return results

        # 确保以字符串阶段键遍历，并按阶段顺序排序
        stage_keys = sorted(stage_indices.keys(), key=lambda x: int(x) if str(x).isdigit() else 999)

        for stage_key in stage_keys:
            frames = stage_indices.get(stage_key, []) or []
            stage_result_items = []

            for view, condition_list in self.conditions.items():
                if selected_views and view not in selected_views:
                    continue
                for condition_name in condition_list:
                    func = self.check_functions.get(view, {}).get(condition_name)

                    # 生效阶段过滤
                    allowed_stages = self._get_allowed_stages_for_condition(view, condition_name, stage_indices)
                    if str(stage_key) not in allowed_stages:
                        # 本阶段不适用该条件，记录占位并跳过计算
                        stage_result_items.append({
                            "condition": condition_name,
                            "view": view,
                            "check_function": func.__name__ if callable(func) else None,
                            "passed": True,
                            "failed_frames": [],
                            "frame_results": {},
                            "exceed_by": {},
                        })
                    continue
                    
                    # 准备帧级结果
                    frame_results = {}
                    failed_frames = []
                    exceed_by = {}

                    if callable(func):
                        for fi in frames:
                            # 安全边界判断
                            if isinstance(fi, int) and 0 <= fi < len(keypoints):
                                try:
                                    ret = func(keypoints[fi], stage_indices=stage_indices, all_keypoints=keypoints)
                                    if isinstance(ret, dict):
                                        res = bool(ret.get("ok", False))
                                        exc = ret.get("exceed", None)
                                        if exc is not None and not res:
                                            exceed_by[fi] = exc
                                    else:
                                        res = bool(ret)
                                except Exception:
                                    res = False
                                frame_results[fi] = res
                                if not res:
                                    failed_frames.append(fi)
                    else:
                        # 未实现的检查函数，标记为空并默认通过
                        func = None

                    passed = True
                    if frame_results:
                        passed = len(failed_frames) == 0

                    stage_result_items.append({
                        "condition": condition_name,
                        "view": view,
                        "check_function": func.__name__ if func else None,
                        "passed": passed,
                        "failed_frames": failed_frames,
                        "frame_results": frame_results,
                        "exceed_by": exceed_by,
                    })

            results[str(stage_key)] = {
                "name": STAGE_NAMES.get(str(stage_key), f"阶段{stage_key}"),
                "results": stage_result_items,
            }

        return results

    def _get_allowed_stages_for_condition(self, view, condition_name, stage_indices):
        """
        根据配置解析检测项的生效阶段表达式，返回允许的阶段键集合（字符串）。
        支持："pX", "pX-pY", 以及逗号分隔的非连续表达式。
        pX 映射为阶段 X-1。
        未配置时默认在所有阶段生效。
        """
        def parse_token(token):
            token = token.strip()
            if not token:
                return []
            if token[0] in ('p', 'P'):
                token = token[1:]
            if '-' in token:
                a, b = token.split('-', 1)
                try:
                    start = int(a) - 1
                    end = int(b) - 1
                except ValueError:
                    return []
                if start > end:
                    start, end = end, start
                return list(range(max(0, start), end + 1))
            else:
                try:
                    v = int(token) - 1
                    return [v] if v >= 0 else []
                except ValueError:
                    return []

        rules = CONDITION_STAGE_RULES.get(view, {}).get(condition_name)
        if not rules:
            return {str(k) for k in stage_indices.keys()}

        allowed = set()
        # 规则可以是字符串或字符串列表；每个字符串内部还可以用逗号分隔
        rule_list = rules if isinstance(rules, list) else [rules]
        for rule in rule_list:
            for token in str(rule).split(','):
                for idx in parse_token(token):
                    allowed.add(str(idx))

        # 仅保留在现有 stage_indices 中存在的阶段
        existing = {str(k) for k in stage_indices.keys()}
        return allowed & existing if allowed else existing


    
    def generate_report(self, results, output_file):
        """
        生成检查报告（仅显示不通过的项目，并按检测项目归类）
        
        Args:
            results: 检查结果
            output_file: 输出文件路径
            
        Returns:
            bool: 是否成功生成报告
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("高尔夫挥杆检查报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 汇总所有不通过项目，按(视角, 检测项目)归类
                def fmt_fail_item(fi, exc):
                    if isinstance(exc, dict):
                        if exc.get("unit") == "deg":
                            return f"{fi}(+{exc.get('value', 0):.1f}°)"
                        if exc.get("unit") == "sw":
                            return f"{fi}(+{exc.get('value', 0):.3f}sw)"
                    elif isinstance(exc, (int, float)):
                        return f"{fi}(+{float(exc):.3f})"
                    return str(fi)

                grouped = {}
                for stage_key, stage_data in results.items():
                    for r in stage_data["results"]:
                        if not r.get("passed") and r.get("failed_frames"):
                            key = (r.get("view"), r.get("condition"))
                            grouped.setdefault(key, {})[stage_key] = {
                                "stage_name": stage_data.get("name", f"阶段{stage_key}"),
                                "failed": [(fi, r.get("exceed_by", {}).get(fi)) for fi in r.get("failed_frames", [])]
                            }

                # 仅输出不通过的项目
                for (view_key, condition) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
                    view_cn = "正面" if view_key == "front" else "侧面"
                    f.write(f"[{view_cn}] {condition}\n")
                    f.write("-" * 50 + "\n")
                    stages = grouped[(view_key, condition)]
                    for sk in sorted(stages.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                        info = stages[sk]
                        fail_list = ", ".join([fmt_fail_item(fi, exc) for (fi, exc) in info["failed"]])
                        f.write(f"阶段{sk} {info['stage_name']}: 不通过的帧: [{fail_list}]\n")
                    f.write("\n")
                
                return True
        
        except Exception as e:
            print(f"生成报告时出错: {str(e)}")
            return False
    
    def generate_csv_report(self, results, output_file):
        """
        生成CSV格式的检查报告（仅输出不通过项，按检测项目归类排序）
        
        Args:
            results: 检查结果
            output_file: 输出文件路径
            
        Returns:
            bool: 是否成功生成报告
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入CSV头部（保持字段名兼容；仅输出不通过项，并以视角+检测项目为首要排序）
                f.write("阶段编号,阶段名称,视角,检查项目,检查结果,不通过帧序列\n")
                
                # 归类：key=(view_key, condition)
                grouped = {}
                for stage_key, stage_data in results.items():
                    stage_name = stage_data.get('name', f"阶段{stage_key}")
                    for r in stage_data.get("results", []):
                        if r.get("passed"):
                            continue
                        failed = r.get("failed_frames") or []
                        if not failed:
                            continue
                        key = (r.get("view"), r.get("condition"))
                        grouped.setdefault(key, []).append((stage_key, stage_name, failed))

                # 输出：按(视角, 检测项目)排序，再按阶段编号排序
                for (view_key, condition) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
                    view_cn = "正面" if view_key == "front" else "侧面"
                    for (stage_key, stage_name, failed) in sorted(grouped[(view_key, condition)], key=lambda x: int(x[0]) if str(x[0]).isdigit() else 999):
                        failed_frames_str = ",".join(map(str, failed))
                        f.write(f"{stage_key},\"{stage_name}\",{view_cn},\"{condition}\",不通过,\"{failed_frames_str}\"\n")
                
                return True
        
        except Exception as e:
            print(f"生成CSV报告时出错: {str(e)}")
            return False
    
    # ===== 以下是各种检查函数 =====
        
    def _calculate_angle(self, p1, p2, p3):
        """
        计算三个点形成的角度
        
        Args:
            p1: 第一个点坐标
            p2: 中间点坐标
            p3: 第三个点坐标
            
        Returns:
            float: 角度（度）
        """
        # 计算向量
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 计算向量的点积
        dot_product = np.dot(v1, v2)
        
        # 计算向量的模
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # 计算夹角的余弦值
        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在[-1, 1]范围内
        
        # 转换为角度
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def calculate_reference_scale(self, keypoints):
        """
        从前30帧计算最大肩宽作为参考尺度
        
        Args:
            keypoints: 关键点数据，形状为(帧数, 关键点数, 坐标维度)的张量
        """
        max_scale = 0
        frames_to_check = min(30, len(keypoints))
        
        for i in range(frames_to_check):
            frame_keypoints = keypoints[i]
            left_shoulder = frame_keypoints[LEFT_SHOULDER]
            right_shoulder = frame_keypoints[RIGHT_SHOULDER]
            shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
            max_scale = max(max_scale, shoulder_width)
        
        if max_scale > 0:
            self.reference_scale = max_scale
            print(f"计算得到的参考尺度为: {self.reference_scale}")
        else:
            print("无法计算参考尺度，将使用每帧计算的方式")
            self.reference_scale = None

    def _get_body_scale(self, keypoints):
        """
        计算身体比例尺度，用于相对距离计算
        
        Args:
            keypoints: 关键点坐标
            
        Returns:
            float: 身体比例尺度（肩宽）
        """
        # 优先使用预计算的参考尺度
        if self.reference_scale is not None:
            return self.reference_scale
            
        # 否则计算当前帧的肩宽
        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        return shoulder_width


    # ====== 统一的占位/简易可视化工具 ======
    def _visualize_text_only(self, title, keypoints, result, frame=None, save_path=None):
        import cv2
        import numpy as np
        if frame is None:
            # 构造简单白底图
            h, w = 480, 640
            frame = np.ones((h, w, 3), dtype=np.uint8) * 255
        h, w = frame.shape[:2]
        color = (0, 150, 0) if result else (0, 0, 255)
        cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(frame, f"Result: {'Pass' if result else 'Fail'}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if save_path:
            cv2.imwrite(save_path, frame)
        return frame

    def visualize_setup_position(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        # 准备姿势统一可视化（简化版）
        return self._visualize_text_only("Setup position overview", keypoints, result, frame, save_path)

    # ====== 检测函数（正面三项实现） ======
    # --- Front: 躯干偏移检测 ---
    def check_torso_sway_front(self, keypoints, stage_indices=None, all_keypoints=None):
        """
        基于阶段0第一帧的左右脚趾X建立两条外侧垂直基准线，允许一定容差。
        要求当前帧的躯干左右边界均在两线之间。
        返回: { ok, exceed }
        exceed 单位 sw(肩宽比例)
        """
        try:
            params = CHECK_PARAMS.get("front", {}).get("torso_sway", {})
            offset_ratio = float(params.get("line_outer_offset_ratio", 0.02))
            tol_ratio = float(params.get("torso_tolerance_ratio", 0.05))

            # 参考使用阶段0第一帧
            if stage_indices is None or all_keypoints is None or "0" not in stage_indices or not stage_indices["0"]:
                return {"ok": True}
            ref_idx = stage_indices["0"][0]
            ref_kp = all_keypoints[ref_idx]

            # 肩宽
            shoulder_width = self._get_body_scale(ref_kp)
            if shoulder_width <= 0:
                return {"ok": True}

            left_toe_x = ref_kp[LEFT_FOOT_INDEX][0]
            right_toe_x = ref_kp[RIGHT_FOOT_INDEX][0]
            # 基准两线：向外各扩 offset_ratio * S
            left_line = left_toe_x - offset_ratio * shoulder_width
            right_line = right_toe_x + offset_ratio * shoulder_width

            # 当前帧 torso 左/右边界
            lsx = keypoints[LEFT_SHOULDER][0]
            rsx = keypoints[RIGHT_SHOULDER][0]
            lhx = keypoints[LEFT_HIP][0]
            rhx = keypoints[RIGHT_HIP][0]
            left_edge = min(lsx, lhx)
            right_edge = max(rsx, rhx)

            # 容差范围扩张
            tol = tol_ratio * shoulder_width
            ok_left = left_edge >= (left_line - tol)
            ok_right = right_edge <= (right_line + tol)
            ok = bool(ok_left and ok_right)

            exceed_val = 0.0
            if not ok:
                # 计算最大越界量（归一化为sw）
                left_exceed = max(0.0, (left_line - tol) - left_edge)
                right_exceed = max(0.0, right_edge - (right_line + tol))
                exceed_val = max(left_exceed, right_exceed) / shoulder_width if shoulder_width > 0 else 0.0

            return {"ok": ok, "exceed": {"unit": "sw", "value": float(exceed_val)}}
        except Exception:
            return {"ok": False}

        
    # --- Front: 髋部旋转检测 ---
    def check_hip_rotation_front(self, keypoints, stage_indices=None, all_keypoints=None):
        """
        髋线与水平线角度的绝对值需在[min_deg, max_deg]区间内。
        exceed 单位 deg
        """
        try:
            params = CHECK_PARAMS.get("front", {}).get("hip_rotation", {})
            min_deg = float(params.get("min_deg", 0.0))
            max_deg = float(params.get("max_deg", 25.0))

            left_hip_pt = keypoints[LEFT_HIP][:2]
            right_hip_pt = keypoints[RIGHT_HIP][:2]
            dx = right_hip_pt[0] - left_hip_pt[0]
            dy = right_hip_pt[1] - left_hip_pt[1]
            if dx == 0 and dy == 0:
                return {"ok": False}
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            ok = (min_deg <= angle <= max_deg)
            exceed = 0.0
            if not ok:
                if angle < min_deg:
                    exceed = float(min_deg - angle)
                elif angle > max_deg:
                    exceed = float(angle - max_deg)
            return {"ok": ok, "exceed": {"unit": "deg", "value": float(exceed)}}
        except Exception:
            return {"ok": False}

    # --- Front: 杆身范围检测 ---
    def check_shaft_range_front(self, keypoints, stage_indices=None, all_keypoints=None):
        """
        使用 right_shoulder -> club_hand 与 club_head -> club_hand 的夹角作为杆身角。
        角度需在[min_deg, max_deg]内；exceed 单位 deg。
        """
        try:
            params = CHECK_PARAMS.get("front", {}).get("shaft_range", {})
            min_deg = float(params.get("min_deg", 150.0))
            max_deg = float(params.get("max_deg", 180.0))

            if keypoints.shape[0] <= max(CLUB_HAND_INDEX, CLUB_HEAD_INDEX):
                return {"ok": False, "exceed": {"unit": "deg", "value": float("nan")}}

            right_shoulder_pt = keypoints[RIGHT_SHOULDER][:2]
            club_hand_pt = keypoints[CLUB_HAND_INDEX][:2]
            club_head_pt = keypoints[CLUB_HEAD_INDEX][:2]

            v1 = club_hand_pt - right_shoulder_pt
            v2 = club_hand_pt - club_head_pt
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                return {"ok": False, "exceed": {"unit": "deg", "value": float("nan")}}
            cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cosang)))
            ok = (min_deg <= angle <= max_deg)
            exceed = 0.0
            if not ok:
                if angle < min_deg:
                    exceed = float(min_deg - angle)
                elif angle > max_deg:
                    exceed = float(angle - max_deg)
            return {"ok": ok, "exceed": {"unit": "deg", "value": float(exceed)}}
        except Exception:
            return {"ok": False}

    def check_head_k_line(self, keypoints, stage_indices=None, all_keypoints=None):
        """基础占位：头部K线检测。当前实现返回 True 保持流程可用。"""
        return True

    def check_hip_line(self, keypoints, stage_indices=None, all_keypoints=None):
        """基础占位：臀线检测。当前实现返回 True 保持流程可用。"""
        return True

    def check_knee_bend(self, keypoints, stage_indices=None, all_keypoints=None):
        """基础占位：膝盖弯曲检测。当前实现返回 True 保持流程可用。"""
        return True

    def check_feet_line(self, keypoints, stage_indices=None, all_keypoints=None):
        """基础占位：双脚连线水平检测。当前实现返回 True 保持流程可用。"""
        return True

    def check_swing_path(self, keypoints, stage_indices=None, all_keypoints=None):
        """基础占位：挥杆轨迹检测。当前实现返回 True 保持流程可用。"""
        return True

    # ====== 可视化占位（与检测函数名对应）======
    def visualize_hip_rotation(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        return self._visualize_text_only("Hip rotation", keypoints, result, frame, save_path)

    def visualize_club_range(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        return self._visualize_text_only("Club range", keypoints, result, frame, save_path)

    def visualize_head_k_line(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        return self._visualize_text_only("Head K-line", keypoints, result, frame, save_path)

    def visualize_hip_line(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        return self._visualize_text_only("Hip line", keypoints, result, frame, save_path)

    def visualize_knee_bend(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        return self._visualize_text_only("Knee bend", keypoints, result, frame, save_path)

    def visualize_feet_line(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        return self._visualize_text_only("Feet line horizontal", keypoints, result, frame, save_path)

    def visualize_swing_path(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        return self._visualize_text_only("Swing path", keypoints, result, frame, save_path)

def load_keypoints(keypoints_file):
    """
    加载关键点数据
    
    Args:
        keypoints_file: 关键点文件路径
        
    Returns:
        numpy.ndarray: 关键点数据
    """
    try:
        keypoints = torch.load(keypoints_file)
        if isinstance(keypoints, torch.Tensor):
            return keypoints.numpy()
        return np.array(keypoints)
    except Exception as e:
        print(f"加载关键点数据时出错: {str(e)}")
        return None

def load_stage_indices(stage_indices_file):
    """
    加载阶段索引数据
    
    Args:
        stage_indices_file: 阶段索引文件路径
        
    Returns:
        dict: 阶段索引字典
    """
    try:
        with open(stage_indices_file, 'r') as f:
            stage_indices = json.load(f)
        return stage_indices
    except Exception as e:
        print(f"加载阶段索引数据时出错: {str(e)}")
        return None

def check_swing(video_name, output_dir=None, generate_visualizations=False, csv_format=False):
    """
    检查挥杆动作是否符合条件
    
    Args:
        video_name: 视频文件名（不含扩展名）
        output_dir: 输出目录，默认为与视频同名的目录
        generate_visualizations: 是否生成可视化辅助线对比图像
        csv_format: 是否生成CSV格式的报告，默认为否
        
    Returns:
        tuple: (成功标志, 报告文件路径)
    """
    try:
        # 确定文件路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = base_dir  # 假设脚本在项目根目录
        
        # 简化视频名称，只保留基本部分
        video_id = video_name
        
        # 首先尝试直接在resultData目录下查找视频目录
        video_dir = os.path.join(project_root, "resultData", video_name)
        if not os.path.exists(video_dir):
            # 尝试使用原始视频目录名
            video_dir = os.path.join(project_root, "resultData", f"{video_name}_raw")
            if not os.path.exists(video_dir):
                # 尝试查找以视频名开头的目录
                resultData_dir = os.path.join(project_root, "resultData")
                if os.path.exists(resultData_dir):
                    matching_dirs = [d for d in os.listdir(resultData_dir) 
                                    if os.path.isdir(os.path.join(resultData_dir, d)) and 
                                    (d.startswith(video_name) or video_name in d)]
                    if matching_dirs:
                        video_dir = os.path.join(resultData_dir, matching_dirs[0])
                        print(f"找到匹配目录: {video_dir}")
                    else:
                        print(f"找不到视频目录: {video_dir}")
                        print(f"resultData目录中的文件: {os.listdir(resultData_dir)}")
                        return False, None
                else:
                    print(f"找不到resultData目录: {resultData_dir}")
                    return False, None
        
        # 关键点和阶段索引文件
        keypoints_file = os.path.join(video_dir, "keypoints.pt")
        stage_indices_file = os.path.join(video_dir, "stage_indices.json")
        
        if not os.path.exists(keypoints_file):
            print(f"找不到关键点文件: {keypoints_file}")
            return False, None
        
        if not os.path.exists(stage_indices_file):
            print(f"找不到阶段索引文件: {stage_indices_file}")
            return False, None
        
        # 加载数据
        keypoints = load_keypoints(keypoints_file)
        stage_indices = load_stage_indices(stage_indices_file)
        
        if keypoints is None or stage_indices is None:
            print("加载数据失败")
            return False, None
        
        # 创建输出目录 - 使用简化的名称
        if output_dir is None:
            output_dir = os.path.join(project_root, DEFAULT_OUTPUT_DIR, video_id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化 vis_dir 变量
        vis_dir = None
        
        # 为可视化创建子目录
        if generate_visualizations:
            vis_dir = os.path.join(output_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
        
        # 创建检查器并预先计算参考尺度
        checker = SwingChecker(keypoints=keypoints)
        results = checker.check_swing(keypoints,    stage_indices, generate_visualizations, vis_dir, video_id)
        
        # 根据选项生成对应格式的报告
        if csv_format:
            report_file = os.path.join(vis_dir, f"{video_id}_report.csv")
            success = checker.generate_csv_report(results, report_file)
            report_type = "CSV格式报告"
        else:
            report_file = os.path.join(vis_dir, f"{video_id}_report.txt")
            success = checker.generate_report(results, report_file)
            report_type = "文本报告"
        
        if success:
            print(f"成功生成挥杆检查{report_type}: {report_file}")
            if generate_visualizations and vis_dir:
                print(f"可视化图像保存在: {vis_dir}")
            return True, report_file
        else:
            print(f"生成{report_type}失败")
            return False, None
    
    except Exception as e:
        print(f"执行挥杆检查时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def check_swing_front_and_side(video_name, output_dir=None, generate_visualizations=False, csv_format=False):
    """
    同一份报告内生成正面+侧面的检测结果。
    侧面关键点文件名固定为 keypoints_side.pt，路径与正面 keypoints.pt 同目录。
    共用 stage_indices.json。

    Returns:
        tuple: (成功标志, 报告文件路径)
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = base_dir

        video_id = video_name

        # 定位视频目录（沿用正面逻辑）
        video_dir = os.path.join(project_root, "resultData", video_name)
        if not os.path.exists(video_dir):
            video_dir = os.path.join(project_root, "resultData", f"{video_name}_raw")
            if not os.path.exists(video_dir):
                resultData_dir = os.path.join(project_root, "resultData")
                if os.path.exists(resultData_dir):
                    matching_dirs = [d for d in os.listdir(resultData_dir)
                                     if os.path.isdir(os.path.join(resultData_dir, d)) and
                                     (d.startswith(video_name) or video_name in d)]
                    if matching_dirs:
                        video_dir = os.path.join(resultData_dir, matching_dirs[0])
                        print(f"找到匹配目录: {video_dir}")
                    else:
                        print(f"找不到视频目录: {video_dir}")
                        print(f"resultData目录中的文件: {os.listdir(resultData_dir)}")
                        return False, None
                else:
                    print(f"找不到resultData目录: {resultData_dir}")
                    return False, None

        # 文件路径
        keypoints_front_file = os.path.join(video_dir, "keypoints.pt")
        keypoints_side_file = os.path.join(video_dir, "keypoints_side.pt")
        stage_indices_file = os.path.join(video_dir, "stage_indices.json")

        if not os.path.exists(stage_indices_file):
            print(f"找不到阶段索引文件: {stage_indices_file}")
            return False, None

        # 加载数据
        keypoints_front = load_keypoints(keypoints_front_file) if os.path.exists(keypoints_front_file) else None
        keypoints_side = load_keypoints(keypoints_side_file) if os.path.exists(keypoints_side_file) else None
        stage_indices = load_stage_indices(stage_indices_file)

        if stage_indices is None:
            print("加载数据失败: stage_indices 无效")
            return False, None
        if keypoints_front is None and keypoints_side is None:
            print("前后视角关键点均不可用，无法生成报告")
            return False, None

        # 输出目录与报告目录
        if output_dir is None:
            output_dir = os.path.join(project_root, DEFAULT_OUTPUT_DIR, video_id)
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)

        combined_results = {}

        # 正面
        if keypoints_front is not None:
            checker_front = SwingChecker(keypoints=keypoints_front)
            front_results = checker_front.check_swing(
                keypoints_front, stage_indices,
                generate_visualizations=generate_visualizations,
                vis_dir=vis_dir, video_id=video_id, target_views={"front"}
            )
            # 初始化 combined 结果
            for stage_key, data in front_results.items():
                combined_results[stage_key] = {
                    "name": data.get("name"),
                    "results": list(data.get("results", []))
                }

        # 侧面
        if keypoints_side is not None:
            checker_side = SwingChecker(keypoints=keypoints_side)
            side_results = checker_side.check_swing(
                keypoints_side, stage_indices,
                generate_visualizations=generate_visualizations,
                vis_dir=vis_dir, video_id=video_id, target_views={"side"}
            )
            for stage_key, data in side_results.items():
                if stage_key not in combined_results:
                    combined_results[stage_key] = {
                        "name": data.get("name"),
                        "results": []
                    }
                combined_results[stage_key]["results"].extend(list(data.get("results", [])))

        # 生成报告（单文件，包含两视角）
        # 任取一个checker用于调用输出函数
        checker_any = checker_front if keypoints_front is not None else checker_side
        if checker_any is None:
            print("未找到可用检查器")
            return False, None

        if csv_format:
            report_file = os.path.join(vis_dir, f"{video_id}_report.csv")
            success = checker_any.generate_csv_report(combined_results, report_file)
            report_type = "CSV格式报告"
        else:
            report_file = os.path.join(vis_dir, f"{video_id}_report.txt")
            success = checker_any.generate_report(combined_results, report_file)
            report_type = "文本报告"

        if success:
            print(f"成功生成挥杆检查{report_type}: {report_file}")
            return True, report_file
        else:
            print(f"生成{report_type}失败")
            return False, None

    except Exception as e:
        print(f"执行正侧面联合挥杆检查时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    # 命令行参数解析
    import argparse
    
    parser = argparse.ArgumentParser(description="高尔夫挥杆检查工具")
    parser.add_argument("--video", "-v", required=True, help="视频文件名（不含扩展名）")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--visualize", "-vis", action="store_true", help="是否生成可视化辅助线对比图像")
    parser.add_argument("--csv", "-c", action="store_true", help="是否生成CSV格式的报告（默认为文本格式）")
    
    args = parser.parse_args()
    
    success, report_file = check_swing(args.video, args.output, args.visualize, args.csv)
    
    if success:
        print(f"检查完成，报告保存在: {report_file}")
    else:
        print("检查失败")
        sys.exit(1)
