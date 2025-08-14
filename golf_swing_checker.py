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
from config import BODY_POINT_NAMES, STAGE_MAP

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
    "0": {  # 准备姿势
        "front": [
            "左肩略高于右肩",
            "左臂成一条直线且过大腿内侧",
            "双脚宽度与肩同宽",
            "两手虎口线平行指向右耳/右肩"
        ],
        "side": [
            "双臂自然垂直地面，左肩与地面连线垂直",
            "髋部与小腿垂直",
            "双脚连线与目标方向平行",
            "颈椎与尾骨连线保持直线，避免塌腰"
        ]
    },
    "1": {  # 引杆
        "front": [
            "杆头趾部横截面延长线保持一致"
        ],
        "side": [
            "杆身始终指向肚脐延长线",
            "杆身与脊柱垂直，杆面正对胸口"
        ]
    },
    "2": {  # 起杆 - 新增
        "front": [
            "球杆保持水平状态",
            "左臂保持伸直",
            "身体重心稳定"
        ],
        "side": [
            "球杆与地面平行",
            "上身角度保持稳定",
            "髋部转动开始启动"
        ]
    },
    "3": {  # 上杆 - 原来的"2"
        "front": [
            "头部不超过平行地面的K线",
            "躯干保持在双脚外侧垂直线内"
        ],
        "side": [
            "臀部垂直线不可偏离",
            "身体角度线（头部平行线）稳定",
            "脊柱角度与杆面底线延长线一致"
        ]
    },
    "4": {  # 上杆顶点 - 原来的"3"
        "front": [
            "杆身超过头部平行线"
        ],
        "side": [
            "前臂伸直，杆面延长线与手臂平行",
            "杆面保持方正"
        ]
    },
    "5": {  # 下杆 - 原来的"4"
        "front": [
            "头部保持在头部圆圈内",
            "左髋靠近左脚外侧垂直线不要超过"
        ],
        "side": [
            "挥杆轨迹保持在胯部与肩膀连线锥形空间内"
        ]
    },
    "6": {  # 击球 - 原来的"5"
        "front": [
            "保持臂三角角度，避免鸡翅膀"
        ],
        "side": [
            "保持臂三角角度，避免鸡翅膀"
        ]
    },
    "7": {  # 送杆 - 原来的"6"
        "front": [
            "左侧身体靠近左脚外侧垂直线",
            "手臂伸直且交叉，避免右曲球",
            "头部保持在左侧垂直线左侧"
        ],
        "side": [
            "头部低于平行地面的线",
            "臀部贴紧后侧线，避免髋部前移"
        ]
    },
    "8": {  # 收杆 - 原来的"7"
        "front": [
            "左侧身体与左脚外侧垂直线重合"
        ],
        "side": [
            "左大臂平行于地面",
            "身体角度保持稳定"
        ]
    }
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
    
    def _load_conditions(self, conditions_file):
        """
        加载检查条件
        
        Args:
            conditions_file: 检查条件文件路径
            
        Returns:
            dict: 按阶段组织的检查条件字典
        """
        try:
            with open(conditions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析条件文件
            conditions = {}
            current_stage = None
            
            for line in lines[1:]:  # 跳过标题行
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                # 检查是否是新的阶段行
                for stage_key, stage_name in STAGE_NAMES.items():
                    if stage_name in line and not any(p in line for p in ["检查", "要点"]):
                        current_stage = stage_key
                        if current_stage not in conditions:
                            conditions[current_stage] = {"front": [], "side": []}
                        break
                
                if current_stage is None:
                    continue
                
                # 提取检查条件
                if "正面检查要点" in line or "侧面检查要点" in line:
                    continue
                
                front_check = None
                side_check = None
                
                # 分割正面和侧面检查点
                parts = line.split()
                if len(parts) > 0:
                    # 尝试解析行，提取正面和侧面检查点
                    text = line.strip()
                    
                    # 检查是否包含制表符作为分隔
                    if '\t' in text:
                        cols = text.split('\t')
                        # 删除空元素
                        cols = [c.strip() for c in cols if c.strip()]
                        
                        if len(cols) >= 2:
                            front_check = cols[-2] if len(cols) > 1 else None
                            side_check = cols[-1] if len(cols) > 2 else None
                    else:
                        # 没有明确的分隔符，使用启发式方法
                        # 通常正面检查点在前半部分，侧面检查点在后半部分
                        mid_point = len(text) // 2
                        front_check = text[:mid_point].strip()
                        side_check = text[mid_point:].strip()
                
                # 添加非空的检查条件
                if front_check and len(front_check) > 3 and current_stage in conditions:
                    conditions[current_stage]["front"].append(front_check)
                
                if side_check and len(side_check) > 3 and current_stage in conditions:
                    conditions[current_stage]["side"].append(side_check)
            
            return conditions
        
        except Exception as e:
            print(f"加载检查条件文件时出错: {str(e)}")
            return {}
    
    def _register_check_functions(self):
        """
        注册各种检查函数
        
        Returns:
            dict: 检查函数映射
        """
        # 这里使用函数名到函数的映射
        # 每个阶段的每个条件都会有一个专门的检查函数
        check_functions = {
            # 准备姿势阶段
            "0_front_0": self.check_shoulder_height,
            "0_front_1": self.check_left_arm_line,
            "0_front_2": self.check_feet_width,
            # "0_front_3": self.check_hand_grip_parallel,
            
            # 引杆阶段
            "1_front_0": self.check_club_head_consistent,
            
            # 起杆阶段 - 新增
            "2_front_0": self.check_club_horizontal,
            "2_front_1": self.check_left_arm_straight,
            "2_front_2": self.check_body_stability,
            
            # 上杆阶段 - 原来的"2"变成"3"
            "3_front_0": self.check_head_below_k_line,
            "3_front_1": self.check_trunk_within_feet_lines,
            
            # 上杆顶点阶段 - 原来的"3"变成"4"
            "4_front_0": self.check_club_beyond_head_line,
            
            # 下杆阶段 - 原来的"4"变成"5"
            "5_front_0": self.check_head_within_circle,
            "5_front_1": self.check_left_body_at_vertical_line,
            
            # 击球阶段 - 原来的"5"变成"6"
            "6_front_0": self.check_arm_triangle,
            
            # 送杆阶段 - 原来的"6"变成"7"
            "7_front_0": self.check_left_body_at_vertical_line,
            "7_front_1": self.check_arms_straight_crossed,
            "7_front_2": self.check_head_left_of_vertical_line,
            
            # 收杆阶段 - 原来的"7"变成"8" - 修复：添加完整的检查函数
            "8_front_0": self.check_left_body_at_vertical_line,

        }
        
        return check_functions
    
    def check_swing(self, keypoints, stage_indices, generate_visualizations=False, vis_dir=None, video_id=None):
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
        results = self._perform_swing_checks(keypoints, stage_indices)
        
        # 2. 如果需要生成可视化，使用检查结果生成可视化
        if generate_visualizations and vis_dir:
            self._generate_visualizations(keypoints, results, stage_indices, vis_dir, video_id)
        
        return results
    
    def _perform_swing_checks(self, keypoints, stage_indices):
        """
        执行挥杆检查，收集检查结果
        
        Args:
            keypoints: 关键点数据
            stage_indices: 各阶段对应的帧索引字典
            
        Returns:
            dict: 检查结果
        """
        results = {}
        
        # 遍历各个阶段
        for stage_key, frame_indices in stage_indices.items():
            if stage_key not in self.conditions:
                continue
            
            stage_name = STAGE_NAMES.get(stage_key, f"阶段{stage_key}")
            results[stage_key] = {
                "name": stage_name,
                "results": []
            }
            
            # 遍历该阶段的所有检查条件
            for view in ["front", "side"]:
                for i, condition in enumerate(self.conditions[stage_key][view]):
                    check_key = f"{stage_key}_{view}_{i}"
                    check_function = self.check_functions.get(check_key)
                    
                    if check_function is None:
                        # 使用默认函数
                        results[stage_key]["results"].append({
                            "condition": condition,
                            "view": view,
                            "passed": True,
                            "failed_frames": [],
                            "check_function": None
                        })
                        continue
                    
                    # 获取函数名称
                    check_function_name = check_function.__name__
                    
                    # 对该阶段的所有帧应用检查函数
                    failed_frames = []
                    frame_results = {}  # 存储每帧的检查结果
                    
                    for frame_idx in frame_indices:
                        if frame_idx >= len(keypoints):
                            continue
                        
                        frame_keypoints = keypoints[frame_idx]
                        
                        # 特殊处理需要额外参数的检查函数
                        if check_function_name == "check_head_left_of_vertical_line" and stage_key == "7":
                            check_result = check_function(frame_keypoints, stage_indices, keypoints)
                        elif check_function_name == "check_head_within_circle":
                            check_result = check_function(frame_keypoints, stage_indices, keypoints)
                        elif check_function_name == "check_left_body_at_vertical_line":
                            check_result = check_function(frame_keypoints, stage_indices, keypoints)
                        else:
                            check_result = check_function(frame_keypoints)
                            
                        frame_results[frame_idx] = check_result
                        
                        if not check_result:
                            failed_frames.append(frame_idx)
                    
                    check_passed = len(failed_frames) == 0
                    
                    # 添加检查结果
                    results[stage_key]["results"].append({
                        "condition": condition,
                        "view": view,
                        "passed": check_passed,
                        "failed_frames": failed_frames,
                        "check_function": check_function_name,
                        "frame_results": frame_results
                    })
        
        # ====== 新增other类别错误判断 ======
        # 判断挥杆上升（引杆到上杆顶点）与下降（顶点到击球）时间比例
        try:
            # 只要三个关键阶段都存在
            if ("1" in stage_indices and stage_indices["1"] and
                "4" in stage_indices and stage_indices["4"] and
                "6" in stage_indices and stage_indices["6"]):
                # 取引杆开始、上杆顶点、击球的关键帧
                takeaway_start = min(stage_indices["1"])
                top_of_swing = max(stage_indices["4"])
                impact = min(stage_indices["6"])
                # 计算两个阶段的帧数
                takeaway_to_top = top_of_swing - takeaway_start
                top_to_impact = impact - top_of_swing
                # 防止除零
                ratio = takeaway_to_top / top_to_impact if top_to_impact > 0 else 999
                # 合理比例范围（经验值3:1，容忍2.5~3.5）
                passed = 2.5 <= ratio <= 3.5
                # 详细注释：
                # 如果比例偏离3:1，说明下杆过快，疑似用手发力而非腰部旋转
                condition = "上升(引杆到顶点)与下降(顶点到击球)时间比例异常，疑似用手发力"
                # 结果写入other类别
                results["other"] = {
                    "name": "其他错误",
                    "results": [{
                        "condition": condition,
                        "view": "other",
                        "passed": passed,
                        "failed_frames": [],
                        "check_function": "check_swing_tempo_ratio",
                        "frame_results": {
                            "takeaway_to_top": takeaway_to_top,
                            "top_to_impact": top_to_impact,
                            "ratio": ratio
                        }
                    }]
                }
        except Exception as e:
            # 若阶段索引异常，跳过该判断
            results["other"] = {
                "name": "其他错误",
                "results": [{
                    "condition": "上升与下降时间比例判断异常，数据缺失",
                    "view": "other",
                    "passed": True,
                    "failed_frames": [],
                    "check_function": "check_swing_tempo_ratio",
                    "frame_results": {}
                }]
            }
        # ====== end other类别错误判断 ======
        return results
        
    def _generate_visualizations(self, keypoints, check_results, stage_indices, vis_dir, video_id):
        """
        根据检查结果生成可视化图像
        
        Args:
            keypoints: 关键点数据
            check_results: 检查结果
            stage_indices: 各阶段对应的帧索引字典
            vis_dir: 可视化图像保存目录
        """
        import cv2
        
        # 确保可视化目录存在
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)
            
        # 确定图像源目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base_dir, 'resultData', video_id, 'img', 'all')
        if not os.path.exists(img_dir):
            print(f"警告: 找不到原始图像目录: {img_dir}，将使用空白图像")
            
        # 创建阶段文件夹映射（不含中文）
        stage_dirs = {}
        for stage_key, frame_index in stage_indices.items():
            # 使用帧数作为目录名
            stage_dir_name = f"frame_{frame_index[0]}"
            stage_dirs[stage_key] = stage_dir_name
            
            # 创建对应的目录
            stage_path = os.path.join(vis_dir, stage_dir_name) 
            os.makedirs(stage_path, exist_ok=True)
            
        # 遍历各个阶段的检查结果
        for stage_key, stage_data in check_results.items():
            if stage_key not in stage_indices:
                continue
                
            frame_indices = stage_indices[stage_key]
            
            # 准备姿势阶段（阶段0）特殊处理 - 只生成一个可视化图片
            if stage_key == "0":
                # 获取第一帧作为参考帧
                if not frame_indices or frame_indices[0] >= len(keypoints):
                    continue
                    
                sample_idx = frame_indices[0]
                sample_keypoints = keypoints[sample_idx]
                
                # 收集未通过的检查项目
                failed_items = []
                for i, result in enumerate(stage_data["results"]):
                    if "check_function" not in result or result["check_function"] is None:
                        continue
                        
                    if not result["passed"]:
                        failed_items.append(result["check_function"])
                
                # 如果没有未通过的项目，跳过可视化
                if not failed_items:
                    continue
                
                # 尝试加载原始图像
                frame = None
                frame_file = os.path.join(img_dir, f"frame{sample_idx:04d}.jpg")
                if os.path.exists(frame_file):
                    frame = cv2.imread(frame_file)
                
                # 使用全英文文件名（不含中文字符）
                filename = f"setup_position_fails.jpg"
                save_path = os.path.join(vis_dir, stage_dirs[stage_key], filename)
                
                # 为准备姿势阶段生成统一的可视化图像
                self.visualize_setup_position(
                    sample_keypoints,
                    False,  # 结果为False表示有检查项未通过
                    frame=frame,
                    save_path=save_path,
                    stage_indices=stage_indices,
                    all_keypoints=keypoints
                )
                
                continue  # 跳过常规的每个检查项目单独生成可视化的逻辑
            
            # 其他阶段正常处理
            # 获取该阶段的所有检查结果
            for i, result in enumerate(stage_data["results"]):
                # 跳过没有检查函数的结果
                if "check_function" not in result or result["check_function"] is None:
                    continue
                
                check_function_name = result["check_function"]
                frame_results = result.get("frame_results", {})
                failed_frames = result["failed_frames"]
                
                # 简化检查函数名称，只保留最后一部分，并确保没有中文
                func_short_name = check_function_name.split('_')[-1] if '_' in check_function_name else check_function_name
                
                # 选择要可视化的帧
                sample_frames = []
                
                # 始终包含第一帧作为参考
                if frame_indices and frame_indices[0] < len(keypoints):
                    sample_frames.append(frame_indices[0])
                
                # 如果有失败的帧，最多添加2个作为样本
                for fail_idx in failed_frames[:2]:
                    if fail_idx not in sample_frames and fail_idx < len(keypoints):
                        sample_frames.append(fail_idx)
                
                # 为每个样本帧生成可视化图像
                for sample_idx in sample_frames:
                    sample_keypoints = keypoints[sample_idx]
                    sample_result = frame_results.get(sample_idx, True)  # 使用已计算的结果
                    
                    # 尝试加载原始图像
                    frame = None
                    frame_file = os.path.join(img_dir, f"frame{sample_idx:04d}.jpg")
                    if os.path.exists(frame_file):
                        frame = cv2.imread(frame_file)
                    
                    # 使用全英文文件名 (不含中文字符)
                    filename = f"f{sample_idx:04d}_{i}_{func_short_name}_{1 if sample_result else 0}.jpg"
                    save_path = os.path.join(vis_dir, stage_dirs[stage_key], filename)
                    
                    # 为当前检查函数生成可视化图像
                    self.visualize_check_result(
                        check_function_name,
                        sample_keypoints,
                        sample_result,
                        frame=frame,
                        save_path=save_path,
                        stage_name=STAGE_NAMES.get(stage_key, f"Stage {stage_key}"),
                        stage_indices=stage_indices,
                        all_keypoints=keypoints
                    )
    
    def generate_report(self, results, output_file):
        """
        生成检查报告
        
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
                
                for stage_key, stage_data in results.items():
                    f.write(f"阶段: {stage_data['name']} (阶段{stage_key})\n")
                    f.write("-" * 50 + "\n\n")
                    
                    for result in stage_data["results"]:
                        condition = result["condition"]
                        view = "正面" if result["view"] == "front" else "侧面"
                        passed = result["passed"]
                        failed_frames = result["failed_frames"]
                        
                        status = "通过" if passed else "不通过"
                        f.write(f"[{view}] {condition}\n")
                        f.write(f"状态: {status}\n")
                        
                        if not passed:
                            f.write(f"不通过的帧: {failed_frames}\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
                
                return True
        
        except Exception as e:
            print(f"生成报告时出错: {str(e)}")
            return False
    
    def generate_csv_report(self, results, output_file):
        """
        生成CSV格式的检查报告
        
        Args:
            results: 检查结果
            output_file: 输出文件路径
            
        Returns:
            bool: 是否成功生成报告
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 写入CSV头部
                f.write("阶段编号,阶段名称,视角,检查项目,检查结果,不通过帧序列\n")
                
                # 写入每个检查结果
                for stage_key, stage_data in results.items():
                    stage_name = stage_data['name']
                    
                    for result in stage_data["results"]:
                        condition = result["condition"]
                        view = "正面" if result["view"] == "front" else "侧面"
                        passed = result["passed"]
                        failed_frames = result["failed_frames"]
                        
                        status = "通过" if passed else "不通过"
                        failed_frames_str = ",".join(map(str, failed_frames)) if failed_frames else ""
                        
                        # 使用双引号包裹可能包含逗号的文本字段
                        f.write(f"{stage_key},\"{stage_name}\",{view},\"{condition}\",{status},\"{failed_frames_str}\"\n")
                
                return True
        
        except Exception as e:
            print(f"生成CSV报告时出错: {str(e)}")
            return False
    
    # ===== 以下是各种检查函数 =====
    
    def check_shoulder_height(self, keypoints):
        """检查左肩略高于右肩"""
        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        
        # 检查y坐标，注意屏幕坐标系中y轴向下为正方向
        result = left_shoulder[1] < right_shoulder[1]
        
        # 存储检查数据用于可视化
        self.setup_check_data = getattr(self, 'setup_check_data', {})
        self.setup_check_data['shoulder_height'] = {
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'result': result
        }
        
        return result
    
    def check_left_arm_line(self, keypoints):
        """检查左臂成一条直线且过大腿内侧"""
        # 获取左臂关键点
        left_shoulder = keypoints[LEFT_SHOULDER]
        left_elbow = keypoints[LEFT_ELBOW]
        left_wrist = keypoints[LEFT_WRIST]
        
        # 获取左腿内侧点（近似为左髋点）
        left_hip = keypoints[LEFT_HIP]
        right_hip = keypoints[RIGHT_HIP]
        
        # 计算左肩到左腕的直线
        # 使用直线方程 y = mx + b
        if abs(left_wrist[0] - left_shoulder[0]) < 0.001:  # 避免除以零
            m = 1000  # 近似垂直线
        else:
            m = (left_wrist[1] - left_shoulder[1]) / (left_wrist[0] - left_shoulder[0])
        
        b = left_shoulder[1] - m * left_shoulder[0]
        
        # 计算左肘到直线的距离，判断左臂是否成一条直线
        # 点到直线距离公式：|Ax + By + C|/sqrt(A^2 + B^2)，其中直线方程为Ax + By + C = 0
        A = m
        B = -1
        C = b
        elbow_dist = abs(A * left_elbow[0] + B * left_elbow[1] + C) / math.sqrt(A*A + B*B)
        
        # 计算大腿内侧点（取两髋中点再向下偏移）
        hip_center = (left_hip + right_hip) / 2
        thigh_inner = hip_center.copy()
        thigh_inner[1] += 0.05  # 向下偏移5%
        
        # 检查直线是否经过大腿内侧
        # 计算大腿内侧点到直线的距离
        thigh_dist = abs(A * thigh_inner[0] + B * thigh_inner[1] + C) / math.sqrt(A*A + B*B)
        
        # 设定阈值
        arm_straight = elbow_dist < 0.05  # 左臂直线度阈值
        pass_thigh = thigh_dist < 0.1  # 经过大腿内侧阈值
        
        result = arm_straight and pass_thigh
        
        # 存储检查数据用于可视化
        self.setup_check_data = getattr(self, 'setup_check_data', {})
        self.setup_check_data['left_arm_line'] = {
            'left_shoulder': left_shoulder,
            'left_elbow': left_elbow,
            'left_wrist': left_wrist,
            'hip_center': hip_center,
            'thigh_inner': thigh_inner,
            'line_params': (m, b),
            'elbow_dist': elbow_dist,
            'thigh_dist': thigh_dist,
            'arm_straight': arm_straight,
            'pass_thigh': pass_thigh,
            'result': result
        }
        
        return result
    
    def check_feet_width(self, keypoints):
        """检查双脚宽度与肩同宽"""
        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        left_ankle = keypoints[LEFT_ANKLE]
        right_ankle = keypoints[RIGHT_ANKLE]
        
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        feet_width = np.linalg.norm(left_ankle[:2] - right_ankle[:2])
        
        # 计算比例
        ratio = feet_width / shoulder_width if shoulder_width > 0 else 999
        
        # 允许20%的误差
        result = 0.8 <= ratio <= 1.2
        
        # 存储检查数据用于可视化
        self.setup_check_data = getattr(self, 'setup_check_data', {})
        self.setup_check_data['feet_width'] = {
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'left_ankle': left_ankle,
            'right_ankle': right_ankle,
            'shoulder_width': shoulder_width,
            'feet_width': feet_width,
            'ratio': ratio,
            'result': result
        }
        
        return result
    
    def check_hand_grip_parallel(self, keypoints):
        """检查两手虎口线平行且指向右耳或右肩"""
        # 获取手腕和肩膀位置
        left_wrist = keypoints[LEFT_WRIST]
        right_wrist = keypoints[RIGHT_WRIST]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        right_ear = keypoints[RIGHT_EYE]  # 使用右眼作为右耳的近似位置
        
        # 计算手腕连线的方向向量
        wrist_vec = right_wrist - left_wrist
        wrist_vec = wrist_vec / np.linalg.norm(wrist_vec)  # 归一化
        
        # 计算从左手腕到右肩/右耳的方向向量
        wrist_to_shoulder_vec = right_shoulder - left_wrist
        wrist_to_shoulder_vec = wrist_to_shoulder_vec / np.linalg.norm(wrist_to_shoulder_vec)  # 归一化
        
        wrist_to_ear_vec = right_ear - left_wrist
        wrist_to_ear_vec = wrist_to_ear_vec / np.linalg.norm(wrist_to_ear_vec)  # 归一化
        
        # 计算向量之间的夹角
        dot_shoulder = np.dot(wrist_vec, wrist_to_shoulder_vec)
        angle_shoulder = math.acos(np.clip(dot_shoulder, -1.0, 1.0)) * 180 / math.pi
        
        dot_ear = np.dot(wrist_vec, wrist_to_ear_vec)
        angle_ear = math.acos(np.clip(dot_ear, -1.0, 1.0)) * 180 / math.pi
        
        # 判断手腕连线是否指向右肩或右耳（允许20度误差）
        points_to_shoulder = angle_shoulder < 20
        points_to_ear = angle_ear < 20
        
        result = points_to_shoulder or points_to_ear
        
        # 存储检查数据用于可视化
        self.setup_check_data = getattr(self, 'setup_check_data', {})
        self.setup_check_data['hand_grip'] = {
            'left_wrist': left_wrist,
            'right_wrist': right_wrist,
            'right_shoulder': right_shoulder,
            'right_ear': right_ear,
            'wrist_vec': wrist_vec,
            'angle_shoulder': angle_shoulder,
            'angle_ear': angle_ear,
            'points_to_shoulder': points_to_shoulder,
            'points_to_ear': points_to_ear,
            'result': result
        }
        
        return result

    def visualize_setup_position(self, keypoints, result, frame=None, save_path=None, 
                                 stage_indices=None, all_keypoints=None):
        """
        可视化准备姿势阶段的辅助线，只显示未通过的检查项目
        
        根据检查结果选择性绘制辅助线：
        1. 肩膀高度没通过 => 只绘制肩膀连接线
        2. 双脚宽度没通过 => 绘制肩膀连接线和脚步连接线
        3. 左臂直线没通过 => 只绘制左臂连接线
        4. 虎口平行线没通过 => 只绘制虎口平行线
        
        Args:
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            stage_indices: 阶段索引数据
            all_keypoints: 所有帧的关键点数据
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 如果没有检查数据，重新执行检查
        if not hasattr(self, 'setup_check_data'):
            self.check_shoulder_height(keypoints)
            self.check_left_arm_line(keypoints)
            self.check_feet_width(keypoints)
            self.check_hand_grip_parallel(keypoints)
        
        # 获取各项检查结果
        shoulder_check = self.setup_check_data.get('shoulder_height', {})
        arm_check = self.setup_check_data.get('left_arm_line', {})
        feet_check = self.setup_check_data.get('feet_width', {})
        grip_check = self.setup_check_data.get('hand_grip', {})
        
        shoulder_result = shoulder_check.get('result', True)
        arm_result = arm_check.get('result', True)
        feet_result = feet_check.get('result', True)
        grip_result = grip_check.get('result', True)
        
        # 检查是否所有项目都通过了
        all_passed = shoulder_result and arm_result and feet_result and grip_result
        
        # 如果所有项目都通过了，就不生成可视化图片
        if all_passed and save_path:
            print("所有检查项目均通过，不生成可视化图片")
            return None
        
        # 转换关键点为像素坐标
        left_shoulder = (adjusted_keypoints[LEFT_SHOULDER][:2] * [width, height]).astype(int)
        right_shoulder = (adjusted_keypoints[RIGHT_SHOULDER][:2] * [width, height]).astype(int)
        left_elbow = (adjusted_keypoints[LEFT_ELBOW][:2] * [width, height]).astype(int)
        right_elbow = (adjusted_keypoints[RIGHT_ELBOW][:2] * [width, height]).astype(int)
        left_wrist = (adjusted_keypoints[LEFT_WRIST][:2] * [width, height]).astype(int)
        right_wrist = (adjusted_keypoints[RIGHT_WRIST][:2] * [width, height]).astype(int)
        left_hip = (adjusted_keypoints[LEFT_HIP][:2] * [width, height]).astype(int)
        right_hip = (adjusted_keypoints[RIGHT_HIP][:2] * [width, height]).astype(int)
        left_ankle = (adjusted_keypoints[LEFT_ANKLE][:2] * [width, height]).astype(int)
        right_ankle = (adjusted_keypoints[RIGHT_ANKLE][:2] * [width, height]).astype(int)
        right_eye = (adjusted_keypoints[RIGHT_EYE][:2] * [width, height]).astype(int)
        
        # 绘制基本骨架 - 使用灰色线条
        skeleton_color = (150, 150, 150)  # 灰色
        # 绘制躯干
        cv2.line(frame, tuple(left_shoulder), tuple(right_shoulder), skeleton_color, 1)
        cv2.line(frame, tuple(left_shoulder), tuple(left_hip), skeleton_color, 1)
        cv2.line(frame, tuple(right_shoulder), tuple(right_hip), skeleton_color, 1)
        cv2.line(frame, tuple(left_hip), tuple(right_hip), skeleton_color, 1)
        
        # 绘制手臂
        cv2.line(frame, tuple(left_shoulder), tuple(left_elbow), skeleton_color, 1)
        cv2.line(frame, tuple(left_elbow), tuple(left_wrist), skeleton_color, 1)
        cv2.line(frame, tuple(right_shoulder), tuple(right_elbow), skeleton_color, 1)
        cv2.line(frame, tuple(right_elbow), tuple(right_wrist), skeleton_color, 1)
        
        # 绘制腿部
        cv2.line(frame, tuple(left_hip), tuple(left_ankle), skeleton_color, 1)
        cv2.line(frame, tuple(right_hip), tuple(right_ankle), skeleton_color, 1)
        
        # 只绘制未通过的检查项目的辅助线
        
        # 1. 肩膀高度未通过 => 绘制肩膀连接线
        if not shoulder_result:
            shoulder_color = (0, 0, 255)  # 红色
            cv2.line(frame, tuple(left_shoulder), tuple(right_shoulder), shoulder_color, 3)
        
        # 2. 双脚宽度未通过 => 绘制肩膀连接线和脚步连接线
        if not feet_result:
            shoulder_width_color = (0, 0, 255)  # 红色
            feet_width_color = (0, 0, 255)  # 红色
            
            # 绘制肩膀连接线
            cv2.line(frame, tuple(left_shoulder), tuple(right_shoulder), shoulder_width_color, 3)
            
            # 绘制双脚连接线
            cv2.line(frame, tuple(left_ankle), tuple(right_ankle), feet_width_color, 3)
            
            # 可选：绘制宽度比例标注
            if 'ratio' in feet_check:
                ratio = feet_check['ratio']
                cv2.putText(frame, f"{ratio:.2f}x", 
                            ((left_ankle[0] + right_ankle[0]) // 2, (left_ankle[1] + right_ankle[1]) // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, feet_width_color, 2)
        
        # 3. 左臂直线未通过 => 绘制左臂连接线
        if not arm_result:
            arm_color = (0, 0, 255)  # 红色
            
            # 绘制左臂实际线段
            cv2.line(frame, tuple(left_shoulder), tuple(left_elbow), arm_color, 3)
            cv2.line(frame, tuple(left_elbow), tuple(left_wrist), arm_color, 3)
            
            # 绘制左臂理想直线
            if 'line_params' in arm_check:
                m, b = arm_check['line_params']
                # 延长线的起点和终点
                x1 = 0
                y1 = int(m * x1 + b)
                x2 = width
                y2 = int(m * x2 + b)
                
                # 检查延长线是否经过画面
                if 0 <= y1 <= height or 0 <= y2 <= height:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # 紫色理想线
        
        # 4. 虎口平行线未通过 => 绘制虎口平行线
        if not grip_result:
            grip_color = (0, 0, 255)  # 红色
            
            # 绘制手腕连接线
            cv2.line(frame, tuple(left_wrist), tuple(right_wrist), grip_color, 3)
            
            if 'wrist_vec' in grip_check:
                wrist_vec = grip_check['wrist_vec']
                # 计算虎口线方向
                # 从左手腕开始，沿着虎口线方向延伸
                line_length = width * 0.5  # 线长为画面宽度的一半
                dx = int(wrist_vec[0] * line_length)
                dy = int(wrist_vec[1] * line_length)
                
                # 绘制虎口线
                end_point = (left_wrist[0] + dx, left_wrist[1] + dy)
                cv2.line(frame, tuple(left_wrist), end_point, (0, 255, 255), 3)  # 黄色
                
                # 绘制理想指向线
                cv2.line(frame, tuple(left_wrist), tuple(right_eye), (255, 0, 255), 2)  # 紫色指向右耳
                cv2.line(frame, tuple(left_wrist), tuple(right_shoulder), (255, 0, 255), 2)  # 紫色指向右肩
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame
    
    def check_club_head_consistent(self, keypoints):
        """检查杆头趾部横截面延长线保持一致"""
        # 需要杆的信息，简化检查
        return True
    
    def check_club_not_above_hip_line(self, keypoints):
        """检查杆身平行时不能高于髋关节连线水平"""
        # 需要杆的信息，简化检查
        return True
    
    def check_left_arm_straight(self, keypoints):
        """检查左臂不能弯曲"""
        left_shoulder = keypoints[LEFT_SHOULDER]
        left_elbow = keypoints[LEFT_ELBOW]
        left_wrist = keypoints[LEFT_WRIST]
        
        # 计算左肩到左肘和左肘到左腕的向量
        v1 = left_elbow - left_shoulder
        v2 = left_wrist - left_elbow
        
        # 计算两个向量的夹角
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = min(1.0, max(-1.0, cos_angle))  # 确保在[-1, 1]范围内
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        # 如果接近180度，说明左臂伸直
        return angle > 160
    
    def check_club_horizontal(self, keypoints):
        """
        检查球杆是否水平
        
        Args:
            keypoints: 当前帧关键点坐标
            
        Returns:
            bool: 是否通过检查
        """
        # 检查是否有球杆数据
        if len(keypoints) < 35:
            return True  # 没有球杆数据，默认通过
        
        # 获取球杆关键点
        club_grip = keypoints[33][:2]  # 球杆握把端
        club_head = keypoints[34][:2]  # 球杆头端
        
        # 检查关键点是否有效
        if (club_grip[0] <= 0 or club_grip[1] <= 0 or 
            club_head[0] <= 0 or club_head[1] <= 0):
            return True  # 关键点无效，默认通过
        
        # 计算球杆与水平线的夹角
        dx = club_head[0] - club_grip[0]
        dy = club_head[1] - club_grip[1]
        
        if abs(dx) < 1e-6:  # 避免除零错误
            return True
            
        # 计算角度（弧度转角度）
        angle = np.arctan2(abs(dy), abs(dx)) * 180 / np.pi
        
        # 检查球杆是否接近水平（±15度）
        return angle <= 15.0
    
    def check_body_stability(self, keypoints):
        """
        检查身体重心稳定性
        
        判断标准：
        1. 双脚位置保持稳定
        2. 髋部中心位置变化不大
        3. 上身角度保持相对稳定
        
        Args:
            keypoints: 关键点数据
            
        Returns:
            bool: 是否通过检查
        """
        # 获取关键身体部位
        left_hip = keypoints[LEFT_HIP][:2]
        right_hip = keypoints[RIGHT_HIP][:2]
        left_shoulder = keypoints[LEFT_SHOULDER][:2]
        right_shoulder = keypoints[RIGHT_SHOULDER][:2]
        
        # 检查关键点是否有效
        if (left_hip[0] <= 0 or left_hip[1] <= 0 or 
            right_hip[0] <= 0 or right_hip[1] <= 0 or
            left_shoulder[0] <= 0 or left_shoulder[1] <= 0 or
            right_shoulder[0] <= 0 or right_shoulder[1] <= 0):
            return True  # 关键点无效，默认通过
        
        # 计算髋部中心
        hip_center = (left_hip + right_hip) / 2
        
        # 计算肩部中心
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # 计算上身角度（肩部连线与水平线的夹角）
        shoulder_dx = right_shoulder[0] - left_shoulder[0]
        shoulder_dy = right_shoulder[1] - left_shoulder[1]
        shoulder_angle = np.arctan2(abs(shoulder_dy), abs(shoulder_dx)) * 180 / np.pi
        
        # 简化检查：如果肩部角度不超过30度，认为身体稳定
        return shoulder_angle <= 30.0
    
    def check_head_below_k_line(self, keypoints, stage_indices=None, all_keypoints=None):
        """
        检查头部不超过平行地面的K线
        
        使用阶段0的头顶位置作为参考K线，检查当前帧头部是否超过该线
        允许头部长度10%的上下位移
        
        Args:
            keypoints: 当前帧关键点坐标
            stage_indices: 各阶段帧索引
            all_keypoints: 所有帧的关键点数据
            
        Returns:
            bool: 是否通过检查
        """
        # 如果没有提供阶段索引或所有关键点数据，无法建立参考线
        if stage_indices is None or all_keypoints is None or "0" not in stage_indices:
            return True  # 无法检查则默认通过
        
        # 获取阶段0的第一帧作为参考
        stage0_idx = stage_indices["0"][0]
        stage0_keypoints = all_keypoints[stage0_idx]
        
        # 使用与透明辅助线同样的逻辑计算头顶位置
        # 定义面部关键点索引（鼻子、眼睛、耳朵、嘴）
        face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 面部关键点
        
        # 获取阶段0面部关键点
        stage0_face_points = []
        for idx in face_indices:
            if idx < len(stage0_keypoints):
                point = stage0_keypoints[idx][:2]
                if point[0] > 0 and point[1] > 0:  # 过滤无效点
                    stage0_face_points.append(point)
        
        # 获取当前帧面部关键点
        current_face_points = []
        for idx in face_indices:
            if idx < len(keypoints):
                point = keypoints[idx][:2]
                if point[0] > 0 and point[1] > 0:  # 过滤无效点
                    current_face_points.append(point)
        
        # 如果面部关键点不足，使用备用方法
        if not stage0_face_points or not current_face_points:
            return True  # 关键点不足，默认通过
        
        # 计算阶段0面部边界
        stage0_min_y = min(p[1] for p in stage0_face_points)  # 面部最高点
        stage0_max_y = max(p[1] for p in stage0_face_points)  # 面部最低点
        stage0_face_height = stage0_max_y - stage0_min_y
        
        # 计算阶段0头顶位置：最小y值向上延伸2.3倍face_height
        stage0_offset = 2.3 * stage0_face_height
        stage0_head_top_y = stage0_min_y - stage0_offset
        
        # 计算当前帧面部边界
        current_min_y = min(p[1] for p in current_face_points)  # 面部最高点
        current_max_y = max(p[1] for p in current_face_points)  # 面部最低点
        current_face_height = current_max_y - current_min_y
        
        # 计算当前头顶位置
        current_offset = 2.3 * current_face_height
        current_head_top_y = current_min_y - current_offset
        
        # 计算垂直位移
        vertical_displacement = stage0_head_top_y - current_head_top_y
        
        # 计算允许的最大位移（头部长度的10%）
        max_allowed_displacement = 0.1 * stage0_face_height
        
        # 存储检查数据供可视化使用
        self.k_line_data = {
            "stage0_head_top_y": stage0_head_top_y,
            "current_head_top_y": current_head_top_y,
            "vertical_displacement": vertical_displacement,
            "max_allowed_displacement": max_allowed_displacement,
            "stage0_face_height": stage0_face_height
        }
        
        # 检查头部是否超过K线（考虑允许的位移）
        # 如果垂直位移为正，表示当前头部高于阶段0头部
        # 如果垂直位移为负，表示当前头部低于阶段0头部
        # 只有当头部位置高于阶段0头部，且超过允许范围时，才判定为不通过
        return vertical_displacement <= max_allowed_displacement

    def visualize_head_below_k_line(self, keypoints, result, frame=None, save_path=None, 
                                    stage_indices=None, all_keypoints=None):
        """
        可视化头部位置相对于K线的检查结果
        
        Args:
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            stage_indices: 阶段索引数据
            all_keypoints: 所有帧的关键点数据
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 如果没有运行过检查函数或没有必要数据，重新计算
        if not hasattr(self, 'k_line_data') or self.k_line_data is None:
            _ = self.check_head_below_k_line(keypoints, stage_indices, all_keypoints)
            if not hasattr(self, 'k_line_data') or self.k_line_data is None:
                # 如果仍然没有数据，显示错误信息并返回
                if save_path:
                    self._save_image_safely(frame, save_path)
                return frame
        
        # 获取检查数据
        stage0_head_top_y = self.k_line_data["stage0_head_top_y"]
        current_head_top_y = self.k_line_data["current_head_top_y"]
        vertical_displacement = self.k_line_data["vertical_displacement"]
        max_allowed_displacement = self.k_line_data["max_allowed_displacement"]
        
        # 将相对坐标转换为像素坐标
        stage0_head_line_y = int(stage0_head_top_y * height)
        current_head_line_y = int(current_head_top_y * height)
        
        # 计算面部中心位置（用于绘制线条）
        nose_px = (adjusted_keypoints[NOSE][:2] * [width, height]).astype(int)
        face_center_x = nose_px[0]
        
        # 计算线条长度：线条长度为画面宽度的1/3
        line_width = width // 3
        line_start_x = face_center_x - line_width // 2
        line_end_x = face_center_x + line_width // 2
        
        # 绘制阶段0头顶线（K线，红色）
        cv2.line(frame, (line_start_x, stage0_head_line_y), (line_end_x, stage0_head_line_y), (0, 0, 255), 2)
        
        # 绘制当前头顶线（蓝色）
        cv2.line(frame, (line_start_x, current_head_line_y), (line_end_x, current_head_line_y), (255, 0, 0), 2)
        
        # 绘制允许位移的范围线（绿色虚线）
        max_displacement_y = stage0_head_line_y + int(max_allowed_displacement * height)
        self._draw_dashed_line(frame, (line_start_x, max_displacement_y), (line_end_x, max_displacement_y), (0, 255, 0), 1)
        
        # 绘制从当前头顶到阶段0头顶的垂直连线
        cv2.line(frame, (face_center_x, current_head_line_y), (face_center_x, stage0_head_line_y), (255, 0, 255), 2)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame
    
    def check_trunk_within_feet_lines(self, keypoints, stage_indices=None, all_keypoints=None):
        """
        检查躯干保持在双脚外侧垂直线内
        
        使用阶段0作为基准帧，建立左右两侧垂直参考线
        判断当前帧躯干边缘点是否保持在这两条参考线之内
        
        Args:
            keypoints: 当前帧关键点坐标
            stage_indices: 各阶段帧索引
            all_keypoints: 所有帧的关键点数据
            
        Returns:
            bool: 是否通过检查
        """
        # 如果没有提供阶段索引或所有关键点数据，无法建立参考线
        if stage_indices is None or all_keypoints is None or "0" not in stage_indices:
            # 使用简单版本的检查
            nose = keypoints[NOSE]
            left_shoulder = keypoints[LEFT_SHOULDER]
            right_shoulder = keypoints[RIGHT_SHOULDER]
            left_foot = keypoints[LEFT_FOOT_INDEX]
            right_foot = keypoints[RIGHT_FOOT_INDEX]
        
        # 计算肩膀中点
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # 检查肩膀中点是否在脚的范围内
        result = left_foot[0] <= shoulder_center[0] <= right_foot[0]
            
        # 存储检查数据（用于可视化）
        self.trunk_lines_data = {
            "use_stage0": False,
            "left_boundary": left_foot[0],
            "right_boundary": right_foot[0],
            "shoulder_center": shoulder_center[0],
            "left_edge": left_shoulder[0],
            "right_edge": right_shoulder[0]
        }
        
        return result
        
        # 获取阶段0的第一帧作为参考
        stage0_idx = stage_indices["0"][0]
        stage0_keypoints = all_keypoints[stage0_idx]
        
        # 提取阶段0关键点
        stage0_left_shoulder = stage0_keypoints[LEFT_SHOULDER][:2]
        stage0_right_shoulder = stage0_keypoints[RIGHT_SHOULDER][:2]
        stage0_left_hip = stage0_keypoints[LEFT_HIP][:2]
        stage0_right_hip = stage0_keypoints[RIGHT_HIP][:2]
        
        # 计算阶段0的躯干边缘
        stage0_left_edge = min(stage0_left_shoulder[0], stage0_left_hip[0])
        stage0_right_edge = max(stage0_right_shoulder[0], stage0_right_hip[0])
        
        # 提取当前帧躯干边缘
        left_shoulder = keypoints[LEFT_SHOULDER][:2]
        right_shoulder = keypoints[RIGHT_SHOULDER][:2]
        left_hip = keypoints[LEFT_HIP][:2]
        right_hip = keypoints[RIGHT_HIP][:2]
        
        # 计算当前帧的躯干边缘
        current_left_edge = min(left_shoulder[0], left_hip[0])
        current_right_edge = max(right_shoulder[0], right_hip[0])
        
        # 计算躯干宽度，用于设置容错范围
        stage0_trunk_width = stage0_right_edge - stage0_left_edge
        
        # 设置参考线偏移量（向外扩展5%躯干宽度）
        offset = stage0_trunk_width * 0.05
        
        # 计算左右边界线位置
        left_boundary = stage0_left_edge - offset
        right_boundary = stage0_right_edge + offset
        
        # 检查当前帧躯干边缘是否在边界线内
        within_left = current_left_edge >= left_boundary
        within_right = current_right_edge <= right_boundary
        result = within_left and within_right
        
        # 存储检查数据（用于可视化）
        self.trunk_lines_data = {
            "use_stage0": True,
            "left_boundary": left_boundary,
            "right_boundary": right_boundary,
            "stage0_left_edge": stage0_left_edge,
            "stage0_right_edge": stage0_right_edge,
            "current_left_edge": current_left_edge,
            "current_right_edge": current_right_edge,
            "within_left": within_left,
            "within_right": within_right,
            "offset": offset
        }
        
        return result

    def visualize_trunk_within_feet_lines(self, keypoints, result, frame=None, save_path=None, 
                                          stage_indices=None, all_keypoints=None):
        """
        可视化躯干是否保持在垂直边界线内
        
        Args:
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            stage_indices: 阶段索引数据
            all_keypoints: 所有帧的关键点数据
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 如果没有运行过检查函数或没有必要数据，重新计算
        if not hasattr(self, 'trunk_lines_data') or self.trunk_lines_data is None:
            _ = self.check_trunk_within_feet_lines(keypoints, stage_indices, all_keypoints)
            if not hasattr(self, 'trunk_lines_data') or self.trunk_lines_data is None:
                # 如果仍然没有数据，显示错误信息并返回
                if save_path:
                    self._save_image_safely(frame, save_path)
                return frame
        
        # 获取检查数据
        use_stage0 = self.trunk_lines_data.get("use_stage0", False)
        
        # 将关键点转换为像素坐标
        left_shoulder = (adjusted_keypoints[LEFT_SHOULDER][:2] * [width, height]).astype(int)
        right_shoulder = (adjusted_keypoints[RIGHT_SHOULDER][:2] * [width, height]).astype(int)
        left_hip = (adjusted_keypoints[LEFT_HIP][:2] * [width, height]).astype(int)
        right_hip = (adjusted_keypoints[RIGHT_HIP][:2] * [width, height]).astype(int)
        
        # 获取脚部位置
        left_ankle = (adjusted_keypoints[LEFT_ANKLE][:2] * [width, height]).astype(int)
        right_ankle = (adjusted_keypoints[RIGHT_ANKLE][:2] * [width, height]).astype(int)
        left_foot = (adjusted_keypoints[LEFT_FOOT_INDEX][:2] * [width, height]).astype(int)
        right_foot = (adjusted_keypoints[RIGHT_FOOT_INDEX][:2] * [width, height]).astype(int)
        left_heel = (adjusted_keypoints[LEFT_HEEL][:2] * [width, height]).astype(int) if LEFT_HEEL < len(adjusted_keypoints) else left_ankle
        right_heel = (adjusted_keypoints[RIGHT_HEEL][:2] * [width, height]).astype(int) if RIGHT_HEEL < len(adjusted_keypoints) else right_ankle
        
        # 计算脚掌长度 (使用脚趾和脚跟之间的水平距离)
        left_foot_length = abs(left_foot[0] - left_heel[0])
        right_foot_length = abs(right_foot[0] - right_heel[0])
        avg_foot_length = (left_foot_length + right_foot_length) / 2
        half_foot_length = avg_foot_length  # 半个脚掌长度
        
        # 获取脚部最低点
        foot_bottom_y = max(left_ankle[1], right_ankle[1], left_foot[1], right_foot[1])
        
        # 绘制躯干轮廓
        cv2.line(frame, tuple(left_shoulder), tuple(right_shoulder), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_shoulder), tuple(left_hip), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_shoulder), tuple(right_hip), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_hip), tuple(right_hip), (0, 0, 255), 2)
        
        # 绘制躯干边缘点
        cv2.circle(frame, tuple(left_shoulder), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(right_shoulder), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(left_hip), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(right_hip), 5, (0, 0, 255), -1)
        
        if use_stage0:
            # 使用阶段0的边界线
            left_boundary = int(self.trunk_lines_data["left_boundary"] * width)
            right_boundary = int(self.trunk_lines_data["right_boundary"] * width)
            
            # 向两侧各延伸半个脚掌长度
            left_boundary -= int(half_foot_length)
            right_boundary += int(half_foot_length)
            
            # 计算垂直线的起止点
            min_y_point = min(left_shoulder[1], right_shoulder[1])
            
            # 减小线条向上延伸长度，但向下延伸到脚部
            top_extend = int((right_shoulder[1] - min_y_point) * 0.1)
            
            line_top = max(0, min_y_point - top_extend)
            line_bottom = foot_bottom_y  # 延伸到脚部
            
            # 绘制左边界线（红色）
            cv2.line(frame, (left_boundary, line_top), (left_boundary, line_bottom), (0, 0, 255), 2)
            
            # 绘制右边界线（红色）
            cv2.line(frame, (right_boundary, line_top), (right_boundary, line_bottom), (0, 0, 255), 2)
            
            # 标记当前躯干边缘位置
            current_left_edge = int(self.trunk_lines_data["current_left_edge"] * width)
            current_right_edge = int(self.trunk_lines_data["current_right_edge"] * width)
            
            # 绘制当前躯干边缘线（绿色或红色，取决于是否在边界内）
            left_color = (0, 255, 0) if self.trunk_lines_data["within_left"] else (0, 0, 255)
            right_color = (0, 255, 0) if self.trunk_lines_data["within_right"] else (0, 0, 255)
            
            # 左边缘垂直线
            cv2.line(frame, (current_left_edge, line_top), (current_left_edge, line_bottom), left_color, 2)
            
            # 右边缘垂直线
            cv2.line(frame, (current_right_edge, line_top), (current_right_edge, line_bottom), right_color, 2)
        else:
            # 使用当前帧的脚部位置作为边界
            left_boundary = int(self.trunk_lines_data["left_boundary"] * width)
            right_boundary = int(self.trunk_lines_data["right_boundary"] * width)
            
            # 向两侧各延伸半个脚掌长度
            left_boundary += int(half_foot_length)
            right_boundary -= int(half_foot_length)
            
            # 计算垂直线的起止点
            min_y_point = min(left_shoulder[1], right_shoulder[1])
            
            # 减小线条向上延伸长度，但向下延伸到脚部
            top_extend = int((right_shoulder[1] - min_y_point) * 0.1)
            
            line_top = max(0, min_y_point - top_extend)
            line_bottom = foot_bottom_y  # 延伸到脚部
            
            # 绘制左右边界线（红色，延伸到脚部）
            cv2.line(frame, (left_boundary, line_top), (left_boundary, line_bottom), (0, 0, 255), 2)
            cv2.line(frame, (right_boundary, line_top), (right_boundary, line_bottom), (0, 0, 255), 2)
            
            # 标记肩膀中点
            shoulder_center = int(self.trunk_lines_data["shoulder_center"] * width)
            cv2.circle(frame, (shoulder_center, (left_shoulder[1] + right_shoulder[1]) // 2), 5, (0, 255, 0), -1)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame
    
    def check_club_beyond_head_line(self, keypoints, stage_indices=None, all_keypoints=None):
        """
        检查杆身超过头部平行线
        
        使用阶段0的头部位置作为参考，检查当前帧手部是否高于该位置
        
        Args:
            keypoints: 当前帧关键点坐标
            stage_indices: 各阶段帧索引
            all_keypoints: 所有帧的关键点数据
            
        Returns:
            bool: 是否通过检查
        """
        # 获取当前帧手腕位置
        left_wrist = keypoints[LEFT_WRIST][:2]
        right_wrist = keypoints[RIGHT_WRIST][:2]
        
        # 选择较高的手腕位置（Y坐标较小）
        hand_y = min(left_wrist[1], right_wrist[1])
        
        # 参考头顶位置
        head_top_y = None
        
        # 如果提供了阶段索引和所有关键点，则使用阶段0的头部位置作为参考
        if stage_indices is not None and all_keypoints is not None and "0" in stage_indices:
            # 获取阶段0的第一帧作为参考
            stage0_idx = stage_indices["0"][0]
            stage0_keypoints = all_keypoints[stage0_idx]
            
            # 使用透明辅助线同样的逻辑计算头顶位置
            # 定义面部关键点索引（鼻子、眼睛、耳朵、嘴）
            face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 面部关键点
            
            # 获取面部关键点坐标
            face_points = []
            for idx in face_indices:
                if idx < len(stage0_keypoints):
                    point = stage0_keypoints[idx][:2]
                    if point[0] > 0 and point[1] > 0:  # 过滤无效点
                        face_points.append(point)
            
            if face_points:
                # 计算面部边界
                min_y = min(p[1] for p in face_points)  # 面部最高点
                max_y = max(p[1] for p in face_points)  # 面部最低点
                face_height = max_y - min_y
                
                # 根据经验，将头顶估计为：最小y值向上延伸2.3倍face_height
                offset = 2.3 * face_height
                head_top_y = min_y - offset
            else:
                # 如果没有足够的面部关键点，使用原始方法作为后备
                nose_y = stage0_keypoints[NOSE][1]
                left_eye_y = stage0_keypoints[LEFT_EYE][1]
                right_eye_y = stage0_keypoints[RIGHT_EYE][1]
                
                eyes_avg_y = (left_eye_y + right_eye_y) / 2
                nose_to_eyes = nose_y - eyes_avg_y
                
                head_top_y = eyes_avg_y - (nose_to_eyes * 2.3)
            
            # 存储头顶位置用于可视化
            self.head_line_data = {
                "head_top_y": head_top_y
            }
        else:
            # 如果没有阶段0数据，使用当前帧估算
            # 使用透明辅助线同样的逻辑计算头顶位置
            # 定义面部关键点索引（鼻子、眼睛、耳朵、嘴）
            face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 面部关键点
            
            # 获取面部关键点坐标
            face_points = []
            for idx in face_indices:
                if idx < len(keypoints):
                    point = keypoints[idx][:2]
                    if point[0] > 0 and point[1] > 0:  # 过滤无效点
                        face_points.append(point)
            
            if face_points:
                # 计算面部边界
                min_y = min(p[1] for p in face_points)  # 面部最高点
                max_y = max(p[1] for p in face_points)  # 面部最低点
                face_height = max_y - min_y
                
                # 根据经验，将头顶估计为：最小y值向上延伸2.3倍face_height
                offset = 2.3 * face_height
                head_top_y = min_y - offset
            else:
                # 如果没有足够的面部关键点，使用原始方法作为后备
                nose_y = keypoints[NOSE][1]
                left_eye_y = keypoints[LEFT_EYE][1]
                right_eye_y = keypoints[RIGHT_EYE][1]
                
                eyes_avg_y = (left_eye_y + right_eye_y) / 2
                nose_to_eyes = nose_y - eyes_avg_y
                
                head_top_y = eyes_avg_y - (nose_to_eyes * 2.3)
            
            self.head_line_data = {
                "head_top_y": head_top_y
            }
        
        # 检查手部位置是否高于头顶位置
        return hand_y < head_top_y

    def visualize_club_beyond_head_line(self, keypoints, result, frame=None, save_path=None, 
                                        stage_indices=None, all_keypoints=None):
        """
        可视化检查杆身是否超过头部平行线
        
        Args:
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            stage_indices: 阶段索引数据
            all_keypoints: 所有帧的关键点数据
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 检查是否有存储的头顶线数据
        if not hasattr(self, 'head_line_data') or self.head_line_data is None:
            # 如果没有，重新计算
            _ = self.check_club_beyond_head_line(keypoints, stage_indices, all_keypoints)
        
        # 获取头顶线位置
        head_top_y = self.head_line_data.get("head_top_y", 0.2)  # 默认值为0.2
        
        # 将标准化坐标转换为像素坐标
        head_line_y = int(head_top_y * height)
        
        # 获取手腕位置
        left_wrist = (adjusted_keypoints[LEFT_WRIST][:2] * [width, height]).astype(int)
        right_wrist = (adjusted_keypoints[RIGHT_WRIST][:2] * [width, height]).astype(int)
        
        # 选择较高的手腕
        higher_wrist = left_wrist if left_wrist[1] < right_wrist[1] else right_wrist
        
        # 缩短线条宽度，使其更美观
        margin = int(width * 0.1)  # 线条左右各缩短10%
        line_start_x = margin
        line_end_x = width - margin
        
        # 绘制头顶参考线（红色）
        cv2.line(frame, (line_start_x, head_line_y), (line_end_x, head_line_y), (0, 0, 255), 3)
        
        # 绘制手腕位置
        cv2.circle(frame, tuple(left_wrist), 8, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_wrist), 8, (255, 0, 0), -1)
        
        # 绘制从较高手腕到头顶线的垂直连线
        cv2.line(frame, (higher_wrist[0], higher_wrist[1]), (higher_wrist[0], head_line_y), (255, 0, 255), 2)
        
        # 根据检查结果设置文本颜色
        result_color = (0, 255, 0) if result else (0, 0, 255)
        
        # 绘制标题和结果信息
        cv2.putText(frame, "Check if club beyond head line", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Result: {'Pass' if result else 'Fail'}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame
    
    def check_head_within_circle(self, keypoints, stage_indices=None, all_keypoints=None):
        """
        检查头部保持在头部圆圈内
        
        使用阶段0头部位置作为参考，创建一个圆圈，判断当前帧头部是否在圆圈内
        
        Args:
            keypoints: 当前帧关键点坐标
            stage_indices: 各阶段帧索引
            all_keypoints: 所有帧的关键点数据
            
        Returns:
            bool: 是否通过检查
        """
        # 如果没有提供阶段索引或所有关键点数据，无法建立参考圆圈
        if stage_indices is None or all_keypoints is None or "0" not in stage_indices:
            return True  # 无法检查则默认通过
            
        # 获取当前帧头部关键点
        nose = keypoints[NOSE][:2]
        right_eye = keypoints[RIGHT_EYE][:2]
        left_eye = keypoints[LEFT_EYE][:2]
        
        # 计算当前帧头部中心点 (眼睛中点)
        eyes_center = (left_eye + right_eye) / 2
        
        # 获取阶段0的第一帧作为参考
        stage0_idx = stage_indices["0"][0]
        stage0_keypoints = all_keypoints[stage0_idx]
        
        # 获取阶段0头部关键点
        stage0_nose = stage0_keypoints[NOSE][:2]
        stage0_right_eye = stage0_keypoints[RIGHT_EYE][:2]
        stage0_left_eye = stage0_keypoints[LEFT_EYE][:2]
        
        # 计算阶段0头部中心点
        stage0_eyes_center = (stage0_left_eye + stage0_right_eye) / 2
        
        # 计算眼睛间距
        eye_distance = np.linalg.norm(left_eye - right_eye)
        stage0_eye_distance = np.linalg.norm(stage0_left_eye - stage0_right_eye)
        
        # 计算鼻子到眼睛中点的距离
        nose_to_eyes = np.linalg.norm(nose - eyes_center)
        stage0_nose_to_eyes = np.linalg.norm(stage0_nose - stage0_eyes_center)
        
        # 使用阶段0的数据计算头部圆圈半径
        # 取眼睛间距的2倍和鼻子到眼睛距离的3倍中的较大值，确保能包围整个头部
        circle_radius = max(stage0_eye_distance * 2.0, stage0_nose_to_eyes * 3.0)
        
        # 计算当前帧头部中心到阶段0头部中心的距离
        center_distance = np.linalg.norm(eyes_center - stage0_eyes_center)
        
        # 计算各关键点到阶段0头部中心的距离
        nose_distance = np.linalg.norm(nose - stage0_eyes_center)
        left_eye_distance = np.linalg.norm(left_eye - stage0_eyes_center)
        right_eye_distance = np.linalg.norm(right_eye - stage0_eyes_center)
        
        # 计算最大偏移点及距离
        max_distance = max(center_distance, nose_distance, left_eye_distance, right_eye_distance)
        
        # 设定阈值 - 如果任意关键点偏移超过圆半径的90%，判定为不通过
        threshold = circle_radius * 0.9
        
        # 记录检查结果与数据用于可视化
        self.head_circle_check_data = {
            "stage0_center": stage0_eyes_center,
            "circle_radius": circle_radius,
            "current_center": eyes_center,
            "max_distance": max_distance,
            "threshold": threshold
        }
        
        # 返回检查结果
        return max_distance <= threshold

    def visualize_head_within_circle(self, keypoints, result, frame=None, save_path=None, 
                                     stage_indices=None, all_keypoints=None):
        """
        可视化头部位置相对于圆圈的检查结果
        
        Args:
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            stage_indices: 阶段索引数据
            all_keypoints: 所有帧的关键点数据
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 如果没有运行过检查函数或没有必要数据，重新计算
        if not hasattr(self, 'head_circle_check_data') or self.head_circle_check_data is None:
            _ = self.check_head_within_circle(keypoints, stage_indices, all_keypoints)
            if not hasattr(self, 'head_circle_check_data') or self.head_circle_check_data is None:
                # 如果仍然没有数据，显示错误信息并返回
                cv2.putText(frame, "Error: Cannot calculate head circle data", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if save_path:
                    self._save_image_safely(frame, save_path)
                return frame
        
        # 获取检查数据
        stage0_center = self.head_circle_check_data["stage0_center"]
        circle_radius = self.head_circle_check_data["circle_radius"]
        current_center = self.head_circle_check_data["current_center"]
        max_distance = self.head_circle_check_data["max_distance"]
        threshold = self.head_circle_check_data["threshold"]
        
        # 转换为像素坐标
        stage0_center_px = (stage0_center * [width, height]).astype(int)
        circle_radius_px = int(circle_radius * width)  # 假设归一化坐标中的x对应图像宽度
        
        # 将当前帧关键点转换为像素坐标
        nose_px = (adjusted_keypoints[NOSE][:2] * [width, height]).astype(int)
        left_eye_px = (adjusted_keypoints[LEFT_EYE][:2] * [width, height]).astype(int)
        right_eye_px = (adjusted_keypoints[RIGHT_EYE][:2] * [width, height]).astype(int)
        current_center_px = ((left_eye_px + right_eye_px) / 2).astype(int)
        
        # 绘制阶段0头部圆圈
        cv2.circle(frame, tuple(stage0_center_px), circle_radius_px, (0, 0, 255), 2)
        
        # 绘制当前帧头部关键点
        cv2.circle(frame, tuple(nose_px), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_eye_px), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_eye_px), 4, (255, 0, 0), -1)
        
        # 绘制从阶段0头部中心到当前头部中心的连线
        cv2.line(frame, tuple(stage0_center_px), tuple(current_center_px), (255, 0, 255), 2)
        
        # 根据检查结果设置文本颜色
        result_color = (0, 255, 0) if result else (0, 0, 255)
        
        # 绘制标题和信息
        cv2.putText(frame, "Check if head is within the circle", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Max distance: {max_distance:.2f} (threshold: {threshold:.2f})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Circle radius: {circle_radius:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Result: {'Pass' if result else 'Fail'}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        
        # 标记阶段0头部中心
        cv2.circle(frame, tuple(stage0_center_px), 6, (0, 255, 255), -1)
        cv2.putText(frame, "Stage 0 Head Center", (stage0_center_px[0] + 10, stage0_center_px[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame
    
    def check_left_hip_near_vertical_line(self, keypoints):
        """检查左髋靠近左脚外侧垂直线不要超过"""
        left_hip = keypoints[LEFT_HIP]
        left_heel = keypoints[LEFT_HEEL]
        
        # 检查左髋是否靠近左脚垂直线，使用相对距离
        relative_distance = self._relative_horizontal_distance(left_hip, left_heel, keypoints)
        return relative_distance < 0.25  # 相对距离不超过肩宽的1/4
    
    def check_arm_triangle(self, keypoints):
        """
        检查保持臂三角角度，避免"鸡翅膀"
        
        检查手臂和肩膀组成的三角形，手腕处的角度应在45-60度之间
        
        Args:
            keypoints: 关键点坐标
        
        Returns:
            bool: 是否通过检查
        """
        # 获取关键点
        left_shoulder = keypoints[LEFT_SHOULDER][:2]
        right_shoulder = keypoints[RIGHT_SHOULDER][:2]
        left_wrist = keypoints[LEFT_WRIST][:2]
        right_wrist = keypoints[RIGHT_WRIST][:2]
        
        # 检查手腕高度差异
        height_diff = abs(left_wrist[1] - right_wrist[1])
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        # 如果高度差异超过肩宽的25%，则使用位置更低的手腕
        if height_diff > 0.25 * shoulder_width:
            # 确定哪个手腕更低（y值更大）
            if left_wrist[1] > right_wrist[1]:
                # 左手更低，用左手坐标代替右手
                right_wrist = left_wrist.copy()
            else:
                # 右手更低，用右手坐标代替左手
                left_wrist = right_wrist.copy()
        
        # 计算向量 - 从手腕到肩膀的向量
        left_arm_vec = left_shoulder - left_wrist
        right_arm_vec = right_shoulder - right_wrist
        
        # 计算夹角 - 手腕处的角度
        dot_product = np.dot(left_arm_vec, right_arm_vec)
        norm_left = np.linalg.norm(left_arm_vec)
        norm_right = np.linalg.norm(right_arm_vec)
        
        # 计算夹角的余弦值
        cos_angle = dot_product / (norm_left * norm_right)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在[-1, 1]范围内
        
        # 转换为角度
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        # 检查角度是否在45-60度范围内
        return 45 <= angle <= 60
    
    def check_arms_straight_crossed(self, keypoints):
        """
        检查手臂伸直且交叉，避免右曲球
        
        检查条件：
        1. 使用肩膀到手腕的直线距离评估手臂伸直度
        2. 手臂应该交叉（左手腕在右手腕的右侧）
        
        Args:
            keypoints: 关键点坐标
            
        Returns:
            bool: 是否通过检查
        """
        # 获取关键点
        left_shoulder = keypoints[LEFT_SHOULDER][:2]
        left_elbow = keypoints[LEFT_ELBOW][:2]
        left_wrist = keypoints[LEFT_WRIST][:2]
        right_shoulder = keypoints[RIGHT_SHOULDER][:2]
        right_elbow = keypoints[RIGHT_ELBOW][:2]
        right_wrist = keypoints[RIGHT_WRIST][:2]
        
        # 计算直线距离
        left_direct_distance = np.linalg.norm(left_shoulder - left_wrist)
        left_path_distance = np.linalg.norm(left_shoulder - left_elbow) + np.linalg.norm(left_elbow - left_wrist)
        
        right_direct_distance = np.linalg.norm(right_shoulder - right_wrist)
        right_path_distance = np.linalg.norm(right_shoulder - right_elbow) + np.linalg.norm(right_elbow - right_wrist)
        
        # 计算直线度（直线距离/路径距离，越接近1越直）
        left_straightness = left_direct_distance / left_path_distance if left_path_distance > 0 else 0
        right_straightness = right_direct_distance / right_path_distance if right_path_distance > 0 else 0
        
        # 判断手臂是否足够直（直线度>0.95相当于约18度的弯曲）
        left_arm_straight = left_straightness > 0.95
        right_arm_straight = right_straightness > 0.95
        
        # 备用：传统角度计算方法，用于可视化
        left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # 存储计算结果用于可视化
        self.arm_check_data = {
            "left_straightness": left_straightness,
            "right_straightness": right_straightness,
            "left_arm_angle": left_arm_angle,
            "right_arm_angle": right_arm_angle
        }
        
        # 检查手臂是否交叉（左手腕在右手腕的右侧）
        arms_crossed = left_wrist[0] > right_wrist[0]
        
        # 判断是否同时满足手臂伸直和交叉条件
        return left_arm_straight and right_arm_straight and arms_crossed
    
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
    
    def check_head_left_of_vertical_line(self, keypoints, stage_indices=None, all_keypoints=None):
        """检查头部保持在左侧垂直线左侧"""
        # 使用右眼而不是鼻子作为参考点
        right_eye = keypoints[RIGHT_EYE]  # 假设有RIGHT_EYE常量，如果没有需要定义
        
        # 如果提供了阶段索引和所有关键点，则使用阶段0的头部右侧位置作为参考
        if stage_indices is not None and all_keypoints is not None and "0" in stage_indices:
            # 获取阶段0的第一帧作为参考
            stage0_idx = stage_indices["0"][0]
            stage0_keypoints = all_keypoints[stage0_idx]
            # 获取阶段0的右眼位置
            stage0_right_eye = stage0_keypoints[RIGHT_EYE]
            # 头部右侧位置（简单地将右眼x坐标右移一些距离，可以根据需要调整）
            head_right_x = stage0_right_eye[0] + 0.03  # 右移3%的宽度，因为眼睛比鼻子更靠右
            
            # 检查当前头部是否在阶段0头部右侧线的左侧
            return right_eye[0] < head_right_x
        else:
            # 如果没有提供阶段0参考，回退到使用左脚作为参考
            left_foot = keypoints[LEFT_FOOT_INDEX]
            return right_eye[0] < left_foot[0]

    def visualize_head_left_of_vertical_line(self, keypoints, result, frame=None, save_path=None, 
                                             stage_indices=None, all_keypoints=None):
        """可视化头部位置相对于垂直线的检查结果"""
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 将标准化坐标转换为像素坐标，使用右眼替代鼻子
        right_eye = (adjusted_keypoints[RIGHT_EYE][:2] * [width, height]).astype(int)
        
        # 获取参考线位置
        if stage_indices is not None and all_keypoints is not None and "0" in stage_indices:
            # 获取阶段0的第一帧作为参考
            stage0_idx = stage_indices["0"][0]
            stage0_keypoints = all_keypoints[stage0_idx]
            # 获取阶段0的右眼位置
            stage0_right_eye = stage0_keypoints[RIGHT_EYE]
            # 头部右侧位置
            head_right_x = int((stage0_right_eye[0] + 0.03) * width)
            
            # 减少垂直线的延长量（上下各延长10%）
            line_height = right_eye[1] * 2  # 估计头部到底部的距离
            extension = int(line_height * 0.05)  # 延长10%
            
            # 计算线的顶部和底部位置，考虑延长
            line_top = max(0, right_eye[1] - line_height // 3 - extension)  # 头部上方1/3处
            line_bottom = min(height, right_eye[1] + line_height // 2 + extension)  # 头部下方1/2处
            
            # 绘制阶段0头部右侧线 - 使用红色加粗线条
            cv2.line(frame, (head_right_x, line_top), (head_right_x, line_bottom), 
                    (0, 0, 255), 3)  # 红色线，粗细为3
            
            # 添加标注
            cv2.putText(frame, "Stage 0 head right", (head_right_x + 5, line_top + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            # 使用左脚作为参考
            left_foot = (adjusted_keypoints[LEFT_FOOT_INDEX][:2] * [width, height]).astype(int)
            head_right_x = left_foot[0]
            
            # 减少垂直线延长
            line_height = height // 3  # 估计适当的线高
            extension = int(line_height * 0.1)  # 延长10%
            
            # 计算线的位置，考虑延长
            line_top = max(0, right_eye[1] - line_height // 3 - extension)
            line_bottom = min(height, right_eye[1] + line_height // 2 + extension)
            
            # 绘制垂直参考线 - 使用红色加粗线条
            cv2.line(frame, (head_right_x, line_top), (head_right_x, line_bottom), 
                    (0, 0, 255), 3)  # 红色线，粗细为3
        
        # 根据检查结果设置文本颜色
        result_color = (0, 255, 0) if result else (0, 0, 255)
        
        # 绘制标题和信息
        cv2.putText(frame, "Check if head is left of vertical line", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Result: {'Pass' if result else 'Fail'}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame
    
    def check_left_body_at_vertical_line(self, keypoints, stage_indices=None, all_keypoints=None):
        """
        检查左侧身体与左侧垂直线重合
        
        使用阶段0的左脚趾位置加上偏移量作为垂直参考线，检查左侧身体是否靠近此线
        
        Args:
            keypoints: 当前帧关键点坐标
            stage_indices: 各阶段帧索引
            all_keypoints: 所有帧的关键点数据
            
        Returns:
            bool: 是否通过检查
        """
        # 获取当前帧关键点
        left_shoulder = keypoints[LEFT_SHOULDER][:2]
        left_hip = keypoints[LEFT_HIP][:2]
        
        # 计算左侧身体中点
        left_body = (left_shoulder + left_hip) / 2
        
        # 获取身体比例尺度
        body_scale = self._get_body_scale(keypoints)
        
        # 如果提供了阶段索引和所有关键点，则使用阶段0的脚位置作为参考
        if stage_indices is not None and all_keypoints is not None and "0" in stage_indices:
            # 获取阶段0的第一帧作为参考
            stage0_idx = stage_indices["0"][0]
            stage0_keypoints = all_keypoints[stage0_idx]
            
            # 获取阶段0的关键点
            stage0_foot_index = stage0_keypoints[LEFT_FOOT_INDEX][:2]
            stage0_hip = stage0_keypoints[LEFT_HIP][:2]
            
            # 确定垂直线位置 - 基于阶段0的脚趾位置
            vertical_line_x = stage0_foot_index[0]
            
            # 比较脚趾与髋部位置
            if stage0_foot_index[0] < stage0_hip[0]:
                # 如果脚趾X坐标比髋部小(脚在髋内侧)，增加额外偏移
                extra_offset = (stage0_hip[0] - stage0_foot_index[0]) * 0.5  # 差距的50%
                vertical_line_x += extra_offset
        else:
            # 如果没有提供阶段0参考，使用当前帧
            left_foot_index = keypoints[LEFT_FOOT_INDEX][:2]
            
            # 确定垂直线位置 - 基于当前帧的脚趾位置
            vertical_line_x = left_foot_index[0]
            
            # 比较脚趾与髋部位置
            if left_foot_index[0] < left_hip[0]:
                # 如果脚趾X坐标比髋部小(脚在髋内侧)，增加额外偏移
                extra_offset = (left_hip[0] - left_foot_index[0]) * 0.5  # 差距的50%
                vertical_line_x += extra_offset
        
        # 添加基础偏移，确保在脚外侧
        base_offset = body_scale * 0.1  # 基础偏移：肩宽的10%
        vertical_line_x += base_offset
        
        # 计算左侧身体中点到垂直线的水平距离
        horizontal_distance = abs(left_body[0] - vertical_line_x)
        
        # 使用相对距离进行判断，阈值为肩宽的10%
        relative_distance = horizontal_distance / body_scale
        return relative_distance < 0.1
    
    def check_left_arm_parallel_ground(self, keypoints):
        """
        检查左大臂平行于地面
        
        判断标准：
        1. 左肩到左肘的连线与水平线的夹角应小于15度
        2. 左肘的Y坐标应接近左肩的Y坐标
        
        Args:
            keypoints: 关键点坐标
            
        Returns:
            bool: 是否通过检查
        """
        left_shoulder = keypoints[LEFT_SHOULDER][:2]
        left_elbow = keypoints[LEFT_ELBOW][:2]
        
        # 计算左大臂向量（从肩膀到肘部）
        arm_vector = left_elbow - left_shoulder
        
        # 计算与水平线的夹角
        if abs(arm_vector[0]) < 1e-6:  # 避免除零错误
            angle = 90.0  # 垂直情况
        else:
            angle = np.arctan2(abs(arm_vector[1]), abs(arm_vector[0])) * 180 / np.pi
        
        # 检查角度是否小于15度（接近水平）
        is_parallel = angle <= 15.0
        
        # 存储检查数据用于可视化
        self.arm_parallel_data = {
            'left_shoulder': left_shoulder,
            'left_elbow': left_elbow,
            'arm_vector': arm_vector,
            'angle': angle,
            'is_parallel': is_parallel
        }
        
        return is_parallel
    
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

    def _relative_distance(self, p1, p2, keypoints):
        """
        计算两点之间的相对距离（相对于肩宽）
        
        Args:
            p1: 第一个点的坐标
            p2: 第二个点的坐标
            keypoints: 关键点坐标
            
        Returns:
            float: 相对距离，单位为肩宽比例
        """
        dist = np.linalg.norm(p1[:2] - p2[:2])
        scale = self._get_body_scale(keypoints)
        if scale > 0:
            return dist / scale
        return 999  # 避免除零错误，返回一个大值

    def _relative_horizontal_distance(self, p1, p2, keypoints):
        """
        计算两点之间的水平相对距离（相对于肩宽）
        
        Args:
            p1: 第一个点的坐标
            p2: 第二个点的坐标
            keypoints: 关键点坐标
            
        Returns:
            float: 水平相对距离，单位为肩宽比例
        """
        dist = abs(p1[0] - p2[0])
        scale = self._get_body_scale(keypoints)
        if scale > 0:
            return dist / scale
        return 999  # 避免除零错误，返回一个大值

    def _save_image_safely(self, frame, save_path):
        """
        安全地保存图像，提供多种回退策略
        
        Args:
            frame: 要保存的图像
            save_path: 首选保存路径
            
        Returns:
            bool: 是否成功保存
        """
        import cv2
        import tempfile
        import shutil
        
        # 确保路径无中文字符
        try:
            # 尝试对路径进行ASCII编码，如果成功则无中文字符
            save_path.encode('ascii')
        except UnicodeEncodeError:
            # 路径中有非ASCII字符，需要修改路径
            dir_name = os.path.dirname(save_path)
            orig_filename = os.path.basename(save_path)
            # 创建只有ASCII字符的文件名
            safe_filename = ''.join(c for c in orig_filename if ord(c) < 128 and c.isalnum() or c in ['_', '.'])
            save_path = os.path.join(dir_name, safe_filename)
            print(f"修改文件路径为无中文字符: {save_path}")
        
        # 尝试直接保存
        try:
            # 确保目录存在
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"创建目录: {save_dir}")
            
            # 保存图像
            result = cv2.imwrite(save_path, frame)
            
            # 检查保存是否成功
            if result and os.path.exists(save_path):
                print(f"已成功保存可视化图像到: {save_path}")
                print(f"文件大小: {os.path.getsize(save_path)} 字节")
                return True
            else:
                print(f"警告: 图像保存失败: {save_path}")
        except Exception as e:
            print(f"保存图像时出错: {str(e)}")
        
        # 如果直接保存失败，尝试使用临时文件
        try:
            print("尝试使用临时文件保存...")
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
                temp_path = tmpfile.name
                
            # 保存到临时文件
            result = cv2.imwrite(temp_path, frame)
            
            if result and os.path.exists(temp_path):
                # 创建简化的最终路径
                dirname = os.path.dirname(save_path)
                # 生成简单临时文件名
                temp_filename = f"img_{os.path.getmtime(temp_path):.0f}.jpg"
                simple_path = os.path.join(dirname, temp_filename)
                
                # 尝试复制到最终位置
                try:
                    shutil.copy2(temp_path, simple_path)
                    print(f"已通过临时文件保存图像到: {simple_path}")
                    os.unlink(temp_path)  # 删除临时文件
                    return True
                except Exception as e:
                    print(f"复制临时文件失败: {str(e)}")
                    
                    # 如果复制失败，至少保留临时文件
                    print(f"保留临时文件: {temp_path}")
                    return True
            else:
                print(f"临时文件保存失败")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            print(f"使用临时文件保存时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return False
        
    def visualize_check_result(self, check_function_name, keypoints, result, frame=None, save_path=None, stage_name=None, stage_indices=None, all_keypoints=None):
        """
        通用的检查函数可视化框架，根据检查函数名称调用相应的可视化函数
        
        Args:
            check_function_name: 检查函数名称
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            stage_name: 可选的阶段名称，用于标注
            
        Returns:
            numpy.ndarray: 绘制了辅助线的图像，如果没有对应的可视化函数则返回None
        """
        # 确保保存路径无中文字符
        if save_path:
            try:
                save_path.encode('ascii')
            except UnicodeEncodeError:
                # 处理含有中文的路径
                dir_name = os.path.dirname(save_path)
                # 生成安全的文件名 - 使用时间戳确保唯一性
                import time
                safe_filename = f"vis_{int(time.time())}_{hash(check_function_name) % 10000}.jpg"
                save_path = os.path.join(dir_name, safe_filename)
                print(f"修改为安全文件路径: {save_path}")
        
        # 如果check_function_name为None但提供了stage_name，生成通用阶段可视化
        if check_function_name is None and stage_name is not None:
            # 创建基本的阶段关键点可视化
            return self._visualize_stage_keypoints(keypoints, stage_name, frame, save_path)
        
        # 准备姿势阶段的所有检查函数使用共用的可视化函数
        if check_function_name in ["check_shoulder_height", "check_left_arm_line", "check_feet_width", "check_hand_grip_parallel"]:
            return self.visualize_setup_position(keypoints, result, frame, save_path, stage_indices, all_keypoints)
            
        # 构建可视化函数名称
        if check_function_name:
            vis_func_name = f"visualize_{check_function_name[6:]}"  # 去掉"check_"前缀，加上"visualize_"前缀
            
            # 检查是否存在对应的可视化函数
            if hasattr(self, vis_func_name):
                vis_func = getattr(self, vis_func_name)
                return vis_func(keypoints, result, frame, save_path, stage_indices, all_keypoints)
            else:
                # 如果没有专门的可视化函数，使用通用可视化
                return self._visualize_stage_keypoints(keypoints, f"{check_function_name}: {'Pass' if result else 'Fail'}", frame, save_path)
        
        # 如果没有专门的可视化函数，返回None
        return None

    def _visualize_stage_keypoints(self, keypoints, stage_name, frame=None, save_path=None):
        """
        创建阶段关键点的基本可视化
        
        Args:
            keypoints: 关键点坐标
            stage_name: 阶段名称
            frame: 可选的帧图像
            save_path: 保存路径
            
        Returns:
            numpy.ndarray: 绘制了关键点和基本辅助线的图像
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像，尺寸根据关键点范围确定
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标以适应新图像
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            # 如果提供了帧图像，直接使用原始关键点
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 将标准化坐标转换为像素坐标
        nose = (adjusted_keypoints[NOSE][:2] * [width, height]).astype(int)
        left_shoulder = (adjusted_keypoints[LEFT_SHOULDER][:2] * [width, height]).astype(int)
        right_shoulder = (adjusted_keypoints[RIGHT_SHOULDER][:2] * [width, height]).astype(int)
        left_elbow = (adjusted_keypoints[LEFT_ELBOW][:2] * [width, height]).astype(int)
        right_elbow = (adjusted_keypoints[RIGHT_ELBOW][:2] * [width, height]).astype(int)
        left_wrist = (adjusted_keypoints[LEFT_WRIST][:2] * [width, height]).astype(int)
        right_wrist = (adjusted_keypoints[RIGHT_WRIST][:2] * [width, height]).astype(int)
        left_hip = (adjusted_keypoints[LEFT_HIP][:2] * [width, height]).astype(int)
        right_hip = (adjusted_keypoints[RIGHT_HIP][:2] * [width, height]).astype(int)
        left_knee = (adjusted_keypoints[LEFT_KNEE][:2] * [width, height]).astype(int)
        right_knee = (adjusted_keypoints[RIGHT_KNEE][:2] * [width, height]).astype(int)
        left_ankle = (adjusted_keypoints[LEFT_ANKLE][:2] * [width, height]).astype(int)
        right_ankle = (adjusted_keypoints[RIGHT_ANKLE][:2] * [width, height]).astype(int)
        left_foot_index = (adjusted_keypoints[LEFT_FOOT_INDEX][:2] * [width, height]).astype(int)
        right_foot_index = (adjusted_keypoints[RIGHT_FOOT_INDEX][:2] * [width, height]).astype(int)
        
        # 绘制躯干
        cv2.line(frame, tuple(nose), tuple((left_shoulder + right_shoulder) // 2), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_shoulder), tuple(right_shoulder), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_shoulder), tuple(left_elbow), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_shoulder), tuple(right_elbow), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_elbow), tuple(left_wrist), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_elbow), tuple(right_wrist), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_shoulder), tuple(left_hip), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_shoulder), tuple(right_hip), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_hip), tuple(right_hip), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_hip), tuple(left_knee), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_hip), tuple(right_knee), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_knee), tuple(left_ankle), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_knee), tuple(right_ankle), (0, 0, 255), 2)
        
        # 绘制关键点
        cv2.circle(frame, tuple(nose), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_shoulder), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_shoulder), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_elbow), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_elbow), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_wrist), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_wrist), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_hip), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_hip), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(left_knee), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_knee), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(left_ankle), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_ankle), 4, (0, 255, 0), -1)
        
        # 绘制垂直参考线
        left_foot_x = left_foot_index[0]
        right_foot_x = right_foot_index[0]
        
        # 使用虚线绘制垂直参考线
        self._draw_dashed_line(frame, (int(left_foot_x), 0), (int(left_foot_x), height), (0, 0, 255), 2)
        self._draw_dashed_line(frame, (int(right_foot_x), 0), (int(right_foot_x), height), (0, 0, 255), 2)
        
        # 绘制标题和阶段信息
        cv2.putText(frame, f"阶段: {stage_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"肩宽: {self._get_body_scale(keypoints):.2f}像素", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=10, gap_length=10):
        """
        绘制虚线，替代cv2.LINE_DASH
        
        Args:
            img: 要绘制的图像
            pt1: 起点坐标(x1, y1)
            pt2: 终点坐标(x2, y2)
            color: 线条颜色
            thickness: 线条粗细
            dash_length: 虚线段长度
            gap_length: 虚线间隔长度
        """
        import numpy as np
        
        dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
        if dist == 0:
            return
            
        # 计算方向向量
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist
        
        # 画虚线
        step = dash_length + gap_length
        curr_pos = 0
        while curr_pos < dist:
            start = (int(pt1[0] + dx * curr_pos), int(pt1[1] + dy * curr_pos))
            end_pos = min(curr_pos + dash_length, dist)
            end = (int(pt1[0] + dx * end_pos), int(pt1[1] + dy * end_pos))
            
            import cv2
            cv2.line(img, start, end, color, thickness)
            curr_pos += step

    def visualize_left_body_at_vertical_line(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        """
        可视化左侧身体与左侧垂直线重合的检查结果
        
        Args:
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            stage_indices: 阶段索引数据
            all_keypoints: 所有帧的关键点数据
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像，尺寸根据关键点范围确定
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 将标准化坐标转换为像素坐标并转换为整数
        left_shoulder = (adjusted_keypoints[LEFT_SHOULDER][:2] * [width, height]).astype(int)
        right_shoulder = (adjusted_keypoints[RIGHT_SHOULDER][:2] * [width, height]).astype(int)
        left_hip = (adjusted_keypoints[LEFT_HIP][:2] * [width, height]).astype(int)
        right_hip = (adjusted_keypoints[RIGHT_HIP][:2] * [width, height]).astype(int)
        left_knee = (adjusted_keypoints[LEFT_KNEE][:2] * [width, height]).astype(int)
        left_ankle = (adjusted_keypoints[LEFT_ANKLE][:2] * [width, height]).astype(int)
        left_foot_index = (adjusted_keypoints[LEFT_FOOT_INDEX][:2] * [width, height]).astype(int)
        
        # 计算左侧身体中点
        left_body = ((left_shoulder + left_hip) / 2).astype(int)
        
        # 使用脚趾的X坐标来定义垂直线基准
        body_scale = self._get_body_scale(keypoints)
        
        # 如果提供了阶段索引和所有关键点，则使用阶段0的脚位置作为参考
        if stage_indices is not None and all_keypoints is not None and "0" in stage_indices:
            # 获取阶段0的第一帧作为参考
            stage0_idx = stage_indices["0"][0]
            stage0_keypoints = all_keypoints[stage0_idx]
            
            # 将阶段0关键点坐标转换为像素坐标
            stage0_foot_index = (stage0_keypoints[LEFT_FOOT_INDEX][:2] * [width, height]).astype(int)
            stage0_hip = (stage0_keypoints[LEFT_HIP][:2] * [width, height]).astype(int)
            
            # 确定垂直线位置 - 基于阶段0的脚趾位置
            vertical_line_x = stage0_foot_index[0]
            
            # 比较脚趾与髋部位置
            if stage0_foot_index[0] < stage0_hip[0]:
                # 如果脚趾X坐标比髋部小(脚在髋内侧)，增加额外偏移
                extra_offset = (stage0_hip[0] - stage0_foot_index[0]) * 0.5  # 差距的50%
                vertical_line_x += int(extra_offset)
                
            # 添加标注，说明使用阶段0作为参考
            reference_info = "Using Stage 0 as reference"
        else:
            # 如果没有提供阶段0参考，使用当前帧
            # 确定垂直线位置 - 基于当前帧的脚趾位置
            vertical_line_x = left_foot_index[0]
            
            # 比较脚趾与髋部位置
            if left_foot_index[0] < left_hip[0]:
                # 如果脚趾X坐标比髋部小(脚在髋内侧)，增加额外偏移
                extra_offset = (left_hip[0] - left_foot_index[0]) * 0.5  # 差距的50%
                vertical_line_x += int(extra_offset)
                
            # 添加标注，说明使用当前帧作为参考
            reference_info = "Using current frame as reference"
        
        # 添加基础偏移，确保在脚外侧
        base_offset = int(body_scale * 0.2 * width)  # 基础偏移：肩宽的10%
        vertical_line_x += base_offset
        
        # 定义垂直线的起止点（不再贯穿整个图像）
        line_top = min(left_shoulder[1], right_shoulder[1]) - 30  # 肩膀以上30像素
        line_bottom = max(left_ankle[1], left_foot_index[1]) + 20  # 脚部以下20像素
        
        # 计算左侧身体中点到垂直线的水平距离
        horizontal_distance = abs(left_body[0] - vertical_line_x)
        
        # 使用相对距离进行判断
        relative_distance = horizontal_distance / (body_scale * width)
        
        # 绘制骨架
        # 绘制躯干
        cv2.line(frame, tuple(left_shoulder), tuple(right_shoulder), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_shoulder), tuple(left_hip), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_shoulder), tuple(right_hip), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_hip), tuple(right_hip), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_hip), tuple(left_knee), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_knee), tuple(left_ankle), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_ankle), tuple(left_foot_index), (0, 0, 255), 2)
        
        # 绘制关键点
        cv2.circle(frame, tuple(left_shoulder), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_shoulder), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_hip), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_hip), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(left_knee), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(left_ankle), 4, (0, 255, 0), -1)
        cv2.circle(frame, tuple(left_foot_index), 4, (0, 255, 0), -1)
        
        # 绘制左侧身体中点
        cv2.circle(frame, tuple(left_body), 6, (0, 0, 255), -1)
        
        # 绘制左脚垂直线（只在指定范围内显示）
        self._draw_dashed_line(frame, 
                               (int(vertical_line_x), line_top), 
                               (int(vertical_line_x), line_bottom), 
                               (0, 255, 255), 2)
        
        # 绘制左侧身体中点到垂直线的水平连线
        cv2.line(frame, tuple(left_body), (int(vertical_line_x), left_body[1]), (255, 0, 255), 2)
        
        # 计算允许的最大距离范围 - 更严格的标准(1/10肩宽)
        max_allowed_distance = 0.1 * body_scale * width
        
        # 绘制允许的范围区域（透明绿色区域）
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                      (int(vertical_line_x - max_allowed_distance), line_top), 
                      (int(vertical_line_x + max_allowed_distance), line_bottom), 
                      (0, 255, 0), -1)  # 实心填充
        # 添加透明度
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制允许范围边界线
        self._draw_dashed_line(frame, 
                (int(vertical_line_x - max_allowed_distance), line_top),
                (int(vertical_line_x - max_allowed_distance), line_bottom),
                (0, 200, 0), 1)
        self._draw_dashed_line(frame, 
                (int(vertical_line_x + max_allowed_distance), line_top),
                (int(vertical_line_x + max_allowed_distance), line_bottom),
                (0, 200, 0), 1)
        
        # 根据检查结果设置文本颜色
        result_color = (0, 255, 0) if result else (0, 0, 255)
        
        # 绘制标题和距离信息 - 使用英文
        cv2.putText(frame, "Check if left body aligns with left vertical line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, reference_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Relative distance: {relative_distance:.2f} (threshold: 0.1)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Absolute distance: {horizontal_distance:.2f} pixels", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Result: {'Pass' if result else 'Fail'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame

    def visualize_arms_straight_crossed(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        """
        可视化手臂伸直和交叉的检查结果
        
        Args:
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            
        Returns:
            numpy.ndarray: 绘制了辅助线的图像
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像，尺寸根据关键点范围确定
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标以适应新图像
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            # 如果提供了帧图像，直接使用原始关键点
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 将标准化坐标转换为像素坐标并转换为整数
        left_shoulder = (adjusted_keypoints[LEFT_SHOULDER][:2] * [width, height]).astype(int)
        left_elbow = (adjusted_keypoints[LEFT_ELBOW][:2] * [width, height]).astype(int)
        left_wrist = (adjusted_keypoints[LEFT_WRIST][:2] * [width, height]).astype(int)
        right_shoulder = (adjusted_keypoints[RIGHT_SHOULDER][:2] * [width, height]).astype(int)
        right_elbow = (adjusted_keypoints[RIGHT_ELBOW][:2] * [width, height]).astype(int)
        right_wrist = (adjusted_keypoints[RIGHT_WRIST][:2] * [width, height]).astype(int)
        
        # 获取计算结果数据，如果有的话
        if hasattr(self, 'arm_check_data') and self.arm_check_data is not None:
            left_straightness = self.arm_check_data.get("left_straightness", 0)
            right_straightness = self.arm_check_data.get("right_straightness", 0)
            left_arm_angle = self.arm_check_data.get("left_arm_angle", 0)
            right_arm_angle = self.arm_check_data.get("right_arm_angle", 0)
        else:
            # 计算直线度
            left_direct_distance = np.linalg.norm(left_shoulder - left_wrist)
            left_path_distance = np.linalg.norm(left_shoulder - left_elbow) + np.linalg.norm(left_elbow - left_wrist)
            right_direct_distance = np.linalg.norm(right_shoulder - right_wrist)
            right_path_distance = np.linalg.norm(right_shoulder - right_elbow) + np.linalg.norm(right_elbow - right_wrist)
            
            left_straightness = left_direct_distance / left_path_distance if left_path_distance > 0 else 0
            right_straightness = right_direct_distance / right_path_distance if right_path_distance > 0 else 0
            
            # 计算角度用于显示
            left_arm_angle = self._calculate_angle(
                (adjusted_keypoints[LEFT_SHOULDER][:2] * [width, height]), 
                (adjusted_keypoints[LEFT_ELBOW][:2] * [width, height]), 
                (adjusted_keypoints[LEFT_WRIST][:2] * [width, height])
            )
            right_arm_angle = self._calculate_angle(
                (adjusted_keypoints[RIGHT_SHOULDER][:2] * [width, height]), 
                (adjusted_keypoints[RIGHT_ELBOW][:2] * [width, height]), 
                (adjusted_keypoints[RIGHT_WRIST][:2] * [width, height])
            )
        
        # 绘制手臂骨架
        # 左臂
        cv2.line(frame, tuple(left_shoulder), tuple(left_elbow), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_elbow), tuple(left_wrist), (0, 0, 255), 2)
        
        # 右臂
        cv2.line(frame, tuple(right_shoulder), tuple(right_elbow), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_elbow), tuple(right_wrist), (0, 0, 255), 2)
        
        # 绘制关键点 - 增大关键点尺寸
        cv2.circle(frame, tuple(left_shoulder), 6, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_elbow), 8, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_wrist), 8, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_shoulder), 6, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_elbow), 8, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_wrist), 8, (255, 0, 0), -1)
        
        # 检查手臂是否交叉
        arms_crossed = left_wrist[0] > right_wrist[0]
        
        # 绘制手臂交叉状态指示器
        cross_status_color = (0, 255, 0) if arms_crossed else (0, 0, 255)
        
        # 绘制手腕连线来显示交叉状态
        cv2.line(frame, tuple(left_wrist), tuple(right_wrist), cross_status_color, 2)
        
        # 只有在手臂不够直时才显示角度标注
        if left_straightness < 0.95:
            # 左臂不够直，显示红色标注
            angle_text_pos = ((left_shoulder + left_elbow) // 2).astype(int)
            cv2.putText(frame, f"Straightness: {left_straightness:.2f}", 
                      (angle_text_pos[0], angle_text_pos[1] - 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if right_straightness < 0.95:
            # 右臂不够直，显示红色标注
            angle_text_pos = ((right_shoulder + right_elbow) // 2).astype(int)
            cv2.putText(frame, f"Straightness: {right_straightness:.2f}", 
                      (angle_text_pos[0], angle_text_pos[1] - 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制检查结果
        result_color = (0, 255, 0) if result else (0, 0, 255)
        
        # 绘制标题和信息
        cv2.putText(frame, "Check Arms Straight and Crossed Status", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 显示手臂直线度信息
        cv2.putText(frame, f"Left Arm Straightness: {left_straightness:.2f} (>0.95 pass)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, f"Right Arm Straightness: {right_straightness:.2f} (>0.95 pass)", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 显示交叉状态
        cross_text = "Arms Crossed: Yes" if arms_crossed else "Arms Crossed: No"
        cv2.putText(frame, cross_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cross_status_color, 2)
        
        # 显示总体检查结果
        cv2.putText(frame, f"Check Result: {'Pass' if result else 'Fail'}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame

    def visualize_arm_triangle(self, keypoints, result, frame=None, save_path=None, stage_indices=None, all_keypoints=None):
        """
        可视化臂三角检查结果
        
        Args:
            keypoints: 关键点坐标
            result: 检查结果(True/False)
            frame: 可选的帧图像
            save_path: 保存路径
            stage_indices: 阶段索引数据
            all_keypoints: 所有帧的关键点数据
        """
        import cv2
        import numpy as np
        
        # 创建图像
        if frame is None:
            # 创建空白图像，尺寸根据关键点范围确定
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            width = int(max_x - min_x + 100)
            height = int(max_y - min_y + 100)
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            # 调整关键点坐标以适应新图像
            offset_x, offset_y = int(50 - min_x), int(50 - min_y)
            adjusted_keypoints = keypoints.copy()
            adjusted_keypoints[:, 0] += offset_x
            adjusted_keypoints[:, 1] += offset_y
        else:
            # 如果提供了帧图像，直接使用原始关键点
            adjusted_keypoints = keypoints
            height, width = frame.shape[:2]
        
        # 首先使用原始关键点获取坐标
        left_shoulder = keypoints[LEFT_SHOULDER][:2].copy()
        right_shoulder = keypoints[RIGHT_SHOULDER][:2].copy()
        left_wrist = keypoints[LEFT_WRIST][:2].copy()
        right_wrist = keypoints[RIGHT_WRIST][:2].copy()
        
        # 用于可视化的原始关键点（不被修改）
        orig_left_wrist = left_wrist.copy()
        orig_right_wrist = right_wrist.copy()
        
        # 检查手腕高度差异
        height_diff = abs(left_wrist[1] - right_wrist[1])
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        use_adjusted = False
        lower_wrist = None
        
        # 如果高度差异超过肩宽的25%，则使用位置更低的手腕
        if height_diff > 0.25 * shoulder_width:
            use_adjusted = True
            # 确定哪个手腕更低（y值更大）
            if left_wrist[1] > right_wrist[1]:
                # 左手更低，用左手坐标代替右手
                lower_wrist = "left"
                right_wrist = left_wrist.copy()
            else:
                # 右手更低，用右手坐标代替左手
                lower_wrist = "right"
                left_wrist = right_wrist.copy()
        
        # 计算向量 - 从手腕到肩膀的向量
        left_arm_vec = left_shoulder - left_wrist
        right_arm_vec = right_shoulder - right_wrist
        
        # 计算夹角
        dot_product = np.dot(left_arm_vec, right_arm_vec)
        norm_left = np.linalg.norm(left_arm_vec)
        norm_right = np.linalg.norm(right_arm_vec)
        
        cos_angle = dot_product / (norm_left * norm_right)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        # 转换为像素坐标仅用于绘图 - 使用原始坐标绘制骨架
        left_shoulder_px = (adjusted_keypoints[LEFT_SHOULDER][:2] * [width, height]).astype(int)
        right_shoulder_px = (adjusted_keypoints[RIGHT_SHOULDER][:2] * [width, height]).astype(int)
        left_wrist_px = (adjusted_keypoints[LEFT_WRIST][:2] * [width, height]).astype(int)
        right_wrist_px = (adjusted_keypoints[RIGHT_WRIST][:2] * [width, height]).astype(int)
        
        # 调整后的手腕坐标（用于计算角度的坐标）
        adjusted_left_wrist_px = (left_wrist * [width, height]).astype(int)
        adjusted_right_wrist_px = (right_wrist * [width, height]).astype(int)
        
        # 绘制原始手臂三角形 - 以粗红线绘制实际手臂位置
        cv2.line(frame, tuple(left_shoulder_px), tuple(left_wrist_px), (0, 0, 255), 2)
        cv2.line(frame, tuple(right_shoulder_px), tuple(right_wrist_px), (0, 0, 255), 2)
        cv2.line(frame, tuple(left_wrist_px), tuple(right_wrist_px), (0, 0, 255), 2)  # 连接两个手腕
        
        # 如果使用了调整后的坐标，绘制调整后的三角形（用虚线表示）
        if use_adjusted:
            # 使用不同的颜色绘制调整后的线
            self._draw_dashed_line(frame, tuple(left_shoulder_px), tuple(adjusted_left_wrist_px), (255, 0, 255), 1)
            self._draw_dashed_line(frame, tuple(right_shoulder_px), tuple(adjusted_right_wrist_px), (255, 0, 255), 1)
            self._draw_dashed_line(frame, tuple(adjusted_left_wrist_px), tuple(adjusted_right_wrist_px), (255, 0, 255), 1)
        
        # 绘制关键点
        cv2.circle(frame, tuple(left_shoulder_px), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_shoulder_px), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(left_wrist_px), 4, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_wrist_px), 4, (255, 0, 0), -1)
        
        # 计算手腕中点（角度的顶点）
        if use_adjusted:
            wrist_center = tuple(adjusted_left_wrist_px)  # 使用左手腕作为角度中心
        else:
            # 用两个手腕的中点作为角度标注位置
            wrist_center = tuple(((adjusted_left_wrist_px + adjusted_right_wrist_px) // 2).astype(int))
        
        # 根据角度确定颜色
        angle_color = (0, 255, 0) if 45 <= angle <= 60 else (0, 0, 255)
        
        # 添加参考线以表示45-60度范围
        # 使用虚线标识合格的角度范围
        if not (45 <= angle <= 60):
            # 使用左臂为基准，绘制45度和60度的参考线
            
            # 首先计算左臂和右臂的向量（以手腕为起点）
            left_arm_vec_px = np.array(left_shoulder_px) - np.array(left_wrist_px)
            right_arm_vec_px = np.array(right_shoulder_px) - np.array(right_wrist_px)
            
            # 归一化向量长度
            left_arm_len = np.linalg.norm(left_arm_vec_px)
            left_arm_unit = left_arm_vec_px / left_arm_len if left_arm_len > 0 else np.array([0, -1])
            
            right_arm_len = np.linalg.norm(right_arm_vec_px)
            right_arm_unit = right_arm_vec_px / right_arm_len if right_arm_len > 0 else np.array([0, -1])
            
            # 为了可视化效果，延长手臂线条
            extend_length = max(left_arm_len, right_arm_len) * 1.2
            
        
        # 根据检查结果设置文本颜色
        result_color = (0, 255, 0) if result else (0, 0, 255)
        
        # 绘制标题和信息
        cv2.putText(frame, "Check Arm Triangle Angle", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f"Angle: {angle:.1f}° (ideal: 45-60°)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 如果使用了调整后的坐标，添加相关信息
        if use_adjusted:
            cv2.putText(frame, f"Height diff: {height_diff:.2f} (>{0.25*shoulder_width:.2f})", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"Using {lower_wrist} hand for both", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"Result: {'Pass' if result else 'Fail'}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        else:
            cv2.putText(frame, f"Result: {'Pass' if result else 'Fail'}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        
        # 在手臂三角形中央显示角度值
        cv2.putText(frame, f"{angle:.1f}°", wrist_center, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, angle_color, 2)
        
        # 保存图像
        if save_path:
            self._save_image_safely(frame, save_path)
        
        return frame

    def check_left_arm_parallel_ground(self, keypoints):
        """检查左大臂平行于地面"""
        left_shoulder = keypoints[LEFT_SHOULDER]
        left_elbow = keypoints[LEFT_ELBOW]
        left_wrist = keypoints[LEFT_WRIST]
        
        # 检查左大臂与地面是否平行
        result = abs(left_elbow[1] - left_wrist[1]) < 0.1
        
        # 存储检查数据用于可视化
        self.side_check_data = getattr(self, 'side_check_data', {})
        self.side_check_data['left_arm_parallel_ground'] = {
            'left_shoulder': left_shoulder,
            'left_elbow': left_elbow,
            'left_wrist': left_wrist,
            'result': result
        }
        
        return result


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
