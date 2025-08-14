# video_processor.py
# ---------------------------------------------------
import csv
import os
import math
import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple

from config import (
    STAGE_MAP,
    MEDIAPIPE_POSE_CONNECTIONS,
    BODY_POINT_NAMES,
    MARGIN_CONFIG
)
from keypoint_processor import KeypointProcessor
from swing_analyzer import SwingAnalyzer


def _try_pose_landmarker_v2(video_path: str,
                            task_model_path: str,
                            visibility_thresh: float = 0.5,
                            ema_alpha: float = 0.6) -> torch.Tensor:
    """
    使用 MediaPipe Pose Landmarker v2 (heavy) 提取关键点；若失败由上层回退。

    说明：
    - 需要本地存在 `.task` 模型文件（建议放在 `model/pose_landmarker_heavy.task`）
    - 输出保持 (N,33,2)，与现有管线完全兼容
    - 对 landmarks 做可见度过滤与 EMA 平滑，显著提升稳定性
    """
    # 延迟导入，避免环境缺少 tasks 时模块导入失败
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    # 打开视频并读取帧率（供 VIDEO 模式计时戳使用）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ms_per_frame = int(1000.0 / max(fps, 1e-6))

    # 构建 Landmarker 选项（heavy 模型）
    base_opts = BaseOptions(model_asset_path=task_model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    keypoints_list = []
    ema_prev = None  # EMA 平滑的前一帧
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> SRGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO 模式需要单调递增的时间戳(ms)
        result = landmarker.detect_for_video(mp_image, frame_index * ms_per_frame)
        frame_index += 1

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            # 只取第一个人体（num_poses=1）
            lms = result.pose_landmarks[0]
            kp = []
            for lm in lms:
                # 低可见度置零，减少误检噪声
                if hasattr(lm, 'visibility') and lm.visibility is not None and lm.visibility < visibility_thresh:
                    kp.append([0.0, 0.0])
                else:
                    kp.append([lm.x, lm.y])
            curr = torch.tensor(kp, dtype=torch.float32)
        else:
            curr = torch.zeros((33, 2), dtype=torch.float32)

        # 应用 EMA 平滑；在关键点缺失较多时仍能稳定过渡
        if ema_prev is None:
            smoothed = curr
        else:
            smoothed = ema_alpha * curr + (1.0 - ema_alpha) * ema_prev
        ema_prev = smoothed

        keypoints_list.append(smoothed)

    cap.release()
    landmarker.close()

    if not keypoints_list:
        return torch.zeros((0, 33, 2), dtype=torch.float32)
    return torch.stack(keypoints_list)

def extract_keypoints_from_video(video_path: str) -> torch.Tensor:
    """
    姿态关键点提取统一入口（优先 Landmarker v2 heavy，失败则回退到经典 Pose）。

    输出：torch.Tensor (N,33,2)，与既有管线完全兼容。
    """
    # 优先尝试使用 Landmarker v2 heavy（更准确稳定）。
    # 默认模型路径：`./model/pose_landmarker_heavy.task`
    task_path_env = os.environ.get("POSE_LANDMARKER_TASK", "")
    candidate_paths = [
        task_path_env.strip(),
        os.path.join("model", "pose_landmarker_heavy.task")
    ]
    task_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    if task_path is not None:
        try:
            print(f"[INFO] 使用 Pose Landmarker v2 heavy: {task_path}")
            return _try_pose_landmarker_v2(
                video_path=video_path,
                task_model_path=task_path,
                visibility_thresh=0.5,
                ema_alpha=0.6,
            )
        except Exception as lm_err:
            print(f"[WARN] Pose Landmarker v2 失败，回退至经典 MediaPipe Pose: {lm_err}")

    # 回退：使用经典 MediaPipe Pose（保持与旧版本完全一致但参数更保守、更稳健）
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    ema_prev = None  # 与 v2 保持一致的 EMA 平滑
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            kp = []
            for landmark in landmarks:
                if hasattr(landmark, 'visibility') and landmark.visibility is not None and landmark.visibility < 0.5:
                    kp.append([0.0, 0.0])
                else:
                    kp.append([landmark.x, landmark.y])
            curr = torch.tensor(kp, dtype=torch.float32)
        else:
            curr = torch.zeros((33, 2), dtype=torch.float32)

        if ema_prev is None:
            smoothed = curr
        else:
            smoothed = 0.6 * curr + 0.4 * ema_prev
        ema_prev = smoothed

        keypoints_list.append(smoothed)
    cap.release()
    pose.close()
    return torch.stack(keypoints_list)


def compute_stage_intervals(stage_indices: Dict[str, List[int]], frame_count: int) -> List[Tuple[int, int, str]]:
    """
    构造阶段区间列表 intervals=[(start,end,stageName),...]
    """
    intervals: List[Tuple[int,int,str]] = []
    stage_min_list = []
    for stg_name, frames in stage_indices.items():
        if frames:
            mn = min(frames)
            mx = max(frames)
            stage_min_list.append((mn, mx, stg_name))
    if not stage_min_list:
        intervals.append((0, frame_count-1, "setup"))
        return intervals

    stage_min_list.sort(key=lambda x: x[0])
    for i in range(len(stage_min_list)):
        cur_min, cur_max, stg_name = stage_min_list[i]
        next_min = frame_count
        if i+1 < len(stage_min_list):
            next_min = stage_min_list[i+1][0]

        if stg_name in MARGIN_CONFIG:
            margin = MARGIN_CONFIG[stg_name]
            start_ = max(0, cur_min - margin)
            end_   = min(frame_count-1, cur_max + margin)
            if i - 1 >= 0:
                last_max = intervals[-1][1]
                start_ = min(cur_min, last_max)
            intervals.append((start_, end_, stg_name))
        else:
            if i - 1 >= 0:
                cur_min = intervals[-1][1]
            else:
                cur_min = 0
                last_max = stage_min_list[0][1]
                next_min = min(last_max, next_min)
            intervals.append((cur_min, next_min-1, stg_name))
    last_start, last_end, last_stg = intervals[-1]
    if last_end < frame_count-1:
        intervals[-1] = (last_start, frame_count-1, last_stg)
    return intervals


def fill_frames_with_intervals(intervals: List[Tuple[int,int,str]], frame_count: int) -> List[str]:
    """
    后区间覆盖先区间
    """
    result = ["setup"] * frame_count
    for (st, ed, stName) in intervals:
        if st > ed or st >= frame_count:
            continue
        for f in range(st, min(ed+1, frame_count)):
            result[f] = stName
    return result





def process_new_video(video_path: str,
                      stage_indices: Dict[str, List[int]],
                      output_folder: str,
                      keypoint_data: torch.Tensor,
                      analyzer: SwingAnalyzer,
                      csv_filename: str = "allData.csv",
                      img_folder: str = None,
                      rotation_type: str = "none") -> Tuple[List[np.ndarray], List[int]]:
    """
    处理视频并生成CSV以及错误检测所需数据，不再处理图像
    """
    # 1. 准备输出目录和文件路径
    os.makedirs(output_folder, exist_ok=True)
    csv_dir = os.path.join(output_folder, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, csv_filename)

    # 2. 打开视频，获取基本信息
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 计算阶段区间和每帧的阶段归属
    intervals = compute_stage_intervals(stage_indices, frame_count)
    stage_for_frame = fill_frames_with_intervals(intervals, frame_count)

    # 收集关键帧集合
    key_frames_set = set()
    for stg_name, flist in stage_indices.items():
        for fidx in flist:
            key_frames_set.add(fidx)

    # 4. 准备CSV列名
    # 位置列
    pos_cols = []
    for bp in BODY_POINT_NAMES:
        pos_cols.append(f"{bp}_x")
        pos_cols.append(f"{bp}_y")


    # 完整表头
    header = ["frame_index", "stage", "is_keyframe", "fileName"] + pos_cols 

    # 5. 准备存储错误检测数据的列表
    points_err_list = []
    stageid_err_list = []
    rev_map = {v: k for k, v in STAGE_MAP.items()}

    # 6. 处理每一帧并写入CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)

        for i in range(frame_count):
            # 获取当前帧的阶段信息
            stgName = stage_for_frame[i]
            is_key = (i in key_frames_set)
            stg_id = rev_map.get(stgName, 0)

            # 获取位置数据
            if i < keypoint_data.shape[0]:
                coords_2d = keypoint_data[i]
                pos_list = []
                for pidx in range(33):
                    pos_list.append(float(coords_2d[pidx, 0]))
                    pos_list.append(float(coords_2d[pidx, 1]))
            else:
                pos_list = [0.0] * 66


            # 保持fileName列兼容性
            dummy_filename = f"frame{i:04d}.jpg"

            # 写入CSV行
            row_data = [i, stgName, int(is_key), dummy_filename] + pos_list 
            writer.writerow(row_data)

            # 添加到错误检测数据
            points_err_list.append(np.array(pos_list, dtype=np.float32))
            stageid_err_list.append(stg_id)

    # 7. 释放资源并打印信息
        cap.release()
        print("[INFO] CSV =>", csv_path)

    return points_err_list, stageid_err_list


class VideoProcessor:
    @staticmethod
    def handle_video(video_path: str,
                     stage_indices: Dict[str, List[int]],
                     output_folder: str,
                     keypoint_data: torch.Tensor,
                     analyzer: SwingAnalyzer,
                     csv_filename: str = "allData.csv",
                     img_folder: str = None,
                     rotation_type: str = "none"):
        """
        处理视频并返回错误检测所需数据
        """
        return process_new_video(
            video_path=video_path,
            stage_indices=stage_indices,
            output_folder=output_folder,
            keypoint_data=keypoint_data,
            analyzer=analyzer,
            csv_filename=csv_filename,
            img_folder=img_folder,
            rotation_type=rotation_type
        )
