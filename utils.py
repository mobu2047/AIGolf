# utils.py
# -------------------
# 存放通用的工具函数，如时序平滑、角度差计算、方向校验等。

import math
import torch
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import json
import time
from PIL import Image, ImageDraw
from collections import defaultdict
import glob
import shutil
import subprocess
import platform
from datetime import datetime
import sys
from functools import lru_cache
from multiprocessing import Pool

# 添加计时器配置
ENABLE_TIMING = True  # 计时器开关

# 添加缓存控制
ENABLE_CACHE = True  # 缓存开关

# 图像处理优化配置
IMAGE_COMPRESSION_QUALITY = 85  # JPEG压缩质量(0-100)
IMAGE_MAX_DIMENSION = 3840     # 修改为4K分辨率(3840x2160)
ENABLE_IMAGE_OPTIMIZATION = True  # 图像优化开关
PRESERVE_ASPECT_RATIO = True   # 保持宽高比

# 并行处理配置
ENABLE_PARALLEL = True  # 并行处理开关
MAX_WORKERS = os.cpu_count() or 4  # 自动检测CPU核心数
CHUNK_SIZE = 4  # 进程池分块大小
SHARED_MEMORY_LIMIT = 1024 * 1024 * 1024  # 1GB共享内存限制

class Timer:
    """计时器类，用于统计各个处理步骤的耗时"""
    
    _timings = {}  # 存储各个步骤的耗时
    _start_times = {}  # 存储各个步骤的开始时间
    
    @classmethod
    def start(cls, step_name):
        """开始计时某个步骤"""
        if not ENABLE_TIMING:
            return
        from time import time
        cls._start_times[step_name] = time()
    
    @classmethod
    def end(cls, step_name):
        """结束计时某个步骤"""
        if not ENABLE_TIMING:
            return
        from time import time
        if step_name in cls._start_times:
            duration = time() - cls._start_times[step_name]
            if step_name not in cls._timings:
                cls._timings[step_name] = []
            cls._timings[step_name].append(duration)
            del cls._start_times[step_name]
    
    @classmethod
    def get_stats(cls):
        """获取所有步骤的统计信息"""
        if not ENABLE_TIMING:
            return {}
        import numpy as np
        stats = {}
        for step_name, durations in cls._timings.items():
            stats[step_name] = {
                'count': len(durations),
                'total': sum(durations),
                'mean': np.mean(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'std': np.std(durations) if len(durations) > 1 else 0
            }
        return stats
    
    @classmethod
    def print_stats(cls):
        """打印所有步骤的统计信息"""
        if not ENABLE_TIMING:
            return
        stats = cls.get_stats()
        print("\n=== 性能统计 ===")
        for step_name, step_stats in stats.items():
            print(f"\n{step_name}:")
            print(f"  执行次数: {step_stats['count']}")
            print(f"  总耗时: {step_stats['total']:.2f}秒")
            print(f"  平均耗时: {step_stats['mean']:.2f}秒")
            print(f"  最短耗时: {step_stats['min']:.2f}秒")
            print(f"  最长耗时: {step_stats['max']:.2f}秒")
            print(f"  标准差: {step_stats['std']:.2f}秒")
    
    @classmethod
    def reset(cls):
        """重置计时器"""
        if not ENABLE_TIMING:
            return
        cls._timings.clear()
        cls._start_times.clear()

def set_timing_enabled(enabled: bool):
    """设置是否启用计时功能"""
    global ENABLE_TIMING
    ENABLE_TIMING = enabled

def set_cache_enabled(enabled: bool):
    """设置是否启用缓存"""
    global ENABLE_CACHE
    ENABLE_CACHE = enabled
    # 清除所有缓存
    if not enabled:
        clear_all_caches()

def clear_all_caches():
    """清除所有已注册的缓存"""
    angle_difference.cache_clear()
    calculate_line_thickness.cache_clear()
    compute_angle.cache_clear()

def temporal_smoothing(keypoints: torch.Tensor, window_size=5) -> torch.Tensor:
    """
    对 (N,33,2) 的关键点数据使用时序滑动窗口平滑。
    返回与输入形状相同 (N,33,2)。
    """
    if keypoints.shape[0] < 2:
        return keypoints
    pad = window_size // 2
    padded = torch.cat([
        keypoints[:1].repeat(pad, 1, 1),
        keypoints,
        keypoints[-1:].repeat(pad, 1, 1)
    ], dim=0)

    padded = padded.permute(1, 2, 0)
    smoothed = torch.nn.functional.avg_pool1d(
        padded, kernel_size=window_size, stride=1, padding=0
    )
    smoothed = smoothed.permute(2, 0, 1)
    smoothed = smoothed[pad:-pad]
    return smoothed

@lru_cache(maxsize=128)
def angle_difference(current: float, reference: float) -> float:
    """计算带方向的最小角度差（考虑360度环绕）"""
    if not ENABLE_CACHE:
        return _angle_difference_impl(current, reference)
    return _angle_difference_impl(current, reference)

def _angle_difference_impl(current: float, reference: float) -> float:
    """角度差计算的实际实现"""
    diff = (current - reference + 180) % 360 - 180
    return diff + 360 if diff < -180 else diff

def is_proper_rotation(stage: str, angle: float) -> bool:
    """
    示例：判断某个阶段的旋转方向是否符合预期。
    例如 backswing → 负值(右旋), downswing → 正值(左旋)。
    """
    rotation_threshold = 10.0
    if stage == "backswing":
        return angle < -rotation_threshold
    elif stage == "downswing":
        return angle > rotation_threshold
    return True


def combine_stages_to_video(
        stage_for_frame,
        total_frames,
        assistant_first_folder,
        assistant_second_folder,
        output_video_folder=r".\resultData\video",
        output_video_name="result.mp4",
        fps=30,
        start_phase=1,  # 新增：开始阶段
        end_phase=8     # 新增：结束阶段
):
    """
    参数：
      stage_for_frame: list 或 dict，形如 stage_for_frame[i] = 阶段索引 (0 或其他)
                       或 stage_for_frame["1"] = [帧号列表]
      total_frames: 整个视频中总帧数
      assistant_first_folder: 存放0阶段帧的图像文件夹 (e.g. "./resultData/xxx/img/assistant_first")
      assistant_second_folder: 存放其它阶段帧的图像文件夹 (e.g. "./resultData/xxx/img/assistant_second")
      output_video_folder: 输出视频存放目录 (默认 .\resultData\video)
      output_video_name: 输出视频文件名 (默认 result.mp4)
      fps: 视频帧率 (默认25)
      start_phase: 开始阶段（默认1：准备阶段）
      end_phase: 结束阶段（默认8：收杆）
    """
    # 1) 如果输出目录不存在则创建
    os.makedirs(output_video_folder, exist_ok=True)
    output_path = os.path.join(output_video_folder, output_video_name)

    # 2) 先读取第0帧，对其确定帧宽高
    # 处理stage_for_frame根据其数据结构不同
    # 如果stage_for_frame是字典，且键是字符串（如"0"，"1"等），则转换为帧到阶段的映射
    is_stage_dict = isinstance(stage_for_frame, dict) and all(isinstance(k, str) for k in stage_for_frame.keys())
    
    if is_stage_dict:
        # 构建frame_to_stage映射
        frame_to_stage = {}
        for stage_str, frames in stage_for_frame.items():
            stage = int(stage_str)
            for frame in frames:
                frame_to_stage[frame] = stage
        
        # 寻找第一个阶段的分界点
        first_stage = 0
        # 尝试确定最后一个阶段0的帧
        stage0_frames = stage_for_frame.get("1", [])
        if stage0_frames:
            first_stage = min(stage0_frames)
    else:
        print(f"[WARN] 无法找到第0帧图像文件2: {output_path}")
        return
    
    first_img_path = os.path.join(assistant_first_folder, f"assist_{0:04d}.jpg")

    if not os.path.isfile(first_img_path):
        print(f"[WARN] 无法找到第0帧图像文件: {first_img_path}")
        return

    first_frame = cv2.imread(first_img_path)
    if first_frame is None:
        print(f"[ERR ] 无法读取第0帧图像: {first_img_path}")
        return

    height, width = first_frame.shape[:2]

    # 3) 初始化视频写出器，使用H264编解码器
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 如果H264失败，尝试使用两阶段方法：先创建临时视频，然后转换
    if not writer.isOpened():
        print(f"[WARN] H264编解码器直接创建失败，尝试使用两阶段方法...")
        temp_path = output_path + ".temp.avi"
        
        # 使用XVID创建临时视频
        temp_fourcc = cv2.VideoWriter_fourcc(*"XVID")
        temp_writer = cv2.VideoWriter(temp_path, temp_fourcc, fps, (width, height))
        
        if not temp_writer.isOpened():
            print(f"[ERR] 临时视频创建也失败: {temp_path}")
            return

        # 开始准备帧合成
        start = min(stage_for_frame.get("0", []))-int(fps)
        end = max(stage_for_frame.get("7", []))
        print(f"[INFO] 视频剪辑范围: 第{start}帧到第{end}帧")
            
        # 逐帧写入临时视频
        frames_written = 0
        for i in range(end - start + 1):
            if i <= first_stage:
                frame_path = os.path.join(assistant_first_folder, f"assist_{i:04d}.jpg")
            else:
                frame_path = os.path.join(assistant_second_folder, f"assist2_{i:04d}.jpg")

            if not os.path.isfile(frame_path):
                print(f"[WARN] 找不到图像: {frame_path}, 跳过写入.")
                continue

            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                print(f"[WARN] 无法读取图像: {frame_path}, 跳过写入.")
                continue

            # 如果图像尺寸与初始不一致，可以根据需要 resize
            if frame_img.shape[0] != height or frame_img.shape[1] != width:
                frame_img = cv2.resize(frame_img, (width, height))

            temp_writer.write(frame_img)
            frames_written += 1
        
        # 释放临时视频资源
        temp_writer.release()
        
        if frames_written == 0:
            print(f"[ERR] 临时视频创建失败: 没有写入任何帧")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return

        print(f"[INFO] 临时视频创建成功: {temp_path}, 正在转换为H264...")
        
        # 使用ffmpeg转换为H264
        try:
            import subprocess
            import shutil
            
            # 检查系统中是否有ffmpeg
            ffmpeg_available = shutil.which("ffmpeg") is not None
            
            if ffmpeg_available:
                # 使用ffmpeg转换为H264，简化命令参数
                cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_path,
                    "-c:v", "libx264",
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"[INFO] 成功转换为H264: {output_path}")
                    
                    # 删除临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                else:
                    print(f"[WARN] FFmpeg转换失败: {result.stderr}")
                    # 尝试使用最简单的命令
                    simpler_cmd = [
                        "ffmpeg", "-y",
                        "-i", temp_path,
                        output_path
                    ]
                    print(f"[INFO] 尝试使用更简单的命令进行转换...")
                    result = subprocess.run(simpler_cmd, capture_output=True, text=True)
                    
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"[INFO] 成功转换视频: {output_path}")
                        
                        # 删除临时文件
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    else:
                        print(f"[WARN] 简化命令也失败: {result.stderr}")
                        # 回退方案：直接使用avi临时文件
                        shutil.copy2(temp_path, output_path)
                        print(f"[INFO] 使用AVI格式作为备选方案: {output_path}")
            else:
                # ffmpeg不可用，直接使用avi临时文件
                print(f"[WARN] 系统中未找到ffmpeg，直接使用临时视频文件")
                shutil.copy2(temp_path, output_path)
                print(f"[INFO] 已复制视频文件: {output_path}")
        except Exception as e:
            print(f"[ERR] 视频转换过程中出错: {str(e)}")
            # 尝试使用临时文件作为最后的备选方案
            try:
                import shutil
                print(f"[INFO] 尝试使用临时视频文件作为备选...")
                shutil.copy2(temp_path, output_path)
                print(f"[INFO] 已复制视频文件: {output_path}")
                return
            except Exception as copy_err:
                print(f"[ERR] 复制文件失败: {str(copy_err)}")
                return
    else:
        # 直接使用H264
        start = max(0, min(stage_for_frame.get("0", []))-int(fps))
        end = max(stage_for_frame.get("7", []))
        print(f"[INFO] 视频剪辑范围: 第{start}帧到第{end}帧")

        # 逐帧写入（只写入剪辑范围内的帧）
        frames_written = 0
        for i in range(end - start + 1):
            if i <= first_stage:
                frame_path = os.path.join(assistant_first_folder, f"assist_{i:04d}.jpg")
            else:
                frame_path = os.path.join(assistant_second_folder, f"assist2_{i:04d}.jpg")

            if not os.path.isfile(frame_path):
                print(f"[WARN] 找不到图像: {frame_path}, 跳过写入.")
                continue

            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                print(f"[WARN] 无法读取图像: {frame_path}, 跳过写入.")
                continue

            # 如果图像尺寸与初始不一致，可以根据需要 resize
            if frame_img.shape[0] != height or frame_img.shape[1] != width:
                frame_img = cv2.resize(frame_img, (width, height))

            writer.write(frame_img)
            frames_written += 1

        writer.release()
        # 保存视频
        if os.path.exists(output_path) and frames_written > 0:
            print(f"[INFO] 成功使用H264编解码器合成视频 => {output_path}, 共写入 {frames_written} 帧")
            
            # 确保文件权限正确
            ensure_file_permissions(output_path)
            
            return True
        else:
            print(f"[ERR] 视频合成失败: 没有写入任何帧 => {output_path}")
            return False

def extract_golf_swing_video(
    stage_indices,
    input_video_path: str = None,
    output_video_path: str = None,
    start_phase: str = "1",  # 从准备阶段开始
    end_phase: str = "8",    # 到收杆结束
    fps: int = 30
):
    """
    从输入视频中提取高尔夫挥杆关键阶段的视频
    
    参数:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        start_phase: 开始阶段（默认1：准备阶段）
        end_phase: 结束阶段（默认8：收杆）
        fps: 输出视频帧率
    """
    try:
        
        # 3. 确定关键帧范围
        start_frame = stage_indices.get(start_phase)[0]-int(fps)
        end_frame = stage_indices.get(end_phase)[-1]
        
        if start_frame >= end_frame:
            print("[ERR] 无法确定有效的挥杆阶段范围")
            return None, None
            
        print(f"[INFO] 提取挥杆阶段: 第{start_frame}帧到第{end_frame}帧")
        return start_frame, end_frame
        
     
            
    except Exception as e:
        print(f"[ERR] 提取视频片段时发生错误: {str(e)}")
        return None, None

def generate_rotated_video(
    input_video_path: str,
    output_video_path: str,
    fps: int = 30
):
    """
    根据需要生成旋转后的视频供用户查看
    
    参数:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        fps: 输出视频帧率
    """
    try:
        # 1. 读取输入视频
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"[ERR] 无法打开输入视频: {input_video_path}")
            return False
            
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 2. 检查是否需要旋转视频
        need_rotate = (width > height)
        if not need_rotate:
            print(f"[INFO] 视频不需要旋转，尺寸=({height},{width})")
            return False
            
        print(f"[DEBUG] 视频需要旋转，尺寸=({height},{width})")
        
        # 3. 创建输出视频，使用H264编解码器
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (height, width))  # 交换宽高
        
        # 如果H264失败，尝试使用两阶段方法：先创建临时视频，然后转换
        if not out.isOpened():
            print(f"[WARN] H264编解码器直接创建失败，尝试使用两阶段方法...")
            temp_path = output_video_path + ".temp.avi"
            
            # 使用XVID创建临时视频
            temp_fourcc = cv2.VideoWriter_fourcc(*"XVID")
            temp_out = cv2.VideoWriter(temp_path, temp_fourcc, fps, (height, width))  # 交换宽高
            
            if not temp_out.isOpened():
                print(f"[ERR] 临时视频创建也失败: {temp_path}")
                return False
                
            # 逐帧读取并旋转，写入临时视频
            frames_written = 0
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 顺时针旋转90度
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                temp_out.write(rotated_frame)
                frames_written += 1
                
            # 释放资源
            cap.release()
            temp_out.release()
            
            if frames_written == 0:
                print(f"[ERR] 临时视频创建失败: 没有写入任何帧")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
                
            print(f"[INFO] 临时视频创建成功: {temp_path}, 正在转换为H264...")
            
            # 使用ffmpeg转换为H264
            try:
                import subprocess
                import shutil
                
                # 检查系统中是否有ffmpeg
                ffmpeg_available = shutil.which("ffmpeg") is not None
                
                if ffmpeg_available:
                    # 多种转换命令尝试
                    conversion_attempts = [
                        # 1. 尝试使用libx264
                        [
                            "ffmpeg", "-y",
                            "-i", temp_path,
                            "-c:v", "libx264",
                            output_video_path
                        ],
                        # 2. 尝试使用h264_nvenc (NVIDIA GPU加速)
                        [
                            "ffmpeg", "-y",
                            "-i", temp_path,
                            "-c:v", "h264_nvenc",
                            output_video_path
                        ],
                        # 3. 尝试使用avc
                        [
                            "ffmpeg", "-y",
                            "-i", temp_path,
                            "-c:v", "avc",
                            output_video_path
                        ],
                        # 4. 尝试使用h264
                        [
                            "ffmpeg", "-y",
                            "-i", temp_path,
                            "-c:v", "h264",
                            output_video_path
                        ],
                        # 5. 最简单的命令，让ffmpeg自动选择
                        [
                            "ffmpeg", "-y",
                            "-i", temp_path,
                            output_video_path
                        ]
                    ]
                    
                    # 尝试每一种转换命令
                    success = False
                    for i, cmd in enumerate(conversion_attempts):
                        try:
                            print(f"[INFO] 尝试转换方法 {i+1}/{len(conversion_attempts)}: {' '.join(cmd)}")
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                                print(f"[INFO] 成功转换为H264旋转视频: {output_video_path}")
                                
                                # 删除临时文件
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                
                                # 确保文件权限正确
                                ensure_file_permissions(output_video_path)
                                
                                success = True
                                break
                            else:
                                print(f"[WARN] 方法 {i+1} 转换失败: {result.stderr}")
                        except Exception as e:
                            print(f"[WARN] 方法 {i+1} 执行出错: {str(e)}")
                            continue
                    
                    if success:
                        return True
                    else:
                        print(f"[ERR] 所有转换方法都失败了")
                        return False
                else:
                    print(f"[WARN] 系统中未找到ffmpeg，无法进行视频转换")
                    return False
            except Exception as e:
                print(f"[ERR] 视频转换过程中出错: {str(e)}")
                return False
        else:
            # 直接使用H264写入
            frames_written = 0
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 顺时针旋转90度
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                out.write(rotated_frame)
                frames_written += 1
                
            # 释放资源
            cap.release()
            out.release()
            
            if frames_written > 0:
                print(f"[INFO] 使用H264编解码器成功生成旋转视频: {output_video_path}, 共写入 {frames_written} 帧")
                
                # 确保文件权限正确
                ensure_file_permissions(output_video_path)
                
                return True
            else:
                print(f"[ERR] 旋转视频创建失败: 没有写入任何帧")
                return False
        
    except Exception as e:
        print(f"[ERR] 生成旋转视频时发生错误: {str(e)}")
        return False

def convert_to_h264(
    input_video_path: str,
    output_video_path: str = None,
    overwrite_input: bool = False
):
    """
    将视频重新编码为H264格式
    
    参数:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径，如果为None则生成临时文件
        overwrite_input: 是否在转换后覆盖原始文件
        
    返回:
        bool: 是否成功转换
    """
    try:
        # 如果未指定输出路径，创建临时输出路径
        if output_video_path is None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                output_video_path = tmp.name
        
        import shutil
        import subprocess
        
        # 检查系统中是否有ffmpeg
        ffmpeg_available = shutil.which("ffmpeg") is not None
        
        if not ffmpeg_available:
            print(f"[ERR] 系统中未找到ffmpeg，无法转换为H264")
            return False
        
        # 尝试使用多种命令进行转换，从最好的选项开始
        conversion_attempts = [
            # 1. 尝试使用libx264
            [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-c:v", "h264",
                output_video_path
            ],
            # 2. 尝试使用h264_nvenc (NVIDIA GPU加速)
            [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-c:v", "h264_nvenc",
                output_video_path
            ],
            # 3. 尝试使用avc
            [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-c:v", "avc",
                output_video_path
            ],
            # 4. 尝试使用h264
            [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-c:v", "libx264",
                output_video_path
            ],
            # 5. 最简单的命令，让ffmpeg自动选择
            [
                "ffmpeg", "-y",
                "-i", input_video_path,
                output_video_path
            ]
        ]
        
        print(f"[INFO] 开始将视频转换为H264格式: {input_video_path} -> {output_video_path}")
        
        # 尝试每一种转换命令
        for i, cmd in enumerate(conversion_attempts):
            try:
                print(f"[INFO] 尝试转换方法 {i+1}/{len(conversion_attempts)}: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                    print(f"[INFO] 成功转换为H264格式: {output_video_path}")
                    
                    # 确保文件权限正确
                    ensure_file_permissions(output_video_path)
                    
                    # 如果需要覆盖原始文件
                    if overwrite_input:
                        # 创建备份
                        backup_path = input_video_path + ".bak"
                        shutil.copy2(input_video_path, backup_path)
                        
                        try:
                            # 用新文件替换原始文件
                            shutil.copy2(output_video_path, input_video_path)
                            
                            # 确保替换后的文件权限正确
                            ensure_file_permissions(input_video_path)
                            
                            # 删除临时输出文件和备份
                            os.remove(output_video_path)
                            os.remove(backup_path)
                            
                            print(f"[INFO] 已将原始文件替换为H264编码版本: {input_video_path}")
                        except Exception as copy_err:
                            print(f"[ERR] 替换原始文件失败: {str(copy_err)}")
                            # 恢复备份
                            shutil.copy2(backup_path, input_video_path)
                            print(f"[INFO] 已从备份恢复原始文件: {input_video_path}")
                            return False
                    
                    return True
                else:
                    print(f"[WARN] 方法 {i+1} 转换失败: {result.stderr}")
            except Exception as e:
                print(f"[WARN] 方法 {i+1} 执行出错: {str(e)}")
                continue
        
        print(f"[ERR] 所有转换方法都失败了")
        return False
                
    except Exception as e:
        print(f"[ERR] 视频转换过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def ensure_file_permissions(file_path, mode=0o644):
    """
    确保文件具有正确的权限，默认设置为所有用户可读，所有者可写
    
    参数:
        file_path: 文件路径
        mode: 权限模式，默认0o644 (所有者读写，其他人只读)
    """
    try:
        # 仅在类Unix系统上设置权限
        if platform.system() != "Windows":
            import os
            current_mode = os.stat(file_path).st_mode & 0o777
            if current_mode != mode:
                os.chmod(file_path, mode)
                print(f"[INFO] 已设置文件权限: {file_path} (mode: {mode:o})")
        
        return True
    except Exception as e:
        print(f"[WARN] 设置文件权限失败: {file_path}, 错误: {str(e)}")
        return False

def draw_skeleton(keypoints, image=None, thickness=2, color=(0, 0, 255), show_image=False):
    """
    根据关键点绘制骨线图
    
    参数:
        keypoints: 形状为(33,2)的numpy数组或tensor，包含33个关键点坐标
        image: 可选，背景图像。如果为None，则创建空白画布
        thickness: 线条粗细，默认为2
        color: BGR颜色元组，默认为红色(0,0,255)
        show_image: 是否显示图像，默认为False
        
    返回:
        绘制了骨线图的图像
    """
    import cv2
    import numpy as np
    import torch
    
    # 将tensor转换为numpy数组
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    
    # 确保keypoints是整数坐标
    keypoints = keypoints.astype(np.int32)
    
    # 如果没有提供背景图像，创建空白画布
    if image is None:
        # 找出关键点的边界范围
        valid_points = keypoints[~np.isnan(keypoints).any(axis=1)]  # 过滤掉NaN点
        if len(valid_points) > 0:
            max_x, max_y = np.max(valid_points, axis=0) + 50
            min_x, min_y = np.min(valid_points, axis=0) - 50
            width = max(640, max_x - min_x)
            height = max(480, max_y - min_y)
        else:
            width, height = 640, 480
        
        # 创建空白画布
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
    else:
        # 确保图像是彩色的
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
    # 定义骨骼连接
    # MediaPipe姿态关键点连接定义
    connections = [
        # 面部连接
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # 身体主干
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # 躯干
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), 
        # 手臂
        (11, 13), (13, 15), (12, 14), (14, 16),
        # 手部
        (15, 17), (15, 19), (15, 21), (17, 19),
        (16, 18), (16, 20), (16, 22), (18, 20),
        # 腿部
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32)
    ]
    
    # 绘制关键点和连接线
    for idx, (x, y) in enumerate(keypoints):
        # 检查坐标是否有效
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(image, (x, y), 4, color, -1)  # 绘制关键点
            cv2.putText(image, str(idx), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 绘制骨骼连接
    for connection in connections:
        idx1, idx2 = connection
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            pt1 = tuple(keypoints[idx1])
            pt2 = tuple(keypoints[idx2])
            # 检查坐标是否有效
            if not (np.isnan(pt1[0]) or np.isnan(pt1[1]) or np.isnan(pt2[0]) or np.isnan(pt2[1])):
                cv2.line(image, pt1, pt2, color, thickness)
    
    # 如果需要显示图像，且GUI可用
    # if show_image and GUI_ENABLED:
    #     cv2.imshow("骨线图", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    return image

def combine_img_to_video(
    img_folder: str,
    output_path: str,
    fps: float = 30.0,
    img_pattern: str = "*.jpg",
    overwrite: bool = True
):
    """
    将指定目录下的图像序列合成为视频
    
    参数:
        img_folder: 图像文件夹路径
        output_path: 输出视频路径
        fps: 帧率，默认30fps
        img_pattern: 图像匹配模式，默认"*.jpg"
        overwrite: 是否覆盖现有视频文件
        
    返回:
        dict: 包含状态和信息的字典
    """
    try:
        import os
        import cv2
        import shutil
        import glob
        
        # 检查输入目录是否存在
        if not os.path.exists(img_folder):
            return {"status": "error", "message": f"图像目录不存在: {img_folder}"}
        
        # 确保视频输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取所有图像文件并按顺序排序
        image_files = glob.glob(os.path.join(img_folder, img_pattern))
        if not image_files:
            return {"status": "error", "message": f"目录中没有图像文件: {img_folder}"}
            
        # 按帧号排序
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
        
        # 读取第一张图像以获取尺寸
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            return {"status": "error", "message": f"无法读取图像: {image_files[0]}"}
            
        height, width = first_image.shape[:2]
        
        # 检查是否已存在输出文件
        if os.path.exists(output_path) and not overwrite:
            return {"status": "info", "message": f"视频文件已存在且未指定覆盖: {output_path}"}
        
        # 创建临时输出文件
        temp_output_path = output_path + ".temp.mp4"
        
        # 检查ffmpeg是否可用
        ffmpeg_available = shutil.which("ffmpeg") is not None
        
        # 使用ffmpeg尝试创建H264视频
        if ffmpeg_available:
            # 首先尝试多种H264相关的编码方式
            try:
                import subprocess
                
                # 为ffmpeg命令创建多种H264相关的编码方案
                ffmpeg_encoding_attempts = [
                    # 1. 最标准的libx264
                    [
                        "ffmpeg", "-y",
                        "-framerate", str(fps),
                        "-pattern_type", "glob",
                        "-i", os.path.join(img_folder, img_pattern),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", "23",
                        temp_output_path
                    ],
                    # 2. 尝试h264
                    [
                        "ffmpeg", "-y",
                        "-framerate", str(fps),
                        "-pattern_type", "glob",
                        "-i", os.path.join(img_folder, img_pattern),
                        "-c:v", "h264",
                        "-pix_fmt", "yuv420p",
                        temp_output_path
                    ],
                    # 3. 尝试avc
                    [
                        "ffmpeg", "-y",
                        "-framerate", str(fps),
                        "-pattern_type", "glob",
                        "-i", os.path.join(img_folder, img_pattern),
                        "-c:v", "avc",
                        "-pix_fmt", "yuv420p",
                        temp_output_path
                    ],
                    # 4. 最简单的命令，让ffmpeg自动选择
                    [
                        "ffmpeg", "-y",
                        "-framerate", str(fps),
                        "-pattern_type", "glob",
                        "-i", os.path.join(img_folder, img_pattern),
                        temp_output_path
                    ]
                ]
                
                ffmpeg_success = False
                for i, cmd in enumerate(ffmpeg_encoding_attempts):
                    try:
                        print(f"[INFO] 尝试ffmpeg编码方式 {i+1}/{len(ffmpeg_encoding_attempts)}: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                            print(f"[INFO] 成功使用ffmpeg方式 {i+1} 创建视频")
                            ffmpeg_success = True
                            break
                        else:
                            print(f"[WARN] ffmpeg方式 {i+1} 失败: {result.stderr}")
                    except Exception as e:
                        print(f"[WARN] ffmpeg方式 {i+1} 出错: {str(e)}")
                        continue
                
                if not ffmpeg_success:
                    print(f"[WARN] 所有ffmpeg方法都失败，尝试OpenCV方法")
                    raise Exception("所有ffmpeg编码方式均失败")
            except Exception as e:
                print(f"[WARN] ffmpeg合成出错，使用OpenCV备选方案: {str(e)}")
                ffmpeg_available = False
        
        # 如果ffmpeg不可用或失败，使用OpenCV
        if not ffmpeg_available:
            try:
                # 创建临时AVI文件，稍后转换为H264 MP4
                avi_temp_path = temp_output_path + ".avi"
                print(f"[INFO] 使用OpenCV创建临时AVI文件: {avi_temp_path}")
                
                # 对于OpenCV，先使用XVID创建AVI，这是最通用的编码器
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(avi_temp_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    # 尝试不同的编码器
                    encoding_options = ['H264', 'DIVX', 'MJPG', 'MP4V']
                    for codec in encoding_options:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            out = cv2.VideoWriter(avi_temp_path, fourcc, fps, (width, height))
                            if out.isOpened():
                                print(f"[INFO] 成功使用 {codec} 编码器")
                                break
                        except Exception as codec_err:
                            print(f"[WARN] {codec} 编码器失败: {str(codec_err)}")
                
                if not out.isOpened():
                    return {"status": "error", "message": "无法创建视频文件，所有编码器都失败"}
                
                # 读取每一帧并写入视频
                frames_written = 0
                print(f"[INFO] 开始处理 {len(image_files)} 帧图像")
                for i, img_path in enumerate(image_files):
                    frame = cv2.imread(img_path)
                    
                    if frame is None:
                        print(f"[WARN] 无法读取图像: {img_path}")
                        continue
                        
                    # 确保帧尺寸正确
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                        
                    out.write(frame)
                    frames_written += 1
                    
                    # 每100帧显示进度
                    if i % 100 == 0 or i == len(image_files) - 1:
                        print(f"[INFO] 已处理 {i+1}/{len(image_files)} 帧")
                
                # 释放资源
                out.release()
                print(f"[INFO] 临时AVI文件创建完成，共写入 {frames_written} 帧")
                
                # 检查临时AVI文件
                if not os.path.exists(avi_temp_path) or os.path.getsize(avi_temp_path) < 1024:
                    return {"status": "error", "message": "视频生成失败，临时文件为空或太小"}
                
                # 使用ffmpeg将AVI转换为H264 MP4
                if ffmpeg_available:
                    print(f"[INFO] 尝试将AVI转换为H264 MP4")
                    try:
                        # 转换命令
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", avi_temp_path,
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            temp_output_path
                        ]
                        subprocess.run(cmd, capture_output=True, text=True)
                        
                        if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                            print(f"[INFO] 成功将AVI转换为MP4")
                        else:
                            # 如果转换失败，直接使用AVI文件
                            print(f"[WARN] AVI转MP4失败，直接使用AVI文件")
                            temp_output_path = avi_temp_path
                    except Exception as e:
                        print(f"[WARN] 转换失败: {str(e)}，使用AVI文件")
                        temp_output_path = avi_temp_path
                else:
                    temp_output_path = avi_temp_path
                    print(f"[INFO] ffmpeg不可用，使用AVI文件")
            except Exception as e:
                return {"status": "error", "message": f"OpenCV视频合成失败: {str(e)}"}
        
        # 最后将临时文件移动到最终位置
        shutil.move(temp_output_path, output_path)
        print(f"[INFO] 视频合成完成: {output_path}")
        
        # 清理所有临时文件
        for temp_file in [temp_output_path, temp_output_path + ".avi"]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"[WARN] 无法删除临时文件 {temp_file}: {str(e)}")
        
        return {"status": "success", "message": f"成功创建视频: {output_path}"}
    except Exception as e:
        return {"status": "error", "message": f"视频合成过程中出错: {str(e)}"}

def combine_img_to_video_session(
    session_id: str,
    fps: float = 30.0,
    overwrite: bool = True
):
    """
    将img/all目录下的图像序列合成为H264编码的视频
    
    参数:
        session_id: 会话ID
        fps: 帧率，默认30fps
        overwrite: 是否覆盖现有视频文件
        
    返回:
        dict: 包含状态和信息的字典
    """
    try:
        import os
        
        # 构建路径
        session_folder = os.path.join("./resultData", session_id)
        img_folder = os.path.join(session_folder, "img", "all")
        output_path = os.path.join(session_folder, "video", "original.mp4")
        
        return combine_img_to_video(
            img_folder=img_folder,
            output_path=output_path,
            fps=fps,
            img_pattern="*.jpg",
            overwrite=overwrite
        )
    except Exception as e:
        return {"status": "error", "message": f"处理会话视频时出错: {str(e)}"}

@lru_cache(maxsize=128)
def calculate_line_thickness(base_thickness: int, video_width: int, video_height: int) -> int:
    """根据视频分辨率计算线条粗细"""
    if not ENABLE_CACHE:
        return _calculate_line_thickness_impl(base_thickness, video_width, video_height)
    return _calculate_line_thickness_impl(base_thickness, video_width, video_height)

def _calculate_line_thickness_impl(base_thickness: int, video_width: int, video_height: int) -> int:
    """线条粗细计算的实际实现"""
    base_diagonal = (1920 ** 2 + 1080 ** 2) ** 0.5
    current_diagonal = (video_width ** 2 + video_height ** 2) ** 0.5
    scale = current_diagonal / base_diagonal
    return max(1, int(base_thickness * scale))

@lru_cache(maxsize=256)
def compute_angle(p1: tuple, p2: tuple, p3: tuple) -> float:
    """计算三个点形成的角度"""
    if not ENABLE_CACHE:
        return _compute_angle_impl(p1, p2, p3)
    return _compute_angle_impl(p1, p2, p3)

def _compute_angle_impl(p1: tuple, p2: tuple, p3: tuple) -> float:
    """角度计算的实际实现"""
    import numpy as np
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def set_image_optimization_enabled(enabled: bool):
    """设置是否启用图像优化"""
    global ENABLE_IMAGE_OPTIMIZATION
    ENABLE_IMAGE_OPTIMIZATION = enabled

def set_image_compression_quality(quality: int):
    """设置JPEG压缩质量(0-100)"""
    global IMAGE_COMPRESSION_QUALITY
    IMAGE_COMPRESSION_QUALITY = max(0, min(100, quality))

def optimize_image_save(image: np.ndarray, save_path: str, optimize: bool = None, preserve_resolution: bool = False) -> bool:
    """
    优化图像保存过程
    
    参数:
        image: OpenCV图像数组
        save_path: 保存路径
        optimize: 是否启用优化，默认使用全局设置
        preserve_resolution: 是否保持原始分辨率，用于处理4K等高分辨率视频
        
    返回:
        bool: 是否保存成功
    """
    try:
        if optimize is None:
            optimize = ENABLE_IMAGE_OPTIMIZATION
            
        if not optimize:
            return cv2.imwrite(save_path, image)
            
        # 检查文件扩展名
        ext = os.path.splitext(save_path)[1].lower()
        
        if ext == '.jpg' or ext == '.jpeg':
            # JPEG优化
            params = [cv2.IMWRITE_JPEG_QUALITY, IMAGE_COMPRESSION_QUALITY]
        elif ext == '.png':
            # PNG优化
            params = [cv2.IMWRITE_PNG_COMPRESSION, 7]  # 0-9，越大压缩率越高
        else:
            # 其他格式使用默认参数
            params = []
            
        # 尺寸优化（考虑4K视频的情况）
        h, w = image.shape[:2]
        if not preserve_resolution and max(h, w) > IMAGE_MAX_DIMENSION:
            if PRESERVE_ASPECT_RATIO:
                # 保持宽高比
                scale = IMAGE_MAX_DIMENSION / max(h, w)
                new_size = (int(w * scale), int(h * scale))
            else:
                # 强制限制最大尺寸
                new_size = (
                    min(w, IMAGE_MAX_DIMENSION),
                    min(h, IMAGE_MAX_DIMENSION)
                )
            # 使用INTER_AREA进行下采样，效果更好
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
        # 保存图像
        success = cv2.imwrite(save_path, image, params)
        
        if success:
        # 确保文件权限正确
            ensure_file_permissions(save_path)
            
        return success
    except Exception as e:
        print(f"[WARN] 图像保存失败 {save_path}: {str(e)}")
        return False

def optimize_image_read(image_path: str, optimize: bool = None, preserve_resolution: bool = False) -> np.ndarray:
    """
    优化图像读取过程
    
    参数:
        image_path: 图像路径
        optimize: 是否启用优化，默认使用全局设置
        preserve_resolution: 是否保持原始分辨率，用于处理4K等高分辨率视频
        
    返回:
        numpy.ndarray: 图像数组，读取失败返回None
    """
    try:
        if optimize is None:
            optimize = ENABLE_IMAGE_OPTIMIZATION
            
        if not optimize or preserve_resolution:
            return cv2.imread(image_path)
            
        # 对于大图像，先尝试使用降采样读取
        image = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)  # 降采样比例改为2
        
        if image is None:
            # 如果降采样读取失败，尝试普通读取
            image = cv2.imread(image_path)
            
        return image
    except Exception as e:
        print(f"[WARN] 图像读取失败 {image_path}: {str(e)}")
        return None

# 优化后的图像批量保存函数
def save_images_batch(images: list, save_dir: str, prefix: str = "frame", optimize: bool = None, preserve_resolution: bool = False) -> bool:
    """
    批量保存图像
    
    参数:
        images: 图像列表
        save_dir: 保存目录
        prefix: 文件名前缀
        optimize: 是否启用优化，默认使用全局设置
        preserve_resolution: 是否保持原始分辨率，用于处理4K等高分辨率视频
        
    返回:
        bool: 是否全部保存成功
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        success_count = 0
        
        for i, image in enumerate(images):
            save_path = os.path.join(save_dir, f"{prefix}{i:04d}.jpg")
            if optimize_image_save(image, save_path, optimize, preserve_resolution):
                success_count += 1
                
        return success_count == len(images)
    except Exception as e:
        print(f"[WARN] 批量保存图像失败: {str(e)}")
        return False

def detect_video_resolution(video_path: str) -> tuple:
    """
    检测视频分辨率
    
    参数:
        video_path: 视频文件路径
        
    返回:
        tuple: (width, height) 或 None（如果检测失败）
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        return (width, height)
    except Exception as e:
        print(f"[WARN] 视频分辨率检测失败 {video_path}: {str(e)}")
        return None

def set_parallel_enabled(enabled: bool):
    """设置是否启用并行处理"""
    global ENABLE_PARALLEL
    ENABLE_PARALLEL = enabled

def set_max_workers(workers: int):
    """设置最大工作进程数"""
    global MAX_WORKERS
    MAX_WORKERS = max(1, min(workers, os.cpu_count() or 4))

def optimize_batch_size(frame_size: tuple, available_memory: int = None) -> int:
    """
    根据帧大小和可用内存优化批处理大小
    
    参数:
        frame_size: (height, width, channels)
        available_memory: 可用内存（字节）
        
    返回:
        最优批处理大小
    """
    if available_memory is None:
        import psutil
        available_memory = psutil.virtual_memory().available
    
    # 计算单帧内存占用
    frame_memory = frame_size[0] * frame_size[1] * frame_size[2] * 4  # 4字节/像素
    
    # 预留50%内存给其他进程
    safe_memory = available_memory * 0.5
    
    # 计算最大批处理大小
    max_batch = int(safe_memory / frame_memory)
    
    # 确保批处理大小在合理范围内
    return max(1, min(max_batch, 64))

def create_shared_array(shape: tuple, dtype=np.float32):
    """创建共享内存数组"""
    from multiprocessing import shared_memory
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if size > SHARED_MEMORY_LIMIT:
        raise ValueError(f"共享内存大小超过限制: {size} > {SHARED_MEMORY_LIMIT}")
    shm = shared_memory.SharedMemory(create=True, size=size)
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf), shm

def extract_keypoints_parallel(
    video_path: str,
    model,
    start_frame: int = None,
    end_frame: int = None,
    batch_size: int = None
) -> torch.Tensor:
    """
    并行提取视频关键点（优化版）
    """
    try:
        if not ENABLE_PARALLEL:
            return extract_keypoints_sequential(video_path, model, start_frame, end_frame)
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERR] 无法打开视频: {video_path}")
            return torch.tensor([])
        
        # 获取帧大小
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3
        )
        
        # 优化批处理大小
        if batch_size is None:
            batch_size = optimize_batch_size(frame_size)
        print(f"[INFO] 使用批处理大小: {batch_size}")
        
        # 设置起始帧
        if start_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 计算需要处理的帧数
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = end_frame - (start_frame or 0)
        
        # 创建共享内存数组存储结果
        try:
            result_shape = (frame_count, 33, 2)
            shared_array, shm = create_shared_array(result_shape)
        except Exception as e:
            print(f"[WARN] 创建共享内存失败，使用普通数组: {str(e)}")
            shared_array = np.zeros(result_shape, dtype=np.float32)
            shm = None
        
        # 将模型移动到GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        def process_batch(batch_frames, batch_start):
            """处理一批帧的函数"""
            try:
                # 预处理帧
                batch_tensor = torch.stack([
                    preprocess_frame(frame, device)
                    for frame in batch_frames
                    if frame is not None
                ]).to(device)
                
                # 提取关键点
                with torch.no_grad():
                    batch_keypoints = model(batch_tensor)
                    if len(batch_keypoints.shape) == 3:
                        batch_keypoints = batch_keypoints.view(-1, 33, 2)
                    
                    # 将结果写入共享内存
                    for i, kp in enumerate(batch_keypoints.cpu().numpy()):
                        shared_array[batch_start + i] = kp
                
                return True
            except Exception as e:
                print(f"[WARN] 批处理失败: {str(e)}")
                return False
        
        current_frame = 0
        with Pool(MAX_WORKERS) as pool:
            while current_frame < frame_count:
                # 读取一批帧
                batch_frames = []
                batch_start = current_frame
                for _ in range(min(batch_size, frame_count - current_frame)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                    current_frame += 1
                
                if not batch_frames:
                    break
                
                # 使用进程池处理批次
                pool.apply_async(
                    process_batch,
                    args=(batch_frames, batch_start),
                    callback=lambda _: print(
                        f"\r[INFO] 关键点提取进度: {current_frame/frame_count*100:.1f}%",
                        end=""
                    )
                )
            
            # 等待所有任务完成
            pool.close()
            pool.join()
        
        cap.release()
        print("\n[INFO] 关键点提取完成")
        
        # 转换结果为tensor
        result = torch.from_numpy(shared_array.copy())
        
        # 清理共享内存
        if shm is not None:
            shm.close()
            shm.unlink()
        
        print(f"[INFO] 提取到关键点数据，形状: {result.shape}")
        return result
        
    except Exception as e:
        print(f"[ERR] 并行提取关键点失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return torch.tensor([])

def extract_keypoints_sequential(
    video_path: str,
    model,
    start_frame: int = None,
    end_frame: int = None
) -> torch.Tensor:
    """顺序提取关键点（作为并行处理的备选方案）"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return torch.tensor([])
            
        # 设置起始帧
        if start_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 计算需要处理的帧数
        if end_frame is None:
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = end_frame - (start_frame or 0)
        
        all_keypoints = []
        current_frame = start_frame or 0
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 预处理帧
            preprocessed = preprocess_frame(frame, device)
            if preprocessed is None:
                continue
                
            # 提取关键点
            try:
                with torch.no_grad():
                    keypoints = model(preprocessed.unsqueeze(0))
                    # 确保输出形状为(1,33,2)
                    if len(keypoints.shape) == 3:
                        keypoints = keypoints.view(-1, 33, 2)
                    all_keypoints.append(keypoints[0].cpu())
            except Exception as e:
                print(f"[WARN] 单帧关键点提取失败: {str(e)}")
                continue
            
            current_frame += 1
            
            # 打印进度
            progress = (current_frame - (start_frame or 0)) / frame_count * 100
            print(f"\r[INFO] 关键点提取进度: {progress:.1f}%", end="")
        
        cap.release()
        print("\n[INFO] 关键点提取完成")
        
        if not all_keypoints:
            print(f"[WARN] 未提取到任何关键点")
            return torch.tensor([])
            
        # 转换为tensor并确保形状正确
        result = torch.stack(all_keypoints)
        print(f"[INFO] 提取到关键点数据，形状: {result.shape}")
        return result
        
    except Exception as e:
        print(f"[ERR] 顺序提取关键点失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return torch.tensor([])

def preprocess_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """预处理视频帧用于关键点检测"""
    try:
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整大小（如果需要）
        # frame_resized = cv2.resize(frame_rgb, (width, height))
        
        # 归一化
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # 转换为tensor
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
        
        # 添加batch维度并移动到指定设备
        return frame_tensor.to(device)
    except Exception as e:
        print(f"[WARN] 预处理帧失败: {str(e)}")
        return None

def extract_frames_to_images(video_path, output_dir, start_frame, end_frame, fps=30.0):
    """
    将视频中的帧提取为图片
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录
        start_frame: 起始帧
        end_frame: 结束帧
        fps: 帧率
        
    返回:
        frame_count: 提取的帧数
        need_rotate: 是否需要旋转视频
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 检测操作系统，为Windows添加特殊处理
    is_windows = platform.system() == "Windows"
    if is_windows:
        print(f"[INFO] 检测到Windows系统，使用特殊的帧提取方式")
    
    # 尝试多种视频捕获方式，解决Windows下的重影问题
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频: {video_path}")
            return 0, False
            
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        need_rotate = (width > height)  # 判断是否需要旋转
        
        # 确保帧索引在有效范围内
        start_frame = max(0, start_frame) if start_frame is not None else 0
        end_frame = min(total_frame_count - 1, end_frame) if end_frame is not None else total_frame_count - 1
        frame_count = end_frame - start_frame + 1
        
        print(f"[INFO] 正在保存视频帧到 {output_dir}，帧范围：{start_frame}-{end_frame}，共 {frame_count} 帧")
        
        # 实际写入的帧计数
        frames_written = 0
        
        # 在Windows上使用逐帧重新打开的方式解决重影问题
        if is_windows:
            cap.release()  # 先释放资源
            
            # 对于每一帧都重新打开视频文件，避免缓冲区问题
            for i in range(frame_count):
                frame_cap = cv2.VideoCapture(video_path)
                if not frame_cap.isOpened():
                    print(f"[ERROR] 读取第 {start_frame+i} 帧时无法打开视频")
                    continue
                
                # 设置到指定帧位置
                frame_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                
                # 读取单帧
                ret, frame = frame_cap.read()
                frame_cap.release()  # 立即释放资源
                
                if not ret:
                    print(f"[WARN] 读取第 {start_frame+i} 帧失败，跳过")
                    continue
                    
                # 如果需要旋转(横屏视频)，进行90度顺时针旋转
                if need_rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                # 保存图片，使用从0开始的帧索引
                out_name = f"frame{i:04d}.jpg"
                # 设置写入图像的质量
                image_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # 质量范围0-100，95为高质量
                
                if cv2.imwrite(os.path.join(output_dir, out_name), frame, image_quality):
                    frames_written += 1
                else:
                    print(f"[WARN] 保存第 {i} 帧到 {out_name} 失败")
                
                # 释放内存
                del frame
                
                # 每100帧显示进度
                if i % 100 == 0 or i == frame_count - 1:
                    print(f"[INFO] 帧提取进度: {i+1}/{frame_count}，已保存: {frames_written}帧")
                
                # 强制垃圾回收
                if i % 100 == 0:
                    import gc
                    gc.collect()
                    
        else:
            # Linux等其他系统使用原来的方式提取帧
            # 定位到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 设置写入图像的质量
            image_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # 质量范围0-100，95为高质量
            
            # 只处理从start_frame到end_frame的帧
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARN] 读取第 {start_frame+i} 帧失败，提前结束")
                    break
                    
                # 如果需要旋转(横屏视频)，进行90度顺时针旋转
                if need_rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    
                # 保存图片，使用从0开始的帧索引
                out_name = f"frame{i:04d}.jpg"
                if cv2.imwrite(os.path.join(output_dir, out_name), frame, image_quality):
                    frames_written += 1
                else:
                    print(f"[WARN] 保存第 {i} 帧到 {out_name} 失败")
            
            cap.release()
        
        if frames_written != frame_count:
            print(f"[WARN] 应提取 {frame_count} 帧，实际写入 {frames_written} 帧")
        else:
            print(f"[INFO] 成功提取并保存 {frames_written} 帧")
            
        return frames_written, need_rotate
        
    except Exception as e:
        print(f"[ERROR] 帧提取过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 确保资源被释放
        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
        except:
            pass
            
        return 0, False

