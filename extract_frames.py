#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频帧提取工具
用于从视频文件中提取帧，以便后续叠加辅助线
"""

import os
import cv2
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, start_frame=None, end_frame=None, 
                   step=1, prefix="frame", rotate=False, resize=None):
    """
    从视频文件中提取帧
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录
        start_frame: 起始帧索引（从0开始）
        end_frame: 结束帧索引
        step: 帧间隔，1表示每帧都提取，2表示每隔一帧提取一帧
        prefix: 输出文件名前缀
        rotate: 是否旋转视频(90度顺时针)
        resize: 调整大小，格式为(width, height)，例如(1280, 720)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 处理起始和结束帧
    start_frame = 0 if start_frame is None else max(0, start_frame)
    end_frame = total_frames - 1 if end_frame is None else min(total_frames - 1, end_frame)
    
    if start_frame >= total_frames or end_frame < start_frame:
        print(f"[ERROR] 无效的帧范围: {start_frame}-{end_frame}, 视频总帧数: {total_frames}")
        cap.release()
        return
    
    # 设置起始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"[INFO] 视频信息: {width}x{height}, {fps:.2f}FPS, 总帧数: {total_frames}")
    print(f"[INFO] 提取帧范围: {start_frame}-{end_frame}, 步长: {step}")
    
    auto_rotate = False
    if rotate == "auto":
        # 自动检测是否需要旋转（横屏视频）
        auto_rotate = width > height
        print(f"[INFO] 自动检测旋转: {'是' if auto_rotate else '否'} (尺寸: {width}x{height})")
        rotate = auto_rotate
    
    # 提取帧
    frame_index = start_frame
    save_count = 0
    
    with tqdm(total=(end_frame - start_frame + 1) // step) as pbar:
        while frame_index <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 是否需要旋转
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # 是否需要调整大小
            if resize:
                frame = cv2.resize(frame, resize)
            
            # 保存帧
            output_path = os.path.join(output_dir, f"{prefix}_{save_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            save_count += 1
            
            # 更新进度条
            pbar.update(1)
            
            # 跳过帧
            for _ in range(step - 1):
                if frame_index >= end_frame:
                    break
                ret = cap.grab()
                if not ret:
                    break
                frame_index += 1
            
            frame_index += 1
    
    cap.release()
    print(f"[INFO] 已提取 {save_count} 帧到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="视频帧提取工具")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--output", type=str, default="frames", help="输出目录")
    parser.add_argument("--start", type=int, help="起始帧索引（从0开始）")
    parser.add_argument("--end", type=int, help="结束帧索引")
    parser.add_argument("--step", type=int, default=1, help="帧间隔，默认为1（每帧都提取）")
    parser.add_argument("--prefix", type=str, default="frame", help="输出文件名前缀")
    parser.add_argument("--rotate", type=str, default="auto", 
                      choices=["auto", "yes", "no"], help="是否旋转视频，auto-自动检测，yes-旋转，no-不旋转")
    parser.add_argument("--resize", type=str, help="调整大小，格式为widthxheight，例如1280x720")
    parser.add_argument("--session", type=str, help="会话ID，如果指定，则输出到对应的frames目录")
    
    args = parser.parse_args()
    
    # 处理旋转选项
    rotate_option = args.rotate.lower()
    rotate = False
    if rotate_option == "yes":
        rotate = True
    elif rotate_option == "auto":
        rotate = "auto"
    
    # 处理调整大小选项
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split("x"))
            resize = (width, height)
            print(f"[INFO] 将调整尺寸为: {width}x{height}")
        except:
            print(f"[WARN] 无效的调整大小参数: {args.resize}，请使用widthxheight格式，例如1280x720")
    
    # 处理输出目录
    output_dir = args.output
    if args.session:
        output_dir = os.path.join("resultData", args.session, "frames")
    
    # 提取帧
    extract_frames(
        video_path=args.video,
        output_dir=output_dir,
        start_frame=args.start,
        end_frame=args.end,
        step=args.step,
        prefix=args.prefix,
        rotate=rotate,
        resize=resize
    )

if __name__ == "__main__":
    main() 