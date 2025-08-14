#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化增强的关键点数据
展示人体关键点和球杆检测结果
"""

import cv2
import numpy as np
import torch
from golf_club_detector import add_golf_club_to_keypoints, get_golf_club_keypoints, KEYPOINT_CONFIG
from pathlib import Path
import argparse

def draw_pose_keypoints(frame, pose_keypoints, color=(0, 255, 0), radius=3):
    """
    绘制人体关键点
    
    Args:
        frame: 图像帧
        pose_keypoints: 人体关键点 (33, 3)
        color: 绘制颜色
        radius: 点的半径
    """
    height, width = frame.shape[:2]
    
    for i, (x, y, visibility) in enumerate(pose_keypoints):
        if visibility > 0.5:  # 只绘制可见的关键点
            px = int(x * width)
            py = int(y * height)
            cv2.circle(frame, (px, py), radius, color, -1)
            # 添加关键点编号
            cv2.putText(frame, str(i), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def draw_golf_club_keypoints(frame, golf_club_keypoints, color=(0, 0, 255), radius=5):
    """
    绘制球杆关键点
    
    Args:
        frame: 图像帧
        golf_club_keypoints: 球杆关键点 (2, 3)
        color: 绘制颜色
        radius: 点的半径
    """
    height, width = frame.shape[:2]
    
    grip_point = golf_club_keypoints[0]  # 握把端
    head_point = golf_club_keypoints[1]  # 球杆头端
    
    # 绘制握把端
    if grip_point[2] > 0:  # 可见
        px = int(grip_point[0] * width)
        py = int(grip_point[1] * height)
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.putText(frame, "Grip", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 绘制球杆头端
    if head_point[2] > 0:  # 可见
        px = int(head_point[0] * width)
        py = int(head_point[1] * height)
        cv2.circle(frame, (px, py), radius, color, -1)
        cv2.putText(frame, "Head", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 如果两个点都可见，绘制连线
    if grip_point[2] > 0 and head_point[2] > 0:
        grip_px = int(grip_point[0] * width)
        grip_py = int(grip_point[1] * height)
        head_px = int(head_point[0] * width)
        head_py = int(head_point[1] * height)
        cv2.line(frame, (grip_px, grip_py), (head_px, head_py), color, 3)

def visualize_enhanced_keypoints(video_path, output_path=None, show_pose=True, show_golf_club=True):
    """
    可视化增强的关键点数据
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径（可选）
        show_pose: 是否显示人体关键点
        show_golf_club: 是否显示球杆关键点
    """
    
    print(f"开始可视化增强关键点: {video_path}")
    
    # 1. 模拟提取人体关键点（这里用随机数据代替）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
    
    # 2. 生成模拟的人体关键点数据
    print("生成模拟人体关键点数据...")
    pose_keypoints = np.random.rand(total_frames, 33, 3)
    pose_keypoints[:, :, 2] = 0.8  # 设置可见性
    
    # 3. 添加球杆检测信息
    print("添加球杆检测信息...")
    enhanced_keypoints = add_golf_club_to_keypoints(
        existing_keypoints=pose_keypoints,
        video_path=video_path,
        confidence=0.2
    )
    
    # 4. 提取球杆关键点
    golf_club_keypoints = get_golf_club_keypoints(enhanced_keypoints)
    
    # 5. 设置输出视频
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("开始可视化...")
    frame_count = 0
    
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开始
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 创建可视化帧
            vis_frame = frame.copy()
            
            # 绘制人体关键点
            if show_pose and frame_count < len(enhanced_keypoints):
                pose_kpts = enhanced_keypoints[frame_count, :33, :]
                draw_pose_keypoints(vis_frame, pose_kpts, color=(0, 255, 0), radius=2)
            
            # 绘制球杆关键点
            if show_golf_club and frame_count < len(golf_club_keypoints):
                golf_kpts = golf_club_keypoints[frame_count]
                draw_golf_club_keypoints(vis_frame, golf_kpts, color=(0, 0, 255), radius=6)
            
            # 添加信息文本
            info_text = f"Frame: {frame_count+1}/{total_frames}"
            cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加图例
            if show_pose:
                cv2.putText(vis_frame, "Green: Pose Keypoints", (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if show_golf_club:
                cv2.putText(vis_frame, "Red: Golf Club", (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 保存到输出视频
            if output_path:
                out.write(vis_frame)
            
            # 显示结果
            cv2.imshow('Enhanced Keypoints Visualization', vis_frame)
            
            # 按'q'退出，按空格暂停
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)  # 暂停直到按任意键
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"处理进度: {frame_count}/{total_frames}")
    
    finally:
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    print(f"可视化完成! 处理了 {frame_count} 帧")
    if output_path:
        print(f"输出视频保存到: {output_path}")

def create_keypoints_comparison_image(video_path, frame_idx=50):
    """
    创建关键点对比图像
    
    Args:
        video_path: 视频路径
        frame_idx: 要展示的帧索引
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"无法读取第{frame_idx}帧")
        return
    
    # 生成模拟关键点数据
    total_frames = frame_idx + 1
    pose_keypoints = np.random.rand(total_frames, 33, 3)
    pose_keypoints[:, :, 2] = 0.8
    
    # 添加球杆检测
    enhanced_keypoints = add_golf_club_to_keypoints(
        existing_keypoints=pose_keypoints,
        video_path=video_path,
        confidence=0.2
    )
    
    golf_club_keypoints = get_golf_club_keypoints(enhanced_keypoints)
    
    # 创建对比图像
    height, width = frame.shape[:2]
    comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # 左侧：原始帧 + 人体关键点
    left_frame = frame.copy()
    pose_kpts = enhanced_keypoints[frame_idx, :33, :]
    draw_pose_keypoints(left_frame, pose_kpts, color=(0, 255, 0), radius=3)
    comparison[:, :width] = left_frame
    
    # 右侧：原始帧 + 球杆关键点
    right_frame = frame.copy()
    golf_kpts = golf_club_keypoints[frame_idx]
    draw_golf_club_keypoints(right_frame, golf_kpts, color=(0, 0, 255), radius=6)
    comparison[:, width:] = right_frame
    
    # 添加标题
    cv2.putText(comparison, "Pose Keypoints", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Golf Club Detection", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 保存对比图像
    output_path = f"keypoints_comparison_frame_{frame_idx}.jpg"
    cv2.imwrite(output_path, comparison)
    print(f"对比图像保存到: {output_path}")
    
    # 显示图像
    cv2.imshow('Keypoints Comparison', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="可视化增强的关键点数据")
    parser.add_argument("--video", type=str, default="test/labeled/1.mp4", help="输入视频路径")
    parser.add_argument("--output", type=str,default="test/labeled/1_output.mp4", help="输出视频路径")
    parser.add_argument("--frame", type=int, help="创建单帧对比图像")
    parser.add_argument("--no-pose", action="store_true", help="不显示人体关键点")
    parser.add_argument("--no-golf", action="store_true", help="不显示球杆关键点")
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"视频文件不存在: {args.video}")
        return
    
    if args.frame is not None:
        # 创建单帧对比图像
        create_keypoints_comparison_image(args.video, args.frame)
    else:
        # 可视化整个视频
        visualize_enhanced_keypoints(
            video_path=args.video,
            output_path=args.output,
            show_pose=not args.no_pose,
            show_golf_club=not args.no_golf
        )

if __name__ == "__main__":
    main() 