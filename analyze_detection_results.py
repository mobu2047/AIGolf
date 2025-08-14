#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测结果分析脚本
分析模型检测性能并生成报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def analyze_detection_results(results_file):
    """分析检测结果"""
    
    # 读取结果文件
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("高尔夫球杆检测结果分析报告")
    print("=" * 60)
    
    # 基本信息
    video_info = data['video_info']
    detection_summary = data['detection_summary']
    frame_results = data['frame_results']
    
    print(f"\n📹 视频信息:")
    print(f"  文件路径: {data['video_path']}")
    print(f"  分辨率: {video_info['width']}x{video_info['height']}")
    print(f"  帧率: {video_info['fps']} FPS")
    print(f"  总帧数: {video_info['total_frames']}")
    
    print(f"\n🎯 检测概览:")
    print(f"  总检测数: {detection_summary['total_detections']}")
    print(f"  有检测的帧数: {detection_summary['frames_with_detections']}/{video_info['total_frames']}")
    print(f"  检测率: {detection_summary['detection_rate']*100:.1f}%")
    
    # 详细分析
    detections_per_frame = [result['detections'] for result in frame_results]
    confidences_all = []
    
    for result in frame_results:
        confidences_all.extend(result['confidences'])
    
    print(f"\n📊 检测统计:")
    print(f"  平均每帧检测数: {np.mean(detections_per_frame):.2f}")
    print(f"  最大单帧检测数: {max(detections_per_frame)}")
    print(f"  最小单帧检测数: {min(detections_per_frame)}")
    
    if confidences_all:
        print(f"\n🎲 置信度分析:")
        print(f"  平均置信度: {np.mean(confidences_all):.3f}")
        print(f"  最高置信度: {max(confidences_all):.3f}")
        print(f"  最低置信度: {min(confidences_all):.3f}")
        print(f"  置信度标准差: {np.std(confidences_all):.3f}")
        
        # 置信度分布
        confidence_ranges = [
            (0.05, 0.1, "极低"),
            (0.1, 0.2, "低"),
            (0.2, 0.3, "中低"),
            (0.3, 0.5, "中等"),
            (0.5, 0.7, "高"),
            (0.7, 1.0, "极高")
        ]
        
        print(f"\n📈 置信度分布:")
        for min_conf, max_conf, label in confidence_ranges:
            count = sum(1 for c in confidences_all if min_conf <= c < max_conf)
            percentage = count / len(confidences_all) * 100
            print(f"  {label} ({min_conf:.1f}-{max_conf:.1f}): {count} ({percentage:.1f}%)")
    
    # 时间序列分析
    print(f"\n⏱️ 时间序列分析:")
    
    # 检测连续性
    consecutive_detections = 0
    max_consecutive = 0
    current_consecutive = 0
    
    for result in frame_results:
        if result['detections'] > 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    print(f"  最长连续检测: {max_consecutive} 帧")
    
    # 检测稳定性（相邻帧检测数变化）
    detection_changes = []
    for i in range(1, len(detections_per_frame)):
        change = abs(detections_per_frame[i] - detections_per_frame[i-1])
        detection_changes.append(change)
    
    if detection_changes:
        print(f"  检测稳定性 (平均变化): {np.mean(detection_changes):.2f}")
        print(f"  最大帧间变化: {max(detection_changes)}")
    
    # 性能评估
    print(f"\n🏆 性能评估:")
    
    # 根据检测率评估
    detection_rate = detection_summary['detection_rate']
    if detection_rate >= 0.9:
        performance_level = "优秀"
        performance_color = "🟢"
    elif detection_rate >= 0.7:
        performance_level = "良好"
        performance_color = "🟡"
    elif detection_rate >= 0.5:
        performance_level = "一般"
        performance_color = "🟠"
    else:
        performance_level = "需要改进"
        performance_color = "🔴"
    
    print(f"  总体性能: {performance_color} {performance_level}")
    
    # 根据置信度评估
    if confidences_all:
        avg_confidence = np.mean(confidences_all)
        if avg_confidence >= 0.5:
            confidence_level = "高置信度"
            confidence_color = "🟢"
        elif avg_confidence >= 0.2:
            confidence_level = "中等置信度"
            confidence_color = "🟡"
        else:
            confidence_level = "低置信度"
            confidence_color = "🟠"
        
        print(f"  置信度水平: {confidence_color} {confidence_level}")
    
    # 建议
    print(f"\n💡 改进建议:")
    
    if detection_rate < 0.8:
        print("  • 检测率偏低，建议:")
        print("    - 增加更多训练数据")
        print("    - 调整模型架构")
        print("    - 优化数据增强策略")
    
    if confidences_all and np.mean(confidences_all) < 0.3:
        print("  • 置信度偏低，建议:")
        print("    - 延长训练时间")
        print("    - 调整学习率")
        print("    - 检查标注质量")
    
    if max(detection_changes) > 3:
        print("  • 检测不稳定，建议:")
        print("    - 添加时序一致性约束")
        print("    - 使用跟踪算法平滑结果")
        print("    - 增加视频数据训练")
    
    print(f"\n✅ 分析完成！")
    
    return {
        'detection_rate': detection_rate,
        'avg_confidence': np.mean(confidences_all) if confidences_all else 0,
        'total_detections': detection_summary['total_detections'],
        'performance_level': performance_level
    }

def plot_detection_timeline(results_file, output_path=None):
    """绘制检测时间线图"""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    frame_results = data['frame_results']
    frames = [result['frame'] for result in frame_results]
    detections = [result['detections'] for result in frame_results]
    
    # 计算平均置信度
    avg_confidences = []
    for result in frame_results:
        if result['confidences']:
            avg_confidences.append(np.mean(result['confidences']))
        else:
            avg_confidences.append(0)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 检测数量时间线
    ax1.plot(frames, detections, 'b-', linewidth=2, label='检测数量')
    ax1.fill_between(frames, detections, alpha=0.3)
    ax1.set_xlabel('帧数')
    ax1.set_ylabel('检测数量')
    ax1.set_title('每帧检测数量时间线')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 平均置信度时间线
    ax2.plot(frames, avg_confidences, 'r-', linewidth=2, label='平均置信度')
    ax2.fill_between(frames, avg_confidences, alpha=0.3, color='red')
    ax2.set_xlabel('帧数')
    ax2.set_ylabel('平均置信度')
    ax2.set_title('每帧平均置信度时间线')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"时间线图已保存到: {output_path}")
    else:
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="检测结果分析")
    parser.add_argument("--results", type=str, required=True, help="检测结果JSON文件路径")
    parser.add_argument("--plot", action="store_true", help="生成时间线图表")
    parser.add_argument("--output", type=str, help="图表输出路径")
    
    args = parser.parse_args()
    
    if not Path(args.results).exists():
        print(f"结果文件不存在: {args.results}")
        return
    
    # 分析结果
    analysis = analyze_detection_results(args.results)
    
    # 生成图表
    if args.plot:
        plot_detection_timeline(args.results, args.output)

if __name__ == "__main__":
    main() 