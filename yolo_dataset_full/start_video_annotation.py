#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频标注系统启动脚本
简化的启动接口，用户只需运行此脚本即可开始视频标注
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """主函数"""
    print("🎯 启动高尔夫球杆检测视频标注系统")
    print("=" * 60)
    
    try:
        from video_annotation_system import VideoAnnotationSystem
        
        # 使用推荐的默认参数
        system = VideoAnnotationSystem(
            mode='rotated_bbox',      # 旋转边界框模式（推荐）
            frame_interval=10,        # 每10帧提取一帧
            max_frames_per_video=50   # 每个视频最多50帧
        )
        
        print("\n📋 系统配置:")
        print("   - 标注模式: 旋转边界框")
        print("   - 帧间隔: 10帧")
        print("   - 最大帧数: 50帧/视频")
        print("   - 视频输入: C:\\Users\\Administrator\\Desktop\\AIGolf\\videos")
        print("   - 数据输出: C:\\Users\\Administrator\\Desktop\\AIGolf\\dataset")
        
        print("\n🎮 操作说明:")
        print("   - 用鼠标左键点击球杆的两个端点")
        print("   - 按ESC键跳过当前帧")
        print("   - 标注完成后自动进入下一帧")
        
        input("\n按回车键开始处理视频...")
        
        # 开始处理视频
        system.process_videos()
        
        print("\n🎉 视频标注完成！")
        print("📁 标注数据已保存到: C:\\Users\\Administrator\\Desktop\\AIGolf\\dataset")
        print("🚀 现在可以运行训练脚本: python train_yolo_auto.py")
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保所有依赖已安装：")
        print("  pip install opencv-python numpy")
    except Exception as e:
        print(f"❌ 运行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main() 