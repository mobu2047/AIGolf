#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO高尔夫球杆检测自动化训练主脚本
用户只需运行此脚本即可完成全自动训练

使用方法:
1. 将图像文件放入 yolo_dataset_full/input/images/
2. 将标注文件放入 yolo_dataset_full/input/annotations/ (可选)
3. 运行: python train_yolo_auto.py
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from auto_training_system import AutoTrainingSystem

def main():
    """主函数 - 用户只需运行这个"""
    print(" 高尔夫球杆检测自动化训练系统")
    print("=" * 50)
    
    try:
        # 创建自动训练系统
        auto_trainer = AutoTrainingSystem()
        
        # 运行自动训练
        auto_trainer.run()
        
        print("\n" + "=" * 50)
        print(" 自动化训练流程完成!")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n 用户中断了训练过程")
        
    except Exception as e:
        print(f"\n 训练过程中出现错误: {str(e)}")
        print(" 详细错误信息已保存到日志文件")
        
        # 显示简单的故障排除建议
        print("\n 故障排除建议:")
        print("1. 检查input目录中是否有有效的图像文件")
        print("2. 确保标注文件格式正确")
        print("3. 检查磁盘空间是否充足")
        print("4. 查看logs目录中的详细错误日志")

if __name__ == "__main__":
    main() 