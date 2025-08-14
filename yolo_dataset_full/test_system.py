#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化训练系统测试脚本
验证各个组件是否正常工作
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """测试所有模块是否能正常导入"""
    print("🔍 测试模块导入...")
    
    try:
        from auto_training_system import AutoTrainingSystem
        print("✅ AutoTrainingSystem 导入成功")
    except Exception as e:
        print(f"❌ AutoTrainingSystem 导入失败: {e}")
        return False
    
    try:
        from interactive_annotation import InteractiveAnnotator
        print("✅ InteractiveAnnotator 导入成功")
    except Exception as e:
        print(f"❌ InteractiveAnnotator 导入失败: {e}")
        return False
    
    try:
        from convert_to_yolo import DatasetConverter
        print("✅ DatasetConverter 导入成功")
    except Exception as e:
        print(f"❌ DatasetConverter 导入失败: {e}")
        return False
    
    return True

def test_directory_creation():
    """测试目录创建功能"""
    print("\n🔍 测试目录创建...")
    
    try:
        # 创建临时目录进行测试
        with tempfile.TemporaryDirectory() as temp_dir:
            from auto_training_system import AutoTrainingSystem
            
            # 初始化系统
            system = AutoTrainingSystem(temp_dir)
            
            # 检查目录是否创建（不包括input目录，因为使用固定路径）
            required_dirs = [
                "processed/images/train",
                "processed/images/val",
                "processed/labels/train",
                "processed/labels/val",
                "archive",
                "models/latest",
                "configs",
                "logs",
                "temp"
            ]
            
            for dir_path in required_dirs:
                full_path = Path(temp_dir) / dir_path
                if not full_path.exists():
                    print(f"❌ 目录创建失败: {dir_path}")
                    return False
            
            print("✅ 所有必要目录创建成功")
            print(f"📁 固定输入目录: {system.input_dir}")
            return True
            
    except Exception as e:
        print(f"❌ 目录创建测试失败: {e}")
        return False

def test_data_converter():
    """测试数据转换器"""
    print("\n🔍 测试数据转换器...")
    
    try:
        from convert_to_yolo import DatasetConverter
        
        converter = DatasetConverter()
        
        # 检查支持的格式
        expected_formats = ['coco', 'voc', 'yolo']
        if converter.supported_formats != expected_formats:
            print(f"❌ 支持格式不匹配: {converter.supported_formats}")
            return False
        
        print("✅ 数据转换器初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ 数据转换器测试失败: {e}")
        return False

def test_annotation_tool():
    """测试标注工具"""
    print("\n🔍 测试标注工具...")
    
    try:
        from interactive_annotation import InteractiveAnnotator, ANNOTATION_MODES
        
        # 测试不同模式的初始化
        for mode in ANNOTATION_MODES.keys():
            annotator = InteractiveAnnotator(mode=mode, output_format='yolo')
            if annotator.mode != mode:
                print(f"❌ 标注模式设置失败: {mode}")
                return False
        
        print("✅ 标注工具初始化成功")
        print(f"📁 固定输入目录: {annotator.input_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 标注工具测试失败: {e}")
        return False

def test_dependencies():
    """测试依赖库"""
    print("\n🔍 测试依赖库...")
    
    required_packages = [
        'cv2',
        'numpy',
        'pathlib',
        'json',
        'yaml',
        'torch',
        'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'pathlib':
                from pathlib import Path
            elif package == 'json':
                import json
            elif package == 'yaml':
                import yaml
            elif package == 'torch':
                import torch
            elif package == 'ultralytics':
                from ultralytics import YOLO
            
            print(f"✅ {package} 可用")
            
        except ImportError:
            print(f"❌ {package} 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺失的依赖包: {missing_packages}")
        print("请安装缺失的包:")
        for package in missing_packages:
            if package == 'cv2':
                print("  pip install opencv-python")
            elif package == 'ultralytics':
                print("  pip install ultralytics")
            elif package == 'torch':
                print("  pip install torch")
            else:
                print(f"  pip install {package}")
        return False
    
    return True

def test_system_integration():
    """测试系统集成"""
    print("\n🔍 测试系统集成...")
    
    try:
        # 创建临时目录进行测试
        with tempfile.TemporaryDirectory() as temp_dir:
            from auto_training_system import AutoTrainingSystem
            
            # 初始化系统
            system = AutoTrainingSystem(temp_dir)
            
            # 测试检查新数据功能（固定目录可能存在或不存在）
            has_new_data = system._check_for_new_data()
            
            # 测试现有数据检查功能（应该返回False，因为没有处理过的数据）
            has_existing_data = system._has_existing_data()
            if has_existing_data:
                print("❌ 现有数据检查功能异常")
                return False
            
            print("✅ 系统集成测试通过")
            print(f"📁 固定输入目录: {system.input_dir}")
            print(f"📊 检测到新数据: {has_new_data}")
            return True
            
    except Exception as e:
        print(f"❌ 系统集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 自动化训练系统测试")
    print("=" * 50)
    
    tests = [
        ("模块导入", test_imports),
        ("依赖库", test_dependencies),
        ("目录创建", test_directory_creation),
        ("数据转换器", test_data_converter),
        ("标注工具", test_annotation_tool),
        ("系统集成", test_system_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
    
    print("\n" + "=" * 50)
    print(f"🧪 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用")
        print("\n📖 使用指南:")
        print("1. 将图像文件放入固定目录: C:\\Users\\Administrator\\Desktop\\AIGolf\\videos\\")
        print("2. 将标注文件放入同一目录或annotations子目录 (可选)")
        print("3. 运行: python train_yolo_auto.py")
        print("4. 如需交互式标注: python interactive_annotation.py --mode rotated_bbox")
    else:
        print("⚠️ 部分测试失败，请检查错误信息并修复问题")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 