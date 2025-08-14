#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换工具
将COCO格式、Pascal VOC格式等转换为YOLO格式
"""

import json
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from tqdm import tqdm

class DatasetConverter:
    """数据集格式转换器"""
    
    def __init__(self):
        """初始化转换器"""
        self.supported_formats = ['coco', 'voc', 'yolo']
        print("🔄 数据集格式转换器已初始化")
    
    def convert_dataset(self, input_dir, output_dir, input_format, output_format='yolo'):
        """
        转换数据集格式
        
        Args:
            input_dir: 输入数据集目录
            output_dir: 输出目录
            input_format: 输入格式 ('coco', 'voc', 'yolo')
            output_format: 输出格式 (目前只支持'yolo')
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        
        if input_format not in self.supported_formats:
            raise ValueError(f"不支持的输入格式: {input_format}")
        
        print(f"🔄 开始转换: {input_format.upper()} -> {output_format.upper()}")
        print(f"📁 输入目录: {input_path}")
        print(f"📁 输出目录: {output_path}")
        
        # 创建输出目录结构
        self._create_yolo_structure(output_path)
        
        # 根据输入格式选择转换方法
        if input_format == 'coco':
            self._convert_from_coco(input_path, output_path)
        elif input_format == 'voc':
            self._convert_from_voc(input_path, output_path)
        elif input_format == 'yolo':
            self._copy_yolo_dataset(input_path, output_path)
        
        # 生成YOLO配置文件
        self._generate_yolo_config(output_path)
        
        print("✅ 数据集转换完成!")
    
    def _create_yolo_structure(self, output_path):
        """创建YOLO格式的目录结构"""
        directories = [
            output_path / "images" / "train",
            output_path / "images" / "val",
            output_path / "labels" / "train", 
            output_path / "labels" / "val"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 已创建YOLO目录结构")
    
    def _convert_from_coco(self, input_path, output_path):
        """从COCO格式转换"""
        print("🔄 正在从COCO格式转换...")
        
        # 查找COCO标注文件
        annotation_files = list(input_path.glob("**/*.json"))
        
        if not annotation_files:
            # 尝试查找标准COCO结构
            for split in ['train', 'val', 'test']:
                ann_file = input_path / 'annotations' / f'instances_{split}.json'
                if ann_file.exists():
                    annotation_files.append(ann_file)
        
        if not annotation_files:
            raise ValueError("未找到COCO格式的标注文件")
        
        total_converted = 0
        
        for ann_file in annotation_files:
            print(f"📄 处理标注文件: {ann_file.name}")
            
            # 确定分割类型
            if 'train' in ann_file.name.lower():
                split = 'train'
            elif 'val' in ann_file.name.lower():
                split = 'val'
            else:
                split = 'train'  # 默认为训练集
            
            converted_count = self._convert_coco_file(ann_file, input_path, output_path, split)
            total_converted += converted_count
        
        print(f"✅ COCO转换完成，共转换 {total_converted} 个样本")
    
    def _convert_coco_file(self, ann_file, input_path, output_path, split):
        """转换单个COCO标注文件"""
        with open(ann_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 创建图像ID到文件名的映射
        image_id_to_filename = {}
        image_id_to_size = {}
        
        for image_info in coco_data.get('images', []):
            image_id_to_filename[image_info['id']] = image_info['file_name']
            image_id_to_size[image_info['id']] = (image_info['width'], image_info['height'])
        
        # 按图像分组标注
        annotations_by_image = {}
        for annotation in coco_data.get('annotations', []):
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)
        
        converted_count = 0
        
        # 转换每个图像
        for image_id, annotations in tqdm(annotations_by_image.items(), desc=f"转换{split}集"):
            if image_id not in image_id_to_filename:
                continue
            
            filename = image_id_to_filename[image_id]
            img_width, img_height = image_id_to_size[image_id]
            
            # 查找图像文件
            image_path = self._find_image_file(input_path, filename)
            if not image_path:
                print(f"⚠️ 未找到图像文件: {filename}")
                continue
            
            # 复制图像文件
            target_img_path = output_path / "images" / split / filename
            shutil.copy2(image_path, target_img_path)
            
            # 转换标注
            yolo_annotations = []
            for annotation in annotations:
                yolo_line = self._convert_coco_annotation_to_yolo(annotation, img_width, img_height)
                if yolo_line:
                    yolo_annotations.append(yolo_line)
            
            # 保存YOLO标注文件
            if yolo_annotations:
                label_filename = Path(filename).stem + '.txt'
                label_path = output_path / "labels" / split / label_filename
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations) + '\n')
                
                converted_count += 1
        
        return converted_count
    
    def _convert_coco_annotation_to_yolo(self, annotation, img_width, img_height):
        """将单个COCO标注转换为YOLO格式"""
        if 'bbox' not in annotation:
            return None
        
        # COCO bbox格式: [x, y, width, height]
        x, y, w, h = annotation['bbox']
        
        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        # 类别ID (假设球杆类别为0)
        class_id = 0
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
    
    def _convert_from_voc(self, input_path, output_path):
        """从Pascal VOC格式转换"""
        print("🔄 正在从Pascal VOC格式转换...")
        
        # 查找XML标注文件
        xml_files = list(input_path.glob("**/*.xml"))
        
        if not xml_files:
            raise ValueError("未找到Pascal VOC格式的XML标注文件")
        
        converted_count = 0
        
        for xml_file in tqdm(xml_files, desc="转换VOC标注"):
            try:
                # 解析XML文件
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 获取图像信息
                filename = root.find('filename').text
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                
                # 查找对应的图像文件
                image_path = self._find_image_file(input_path, filename)
                if not image_path:
                    print(f"⚠️ 未找到图像文件: {filename}")
                    continue
                
                # 确定分割类型（简单规则）
                split = 'train' if converted_count % 5 != 0 else 'val'
                
                # 复制图像文件
                target_img_path = output_path / "images" / split / filename
                shutil.copy2(image_path, target_img_path)
                
                # 转换标注
                yolo_annotations = []
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # 转换为YOLO格式
                    x_center = (xmin + xmax) / 2 / img_width
                    y_center = (ymin + ymax) / 2 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    # 类别ID (假设球杆类别为0)
                    class_id = 0
                    
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # 保存YOLO标注文件
                if yolo_annotations:
                    label_filename = Path(filename).stem + '.txt'
                    label_path = output_path / "labels" / split / label_filename
                    
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations) + '\n')
                    
                    converted_count += 1
                    
            except Exception as e:
                print(f"⚠️ 转换XML文件失败 {xml_file.name}: {str(e)}")
                continue
        
        print(f"✅ VOC转换完成，共转换 {converted_count} 个样本")
    
    def _copy_yolo_dataset(self, input_path, output_path):
        """复制已有的YOLO格式数据集"""
        print("🔄 正在复制YOLO格式数据集...")
        
        # 查找YOLO格式的目录结构
        if (input_path / "images").exists() and (input_path / "labels").exists():
            # 标准YOLO结构
            for split in ['train', 'val']:
                img_src = input_path / "images" / split
                label_src = input_path / "labels" / split
                
                if img_src.exists():
                    img_dst = output_path / "images" / split
                    shutil.copytree(img_src, img_dst, dirs_exist_ok=True)
                
                if label_src.exists():
                    label_dst = output_path / "labels" / split
                    shutil.copytree(label_src, label_dst, dirs_exist_ok=True)
        else:
            # 扁平结构，需要重新组织
            self._reorganize_flat_yolo_dataset(input_path, output_path)
        
        print("✅ YOLO数据集复制完成")
    
    def _reorganize_flat_yolo_dataset(self, input_path, output_path):
        """重新组织扁平的YOLO数据集"""
        # 查找所有图像和标注文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(input_path.glob(f"**/*{ext}")))
            image_files.extend(list(input_path.glob(f"**/*{ext.upper()}")))
        
        label_files = list(input_path.glob("**/*.txt"))
        
        # 创建文件名到路径的映射
        label_dict = {f.stem: f for f in label_files}
        
        converted_count = 0
        
        for i, img_file in enumerate(image_files):
            # 确定分割类型 (80% 训练, 20% 验证)
            split = 'train' if i % 5 != 0 else 'val'
            
            # 复制图像文件
            target_img_path = output_path / "images" / split / img_file.name
            shutil.copy2(img_file, target_img_path)
            
            # 查找对应的标注文件
            if img_file.stem in label_dict:
                label_file = label_dict[img_file.stem]
                target_label_path = output_path / "labels" / split / f"{img_file.stem}.txt"
                shutil.copy2(label_file, target_label_path)
                converted_count += 1
        
        print(f"✅ 重新组织完成，共处理 {converted_count} 个样本")
    
    def _find_image_file(self, base_path, filename):
        """查找图像文件"""
        # 尝试不同的可能路径
        possible_paths = [
            base_path / filename,
            base_path / "images" / filename,
            base_path / "train" / filename,
            base_path / "val" / filename,
            base_path / "test" / filename
        ]
        
        # 递归搜索
        for path in base_path.rglob(filename):
            return path
        
        return None
    
    def _generate_yolo_config(self, output_path):
        """生成YOLO配置文件"""
        config = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['golf_club']
        }
        
        config_file = output_path / 'dataset.yaml'
        
        # 手动写入YAML格式（避免依赖yaml库）
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(f"# Golf Club Detection Dataset Configuration\n")
            f.write(f"path: {config['path']}\n")
            f.write(f"train: {config['train']}\n")
            f.write(f"val: {config['val']}\n")
            f.write(f"nc: {config['nc']}\n")
            f.write(f"names: {config['names']}\n")
        
        print(f"📝 YOLO配置文件已生成: {config_file}")
    
    def validate_yolo_dataset(self, dataset_path):
        """验证YOLO数据集的完整性"""
        dataset_path = Path(dataset_path)
        
        print("🔍 验证YOLO数据集...")
        
        issues = []
        
        # 检查目录结构
        required_dirs = [
            "images/train", "images/val",
            "labels/train", "labels/val"
        ]
        
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                issues.append(f"缺少目录: {dir_name}")
        
        # 检查配置文件
        config_file = dataset_path / "dataset.yaml"
        if not config_file.exists():
            issues.append("缺少配置文件: dataset.yaml")
        
        # 检查数据一致性
        for split in ['train', 'val']:
            img_dir = dataset_path / "images" / split
            label_dir = dataset_path / "labels" / split
            
            if img_dir.exists() and label_dir.exists():
                img_files = set(f.stem for f in img_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
                label_files = set(f.stem for f in label_dir.glob("*.txt"))
                
                missing_labels = img_files - label_files
                missing_images = label_files - img_files
                
                if missing_labels:
                    issues.append(f"{split}集中 {len(missing_labels)} 个图像缺少标注文件")
                
                if missing_images:
                    issues.append(f"{split}集中 {len(missing_images)} 个标注文件缺少对应图像")
        
        # 输出验证结果
        if issues:
            print("❌ 数据集验证发现问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ 数据集验证通过")
        
        return len(issues) == 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集格式转换工具")
    parser.add_argument("input_dir", help="输入数据集目录")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--input_format", "-i", choices=['coco', 'voc', 'yolo'], 
                       required=True, help="输入数据格式")
    parser.add_argument("--output_format", "-o", choices=['yolo'], 
                       default='yolo', help="输出数据格式")
    parser.add_argument("--validate", action="store_true", help="转换后验证数据集")
    
    args = parser.parse_args()
    
    try:
        # 创建转换器
        converter = DatasetConverter()
        
        # 执行转换
        converter.convert_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            input_format=args.input_format,
            output_format=args.output_format
        )
        
        # 验证数据集
        if args.validate:
            converter.validate_yolo_dataset(args.output_dir)
        
    except Exception as e:
        print(f"❌ 转换过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 