#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO高尔夫球杆检测自动化训练系统
用户只需放入数据即可自动完成训练的完整流程
"""

import os
import json
import yaml
import shutil
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class AutoTrainingSystem:
    """自动化训练系统核心类"""
    
    def __init__(self, base_dir="yolo_dataset_full"):
        """
        初始化自动化训练系统
        
        Args:
            base_dir: 基础目录路径
        """
        self.base_dir = Path(base_dir)
        # 训练系统的输入路径为视频标注系统的输出路径
        self.input_dir = Path(r"C:\Users\Administrator\Desktop\AIGolf\dataset")
        self.processed_dir = self.base_dir / "processed"
        self.archive_dir = self.base_dir / "archive"
        self.models_dir = self.base_dir / "models"
        self.configs_dir = self.base_dir / "configs"
        self.logs_dir = self.base_dir / "logs"
        self.temp_dir = self.base_dir / "temp"
        # 已训练数据存储目录
        self.trained_data_dir = self.base_dir / "trained_data"
        
        # 确保所有目录存在
        self._ensure_directories()
        
        # 初始化日志
        self.processing_log = []
        
        print(f"🚀 自动化训练系统已初始化")
        print(f"📁 基础目录: {self.base_dir.absolute()}")
        print(f"📁 训练输入目录: {self.input_dir.absolute()} (视频标注输出)")
        print(f"📁 已训练数据目录: {self.trained_data_dir.absolute()}")
    
    def _ensure_directories(self):
        """确保所有必要目录存在"""
        directories = [
            # 不再创建input目录，因为使用固定路径
            self.processed_dir / "images" / "train",
            self.processed_dir / "images" / "val",
            self.processed_dir / "labels" / "train",
            self.processed_dir / "labels" / "val",
            self.archive_dir,
            self.models_dir / "latest",
            self.configs_dir,
            self.logs_dir,
            self.temp_dir,
            # 已训练数据目录
            self.trained_data_dir / "images",
            self.trained_data_dir / "annotations",
            self.trained_data_dir / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 确保固定输入目录存在，如果不存在则创建
        if not self.input_dir.exists():
            print(f"📁 创建训练输入目录: {self.input_dir}")
            self.input_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """
        主运行函数 - 用户只需调用这一个函数
        自动检测新数据并完成训练流程
        """
        print("\n" + "="*60)
        print("🚀 启动自动化训练系统...")
        print("="*60)
        
        try:
            # 1. 检查输入目录是否有新数据
            new_data_found = self._check_for_new_data()
            
            if new_data_found:
                print("📁 发现新数据，开始自动处理...")
                
                # 2. 处理新数据
                batch_id = self._process_new_data()
                
                # 3. 自动训练
                self._auto_train(batch_id)
                
                # 4. 清理输入目录
                self._cleanup_input_directory()
                
                print("✅ 训练完成！新数据已自动整合到训练集中")
            else:
                print("📂 输入目录中未发现新数据")
                
                # 检查是否有现有数据可以训练
                if self._has_existing_data():
                    print("🔄 使用现有数据进行训练...")
                    self._auto_train()
                else:
                    print("❌ 没有可用的训练数据")
                    self._create_input_directory_guide()
                    
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {str(e)}")
            self._handle_error(e)
    
    def _check_for_new_data(self):
        """检查输入目录是否有新数据"""
        if not self.input_dir.exists():
            print(f"⚠️ 训练输入目录不存在: {self.input_dir}")
            return False
        
        # 直接在dataset目录中检查图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(list(self.input_dir.glob(f"*{ext}")))
            images.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
        
        # 也检查images子目录（视频标注系统的输出结构）
        images_subdir = self.input_dir / "images"
        if images_subdir.exists():
            for ext in image_extensions:
                images.extend(list(images_subdir.glob(f"*{ext}")))
                images.extend(list(images_subdir.glob(f"*{ext.upper()}")))
        
        # 检查标注文件（可选）
        annotation_extensions = ['.txt', '.json', '.xml']
        annotations = []
        
        # 在dataset目录中查找标注文件
        for ext in annotation_extensions:
            annotations.extend(list(self.input_dir.glob(f"*{ext}")))
        
        # 也检查annotations子目录（视频标注系统的输出结构）
        annotations_subdir = self.input_dir / "annotations"
        if annotations_subdir.exists():
            for ext in annotation_extensions:
                annotations.extend(list(annotations_subdir.glob(f"*{ext}")))
        
        print(f"📊 发现 {len(images)} 个图像文件")
        print(f"📊 发现 {len(annotations)} 个标注文件")
        
        return len(images) > 0
    
    def _process_new_data(self):
        """自动处理新数据"""
        # 生成批次ID
        batch_id = self._generate_batch_id()
        print(f"📦 处理批次: {batch_id}")
        
        try:
            # 1. 验证新数据
            self._log_step("验证输入数据")
            self._validate_input_data()
            
            # 2. 转换数据格式
            self._log_step("转换数据格式")
            converted_data = self._convert_data_format()
            
            # 3. 去重检查
            self._log_step("检查重复数据")
            unique_data = self._remove_duplicates(converted_data)
            
            # 4. 合并到现有数据集
            self._log_step("合并到训练数据集")
            self._merge_to_processed_dataset(unique_data)
            
            # 5. 备份新数据到archive
            self._log_step("备份数据到归档目录")
            self._archive_batch(batch_id, unique_data)
            
            # 6. 重新分割数据集
            self._log_step("重新分割训练/验证集")
            self._resplit_dataset()
            
            # 7. 更新配置文件
            self._log_step("更新配置文件")
            self._update_configs()
            
            print(f"✅ 批次 {batch_id} 处理完成，共处理 {len(unique_data)} 个样本")
            return batch_id
            
        except Exception as e:
            print(f"❌ 处理批次 {batch_id} 时出错: {str(e)}")
            raise
    
    def _generate_batch_id(self):
        """生成批次ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}"
    
    def _validate_input_data(self):
        """验证输入数据的有效性"""
        if not self.input_dir.exists():
            raise ValueError(f"训练输入目录不存在: {self.input_dir}")
        
        # 检查图像文件是否可读
        valid_images = 0
        for img_path in self.input_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        valid_images += 1
                    else:
                        print(f"⚠️ 无法读取图像: {img_path.name}")
                except Exception as e:
                    print(f"⚠️ 图像文件损坏: {img_path.name} - {str(e)}")
        
        if valid_images == 0:
            raise ValueError("没有有效的图像文件")
        
        print(f"✅ 验证通过，发现 {valid_images} 个有效图像")
    
    def _convert_data_format(self):
        """自动检测并转换数据格式为YOLO格式"""
        converted_data = []
        
        # 获取所有图像文件（支持视频标注系统的输出结构）
        image_files = []
        
        # 直接在dataset目录中查找
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(self.input_dir.glob(f"*{ext}")))
            image_files.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
        
        # 也在images子目录中查找（视频标注系统的输出结构）
        images_subdir = self.input_dir / "images"
        if images_subdir.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(images_subdir.glob(f"*{ext}")))
                image_files.extend(list(images_subdir.glob(f"*{ext.upper()}")))
        
        print(f"🔄 开始转换 {len(image_files)} 个图像的标注...")
        
        for img_path in image_files:
            # 查找对应的标注文件
            annotation_path = self._find_annotation_file(img_path)
            
            if annotation_path:
                try:
                    # 根据文件扩展名转换格式
                    if annotation_path.suffix.lower() == '.json':
                        # COCO格式转换
                        yolo_annotation = self._convert_coco_to_yolo(img_path, annotation_path)
                    elif annotation_path.suffix.lower() == '.txt':
                        # 已经是YOLO格式，直接复制
                        yolo_annotation = annotation_path
                    elif annotation_path.suffix.lower() == '.xml':
                        # Pascal VOC格式转换
                        yolo_annotation = self._convert_voc_to_yolo(img_path, annotation_path)
                    else:
                        print(f"⚠️ 不支持的标注格式: {annotation_path}")
                        continue
                    
                    converted_data.append({
                        'image': img_path,
                        'annotation': yolo_annotation,
                        'image_id': img_path.stem,
                        'source_annotation': annotation_path
                    })
                    
                except Exception as e:
                    print(f"⚠️ 转换标注失败 {annotation_path.name}: {str(e)}")
                    continue
            else:
                # 没有标注文件，需要交互式标注
                print(f"⚠️ 图像 {img_path.name} 没有对应的标注文件，将跳过")
                # 可以在这里调用交互式标注
                continue
        
        print(f"✅ 成功转换 {len(converted_data)} 个样本")
        return converted_data
    
    def _find_annotation_file(self, img_path):
        """查找图像对应的标注文件"""
        base_name = img_path.stem
        
        # 尝试不同的标注文件扩展名和位置
        search_locations = [
            self.input_dir,  # 直接在dataset目录中
            self.input_dir / "annotations",  # 在annotations子目录中（视频标注系统输出）
            img_path.parent  # 在图像文件同目录中
        ]
        
        for location in search_locations:
            if not location.exists():
                continue
                
            for ext in ['.txt', '.json', '.xml']:
                annotation_path = location / f"{base_name}{ext}"
                if annotation_path.exists():
                    return annotation_path
        
        return None
    
    def _convert_coco_to_yolo(self, img_path, coco_path):
        """将COCO格式转换为YOLO格式"""
        # 读取图像尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        img_height, img_width = img.shape[:2]
        
        # 读取COCO标注
        with open(coco_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 创建YOLO格式标注文件
        yolo_path = self.temp_dir / f"{img_path.stem}.txt"
        
        with open(yolo_path, 'w') as f:
            # 处理COCO标注中的每个对象
            if 'annotations' in coco_data:
                for annotation in coco_data['annotations']:
                    if 'bbox' in annotation:
                        # COCO bbox格式: [x, y, width, height]
                        x, y, w, h = annotation['bbox']
                        
                        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        norm_width = w / img_width
                        norm_height = h / img_height
                        
                        # 类别ID (假设球杆类别为0)
                        class_id = 0
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        return yolo_path
    
    def _convert_voc_to_yolo(self, img_path, voc_path):
        """将Pascal VOC格式转换为YOLO格式"""
        import xml.etree.ElementTree as ET
        
        # 读取图像尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        img_height, img_width = img.shape[:2]
        
        # 解析XML文件
        tree = ET.parse(voc_path)
        root = tree.getroot()
        
        # 创建YOLO格式标注文件
        yolo_path = self.temp_dir / f"{img_path.stem}.txt"
        
        with open(yolo_path, 'w') as f:
            for obj in root.findall('object'):
                # 获取边界框坐标
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
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return yolo_path
    
    def _remove_duplicates(self, data_list):
        """去除重复的图像数据"""
        print("🔍 检查重复数据...")
        
        unique_data = []
        seen_hashes = set()
        
        for item in data_list:
            # 计算图像文件的哈希值
            img_hash = self._calculate_file_hash(item['image'])
            
            if img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                unique_data.append(item)
            else:
                print(f"⚠️ 发现重复图像: {item['image'].name}")
        
        removed_count = len(data_list) - len(unique_data)
        if removed_count > 0:
            print(f"🗑️ 移除了 {removed_count} 个重复样本")
        else:
            print("✅ 未发现重复数据")
        
        return unique_data
    
    def _calculate_file_hash(self, file_path):
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _merge_to_processed_dataset(self, unique_data):
        """将新数据合并到已处理的数据集中"""
        print(f"📥 合并 {len(unique_data)} 个样本到训练数据集...")
        
        # 创建临时目录存放新数据
        temp_images_dir = self.temp_dir / "new_images"
        temp_labels_dir = self.temp_dir / "new_labels"
        temp_images_dir.mkdir(exist_ok=True)
        temp_labels_dir.mkdir(exist_ok=True)
        
        merged_count = 0
        
        for item in unique_data:
            try:
                # 生成唯一的文件名（避免与现有文件冲突）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                new_name = f"{item['image_id']}_{timestamp}"
                
                # 复制图像文件
                new_img_path = temp_images_dir / f"{new_name}{item['image'].suffix}"
                shutil.copy2(item['image'], new_img_path)
                
                # 复制标注文件
                new_label_path = temp_labels_dir / f"{new_name}.txt"
                shutil.copy2(item['annotation'], new_label_path)
                
                merged_count += 1
                
            except Exception as e:
                print(f"⚠️ 合并样本失败 {item['image'].name}: {str(e)}")
                continue
        
        print(f"✅ 成功合并 {merged_count} 个样本")
        return merged_count
    
    def _resplit_dataset(self):
        """重新分割数据集为训练集和验证集"""
        print("🔄 重新分割数据集...")
        
        # 收集所有数据（现有的 + 新的）
        all_images = []
        
        # 现有的训练集和验证集
        for split in ['train', 'val']:
            images_dir = self.processed_dir / "images" / split
            if images_dir.exists():
                all_images.extend(list(images_dir.glob("*")))
        
        # 新添加的数据
        temp_images_dir = self.temp_dir / "new_images"
        if temp_images_dir.exists():
            all_images.extend(list(temp_images_dir.glob("*")))
        
        # 清空现有的分割
        for split in ['train', 'val']:
            images_dir = self.processed_dir / "images" / split
            labels_dir = self.processed_dir / "labels" / split
            
            if images_dir.exists():
                shutil.rmtree(images_dir)
            if labels_dir.exists():
                shutil.rmtree(labels_dir)
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 随机分割 (80% 训练, 20% 验证)
        import random
        random.shuffle(all_images)
        
        split_point = int(len(all_images) * 0.8)
        train_images = all_images[:split_point]
        val_images = all_images[split_point:]
        
        # 移动文件到对应目录
        self._move_split_files(train_images, 'train')
        self._move_split_files(val_images, 'val')
        
        print(f"✅ 数据集重新分割完成:")
        print(f"   训练集: {len(train_images)} 个样本")
        print(f"   验证集: {len(val_images)} 个样本")
    
    def _move_split_files(self, image_list, split_name):
        """移动文件到指定的分割目录"""
        images_dir = self.processed_dir / "images" / split_name
        labels_dir = self.processed_dir / "labels" / split_name
        
        for img_path in image_list:
            # 移动图像文件
            new_img_path = images_dir / img_path.name
            if img_path.parent != images_dir:
                shutil.move(str(img_path), str(new_img_path))
            
            # 查找并移动对应的标注文件
            label_name = img_path.stem + ".txt"
            
            # 在多个可能的位置查找标注文件
            possible_label_paths = [
                img_path.parent.parent / "labels" / split_name / label_name,  # 现有结构
                self.temp_dir / "new_labels" / label_name,  # 新数据
                img_path.with_suffix('.txt')  # 同目录
            ]
            
            for label_path in possible_label_paths:
                if label_path.exists():
                    new_label_path = labels_dir / label_name
                    if label_path != new_label_path:
                        shutil.move(str(label_path), str(new_label_path))
                    break
    
    def _update_configs(self):
        """更新所有配置文件"""
        print("📝 更新配置文件...")
        
        # 更新YOLO数据集配置
        dataset_config = {
            'path': str(self.processed_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['golf_club']
        }
        
        config_file = self.configs_dir / 'dataset.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        # 更新训练配置
        training_config = {
            'last_update': datetime.now().isoformat(),
            'dataset_path': str(self.processed_dir.absolute()),
            'model_save_path': str(self.models_dir.absolute()),
            'total_samples': self._count_total_samples()
        }
        
        with open(self.configs_dir / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置文件已更新: {config_file}")
    
    def _count_total_samples(self):
        """统计总样本数"""
        train_count = len(list((self.processed_dir / "images" / "train").glob("*")))
        val_count = len(list((self.processed_dir / "images" / "val").glob("*")))
        return train_count + val_count
    
    def _archive_batch(self, batch_id, data):
        """归档批次数据"""
        batch_dir = self.archive_dir / f"batch_{batch_id}"
        batch_dir.mkdir(exist_ok=True)
        
        # 保存批次信息
        batch_info = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(data),
            'source': 'input_directory',
            'processing_log': self.processing_log.copy(),
            'samples': [
                {
                    'image_name': item['image'].name,
                    'image_id': item['image_id'],
                    'annotation_source': str(item.get('source_annotation', 'unknown'))
                }
                for item in data
            ]
        }
        
        with open(batch_dir / 'batch_info.json', 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        
        print(f"📦 批次信息已保存到: {batch_dir}")
    
    def _auto_train(self, batch_id=None):
        """自动决定训练策略并执行训练"""
        print("\n" + "="*50)
        print("🎯 开始自动训练...")
        print("="*50)
        
        # 检查是否存在已训练模型
        latest_model = self._find_latest_model()
        
        # 获取数据集统计信息
        dataset_stats = self._get_dataset_stats()
        
        # 智能决定训练参数
        training_config = self._determine_training_config(latest_model, dataset_stats, batch_id)
        
        print(f"🎯 训练模式: {training_config['mode']}")
        print(f"📊 数据集统计: {dataset_stats['total_images']} 图像, {dataset_stats['total_annotations']} 标注")
        print(f"⚙️ 训练参数: {training_config['epochs']} 轮, 学习率 {training_config['lr0']}")
        
        # 执行训练
        results = self._execute_training(training_config)
        
        # 训练后处理
        self._post_training_process(results, training_config)
    
    def _find_latest_model(self):
        """查找最新的训练模型"""
        latest_dir = self.models_dir / "latest"
        best_model = latest_dir / "best.pt"
        
        if best_model.exists():
            print(f"🔍 找到已有模型: {best_model}")
            return str(best_model)
        
        print("🔍 未找到已有模型，将进行全新训练")
        return None
    
    def _get_dataset_stats(self):
        """获取数据集统计信息"""
        train_images = list((self.processed_dir / "images" / "train").glob("*"))
        val_images = list((self.processed_dir / "images" / "val").glob("*"))
        
        train_labels = list((self.processed_dir / "labels" / "train").glob("*.txt"))
        val_labels = list((self.processed_dir / "labels" / "val").glob("*.txt"))
        
        # 统计标注数量
        total_annotations = 0
        for label_file in train_labels + val_labels:
            try:
                with open(label_file, 'r') as f:
                    total_annotations += len(f.readlines())
            except:
                continue
        
        return {
            'total_images': len(train_images) + len(val_images),
            'train_images': len(train_images),
            'val_images': len(val_images),
            'total_annotations': total_annotations
        }
    
    def _determine_training_config(self, latest_model, dataset_stats, batch_id):
        """智能确定训练配置"""
        config = {
            'mode': 'fresh',
            'epochs': 100,
            'lr0': 0.01,
            'batch_size': 16,
            'patience': 50,
            'device': 0 if torch.cuda.is_available() else 'cpu'
        }
        
        if latest_model:
            # 有已训练模型，进行增量训练
            config.update({
                'mode': 'incremental',
                'base_model': latest_model,
                'epochs': 50,  # 较少轮数
                'lr0': 0.001,  # 较低学习率
                'patience': 20
            })
            
            # 根据新数据量调整参数
            if batch_id:
                # 这里可以根据新数据比例调整参数
                config['epochs'] = 80
                config['lr0'] = 0.005
        
        # 根据数据集大小调整批次大小
        if dataset_stats['total_images'] < 100:
            config['batch_size'] = 8
        elif dataset_stats['total_images'] > 1000:
            config['batch_size'] = 32
        
        return config
    
    def _execute_training(self, config):
        """执行YOLO训练"""
        print(f"🚀 开始执行训练...")
        
        try:
            # 加载模型
            if config['mode'] == 'incremental' and config.get('base_model'):
                print(f"📥 加载已有模型: {config['base_model']}")
                model = YOLO(config['base_model'])
            else:
                print("📥 加载预训练模型: yolov8n.pt")
                model = YOLO('yolov8n.pt')
            
            # 训练参数
            train_args = {
                'data': str(self.configs_dir / 'dataset.yaml'),
                'epochs': config['epochs'],
                'lr0': config['lr0'],
                'batch': config['batch_size'],
                'patience': config['patience'],
                'device': config['device'],
                'project': str(self.models_dir),
                'name': 'latest',
                'exist_ok': True,
                'save': True,
                'verbose': True,
                'imgsz': 640
            }
            
            print(f"⚙️ 训练参数: {train_args}")
            
            # 开始训练
            results = model.train(**train_args)
            
            print("✅ 训练完成!")
            return results
            
        except Exception as e:
            print(f"❌ 训练过程中出错: {str(e)}")
            raise
    
    def _post_training_process(self, results, config):
        """训练后处理"""
        print("🔄 执行训练后处理...")
        
        # 保存训练日志
        training_log = {
            'timestamp': datetime.now().isoformat(),
            'mode': config['mode'],
            'config': config,
            'results_summary': {
                'save_dir': str(results.save_dir) if hasattr(results, 'save_dir') else 'unknown'
            }
        }
        
        log_file = self.logs_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)
        
        print(f"📝 训练日志已保存: {log_file}")
        
        # 将已训练的数据移动到trained_data目录
        self._move_trained_data_to_archive()
        
        print("✅ 训练后处理完成!")
    
    def _move_trained_data_to_archive(self):
        """将已训练的数据移动到trained_data目录"""
        print("📦 归档已训练的数据...")
        
        # 生成归档批次ID
        archive_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_batch_dir = self.trained_data_dir / f"batch_{archive_batch_id}"
        
        # 创建归档批次目录
        archive_images_dir = archive_batch_dir / "images"
        archive_annotations_dir = archive_batch_dir / "annotations"
        archive_metadata_dir = archive_batch_dir / "metadata"
        
        archive_images_dir.mkdir(parents=True, exist_ok=True)
        archive_annotations_dir.mkdir(parents=True, exist_ok=True)
        archive_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        
        # 移动输入目录中的数据
        if self.input_dir.exists():
            # 移动图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            # 直接在dataset目录中的图像
            for ext in image_extensions:
                for img_file in self.input_dir.glob(f"*{ext}"):
                    target_file = archive_images_dir / img_file.name
                    shutil.move(str(img_file), str(target_file))
                    moved_count += 1
                for img_file in self.input_dir.glob(f"*{ext.upper()}"):
                    target_file = archive_images_dir / img_file.name
                    shutil.move(str(img_file), str(target_file))
                    moved_count += 1
            
            # images子目录中的图像
            images_subdir = self.input_dir / "images"
            if images_subdir.exists():
                for ext in image_extensions:
                    for img_file in images_subdir.glob(f"*{ext}"):
                        target_file = archive_images_dir / img_file.name
                        shutil.move(str(img_file), str(target_file))
                        moved_count += 1
                    for img_file in images_subdir.glob(f"*{ext.upper()}"):
                        target_file = archive_images_dir / img_file.name
                        shutil.move(str(img_file), str(target_file))
                        moved_count += 1
                
                # 如果images目录为空，删除它
                if not any(images_subdir.iterdir()):
                    images_subdir.rmdir()
            
            # 移动标注文件
            annotation_extensions = ['.txt', '.json', '.xml']
            
            # 直接在dataset目录中的标注
            for ext in annotation_extensions:
                for ann_file in self.input_dir.glob(f"*{ext}"):
                    target_file = archive_annotations_dir / ann_file.name
                    shutil.move(str(ann_file), str(target_file))
            
            # annotations子目录中的标注
            annotations_subdir = self.input_dir / "annotations"
            if annotations_subdir.exists():
                for ext in annotation_extensions:
                    for ann_file in annotations_subdir.glob(f"*{ext}"):
                        target_file = archive_annotations_dir / ann_file.name
                        shutil.move(str(ann_file), str(target_file))
                
                # 如果annotations目录为空，删除它
                if not any(annotations_subdir.iterdir()):
                    annotations_subdir.rmdir()
            
            # 移动processed_videos目录（如果存在）
            processed_videos_dir = self.input_dir / "processed_videos"
            if processed_videos_dir.exists():
                target_processed_videos_dir = archive_metadata_dir / "processed_videos"
                shutil.move(str(processed_videos_dir), str(target_processed_videos_dir))
        
        # 创建归档元数据
        archive_metadata = {
            'archive_batch_id': archive_batch_id,
            'archive_time': datetime.now().isoformat(),
            'moved_files_count': moved_count,
            'source_directory': str(self.input_dir),
            'archive_directory': str(archive_batch_dir),
            'training_completed': True,
            'description': '训练完成后归档的数据'
        }
        
        metadata_file = archive_metadata_dir / "archive_info.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(archive_metadata, f, indent=2, ensure_ascii=False)
        
        # 更新总体归档记录
        self._update_trained_data_index(archive_batch_id, archive_metadata)
        
        print(f"📦 已归档 {moved_count} 个文件到: {archive_batch_dir}")
        print(f"📋 归档元数据已保存: {metadata_file}")
    
    def _update_trained_data_index(self, batch_id, metadata):
        """更新已训练数据索引"""
        index_file = self.trained_data_dir / "trained_data_index.json"
        
        # 读取现有索引
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        else:
            index_data = {
                'total_batches': 0,
                'total_files': 0,
                'batches': []
            }
        
        # 添加新批次
        index_data['total_batches'] += 1
        index_data['total_files'] += metadata['moved_files_count']
        index_data['batches'].append({
            'batch_id': batch_id,
            'archive_time': metadata['archive_time'],
            'files_count': metadata['moved_files_count'],
            'batch_directory': f"batch_{batch_id}"
        })
        
        # 保存更新后的索引
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 已训练数据索引已更新: 总批次 {index_data['total_batches']}, 总文件 {index_data['total_files']}")
    
    def _cleanup_input_directory(self):
        """清理训练输入目录"""
        print("🧹 检查训练输入目录清理状态...")
        
        # 检查是否还有剩余文件
        remaining_files = []
        
        if self.input_dir.exists():
            # 检查图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for ext in image_extensions:
                remaining_files.extend(list(self.input_dir.glob(f"*{ext}")))
                remaining_files.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
            
            # 检查子目录
            images_subdir = self.input_dir / "images"
            if images_subdir.exists():
                for ext in image_extensions:
                    remaining_files.extend(list(images_subdir.glob(f"*{ext}")))
                    remaining_files.extend(list(images_subdir.glob(f"*{ext.upper()}")))
            
            # 检查标注文件
            annotation_extensions = ['.txt', '.json', '.xml']
            for ext in annotation_extensions:
                remaining_files.extend(list(self.input_dir.glob(f"*{ext}")))
            
            annotations_subdir = self.input_dir / "annotations"
            if annotations_subdir.exists():
                for ext in annotation_extensions:
                    remaining_files.extend(list(annotations_subdir.glob(f"*{ext}")))
        
        if remaining_files:
            print(f"⚠️ 发现 {len(remaining_files)} 个未处理的文件")
            for file in remaining_files[:5]:  # 只显示前5个
                print(f"   - {file.name}")
            if len(remaining_files) > 5:
                print(f"   ... 还有 {len(remaining_files) - 5} 个文件")
        else:
            print("✅ 训练输入目录已清理完成，可以放入新的数据")
            print(f"📁 准备接收新数据: {self.input_dir}")
    
    def _has_existing_data(self):
        """检查是否有现有的训练数据"""
        train_images = list((self.processed_dir / "images" / "train").glob("*"))
        return len(train_images) > 0
    
    def _create_input_directory_guide(self):
        """创建输入目录使用指南"""
        guide_content = f"""
# 🏌️ 高尔夫球杆检测自动训练系统使用指南

## 数据流程说明

### 1. 视频标注阶段
将视频文件放入视频输入目录：
```
C:\\Users\\Administrator\\Desktop\\AIGolf\\videos\\
├── video1.mp4           # 视频文件 (.mp4, .avi, .mov, .mkv, .wmv)
├── video2.mp4
└── video3.avi
```

运行视频标注系统：
```bash
cd yolo_dataset_full
python video_annotation_system.py --mode rotated_bbox --frame_interval 10 --max_frames 50
```

### 2. 训练数据准备
视频标注系统会自动将标注结果输出到训练输入目录：
```
{self.input_dir}/
├── images/              # 从视频提取的帧图像
│   ├── video1_frame_000001.jpg
│   ├── video1_frame_000011.jpg
│   └── ...
├── annotations/         # YOLO格式标注文件
│   ├── video1_frame_000001.txt
│   ├── video1_frame_000011.txt
│   └── ...
└── processed_videos/    # 已处理视频记录
    └── processed_list.json
```

### 3. 自动训练
运行训练脚本：
```bash
cd yolo_dataset_full
python train_yolo_auto.py
```

### 4. 系统会自动完成
- ✅ 检测训练输入目录中的新数据
- ✅ 转换标注格式（如需要）
- ✅ 去除重复数据
- ✅ 合并到训练集
- ✅ 智能选择训练策略
- ✅ 执行训练
- ✅ 保存模型
- ✅ 归档已训练数据

### 5. 查看结果
- 最新模型: `yolo_dataset_full/models/latest/best.pt`
- 训练日志: `yolo_dataset_full/logs/`
- 数据归档: `yolo_dataset_full/archive/`
- 已训练数据: `yolo_dataset_full/trained_data/`

### 6. 数据管理
训练完成后，已训练的数据会自动移动到 `trained_data` 目录：
```
yolo_dataset_full/trained_data/
├── batch_20231201_143022/    # 按批次组织
│   ├── images/              # 已训练的图像
│   ├── annotations/         # 已训练的标注
│   └── metadata/           # 批次元数据
├── batch_20231202_091545/
└── trained_data_index.json  # 总体索引
```

## 完整工作流程

1. **准备视频**: 将视频文件放入 `C:\\Users\\Administrator\\Desktop\\AIGolf\\videos\\`
2. **视频标注**: 运行 `python video_annotation_system.py`
3. **自动训练**: 运行 `python train_yolo_auto.py`
4. **查看结果**: 检查 `models/latest/best.pt`
5. **继续训练**: 添加新视频，重复步骤1-3

现在请按照上述流程操作。如果训练输入目录为空，请先运行视频标注系统。
"""
        
        guide_file = self.base_dir / "使用指南.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"📖 使用指南已创建: {guide_file}")
        print(f"请按照指南操作：")
        print(f"1. 将视频放入: C:\\Users\\Administrator\\Desktop\\AIGolf\\videos\\")
        print(f"2. 运行视频标注: python video_annotation_system.py")
        print(f"3. 运行训练: python train_yolo_auto.py")
    
    def _log_step(self, step_name):
        """记录处理步骤"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {step_name}"
        self.processing_log.append(log_entry)
        print(f"🔄 {step_name}...")
    
    def _handle_error(self, error):
        """处理错误并保存日志"""
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'traceback': traceback.format_exc(),
            'processing_log': self.processing_log
        }
        
        error_file = self.logs_dir / f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False)
        
        print(f"📋 详细错误信息已保存到: {error_file}") 