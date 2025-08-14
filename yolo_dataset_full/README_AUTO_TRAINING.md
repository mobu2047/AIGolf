# 🏌️ 高尔夫球杆检测完整训练系统

## 🎯 系统概述

这是一个完整的高尔夫球杆检测训练系统，包含视频标注和自动化训练两个核心模块。用户只需要放入视频文件即可完成从视频处理到模型训练的全流程。

### 📁 路径配置

- **视频输入目录：** `C:\Users\Administrator\Desktop\AIGolf\videos`
- **数据集目录：** `C:\Users\Administrator\Desktop\AIGolf\dataset` (视频标注输出 → 训练输入)

### ✨ 核心特性

- ✅ **视频标注**：从视频中提取帧并进行交互式标注
- ✅ **零配置训练**：用户无需了解技术细节
- ✅ **智能化**：自动选择最佳训练策略
- ✅ **增量训练**：新数据自动合并，支持模型持续改进
- ✅ **完全自动化**：数据处理、训练、保存一键完成
- ✅ **数据管理**：完整的数据归档和已训练数据管理
- ✅ **可追溯性**：完整的数据和训练历史记录

## 📁 完整目录结构

```
C:\Users\Administrator\Desktop\AIGolf\
├── videos/                   # 🎬 视频输入目录
│   ├── golf_swing_1.mp4     # 用户放入的视频文件
│   ├── golf_swing_2.mp4
│   └── golf_swing_3.avi
├── dataset/                  # 📊 数据集目录 (视频标注输出 → 训练输入)
│   ├── images/              # 从视频提取的帧图像
│   │   ├── video1_frame_000001.jpg
│   │   └── ...
│   ├── annotations/         # YOLO格式标注文件
│   │   ├── video1_frame_000001.txt
│   │   └── ...
│   └── processed_videos/    # 已处理视频记录
└── yolo_dataset_full/       # 🤖 训练系统目录
    ├── processed/           # 已处理的完整数据集
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   └── labels/
    │       ├── train/
    │       └── val/
    ├── models/              # 训练好的模型
    │   └── latest/
    │       └── best.pt     # 最新的最佳模型
    ├── trained_data/        # 🗃️ 已训练数据归档
    │   ├── batch_20231201_143022/
    │   ├── batch_20231202_091545/
    │   └── trained_data_index.json
    ├── archive/             # 历史数据备份
    ├── configs/             # 配置文件
    ├── logs/               # 训练日志
    └── temp/               # 临时文件
```

## 🚀 快速开始

### 方法一：使用简化启动脚本（推荐）

```bash
# 1. 将视频文件放入videos目录
copy your_videos/* C:\Users\Administrator\Desktop\AIGolf\videos\

# 2. 启动视频标注（一键启动）
cd C:\Users\Administrator\Desktop\AIGolf\yolo_dataset_full
python start_video_annotation.py

# 3. 启动自动训练（一键启动）
python train_yolo_auto.py
```

### 方法二：分步执行

#### 步骤1：视频标注

```bash
# 将视频文件放入videos目录
C:\Users\Administrator\Desktop\AIGolf\videos\
├── video1.mp4          # 支持 .mp4, .avi, .mov, .mkv, .wmv
├── video2.mp4
└── video3.avi

# 运行视频标注系统
cd C:\Users\Administrator\Desktop\AIGolf\yolo_dataset_full
python video_annotation_system.py --mode rotated_bbox --frame_interval 10 --max_frames 50
```

**标注参数说明：**
- `--mode rotated_bbox`：旋转边界框模式（推荐）
- `--frame_interval 10`：每10帧提取一帧
- `--max_frames 50`：每个视频最多提取50帧

#### 步骤2：自动训练

```bash
# 运行自动训练系统
cd C:\Users\Administrator\Desktop\AIGolf\yolo_dataset_full
python train_yolo_auto.py
```

## 📊 支持的数据格式

### 视频格式
- `.mp4` - MP4视频文件
- `.avi` - AVI视频文件  
- `.mov` - QuickTime视频文件
- `.mkv` - Matroska视频文件
- `.wmv` - Windows Media视频文件

### 标注格式
系统自动生成YOLO格式标注，同时保存详细的JSON元数据。

## 🎯 训练模式

系统会根据现有数据自动选择训练模式：

### 🆕 全新训练
- **触发条件**：没有已训练的模型
- **训练参数**：100轮，学习率0.01
- **适用场景**：首次训练

### 🔄 增量训练
- **触发条件**：存在已训练模型且有新数据
- **训练参数**：50轮，学习率0.001
- **适用场景**：添加新数据后的模型改进

### ⏸️ 恢复训练
- **触发条件**：存在已训练模型但无新数据
- **训练参数**：继续之前的训练
- **适用场景**：训练中断后的恢复

## 🛠️ 高级功能

### 视频标注参数调整

```bash
# 高密度标注（更多帧）
python video_annotation_system.py --frame_interval 5 --max_frames 100

# 快速标注（较少帧）
python video_annotation_system.py --frame_interval 20 --max_frames 30

# 不同标注模式
python video_annotation_system.py --mode bbox           # 轴对齐边界框
python video_annotation_system.py --mode rotated_bbox   # 旋转边界框（推荐）
python video_annotation_system.py --mode line          # 线段标注
python video_annotation_system.py --mode polygon       # 多边形标注
```

### 数据管理

#### 查看已训练数据
```bash
# 查看已训练数据索引
cat yolo_dataset_full/trained_data/trained_data_index.json

# 查看特定批次信息
cat yolo_dataset_full/trained_data/batch_20231201_143022/metadata/archive_info.json
```

#### 恢复已训练数据（如需要）
```bash
# 从trained_data目录恢复数据到dataset目录
# 注意：这会覆盖当前的dataset内容
cp -r yolo_dataset_full/trained_data/batch_20231201_143022/images/* C:\Users\Administrator\Desktop\AIGolf\dataset\images\
cp -r yolo_dataset_full/trained_data/batch_20231201_143022/annotations/* C:\Users\Administrator\Desktop\AIGolf\dataset\annotations\
```

## 📈 训练监控

### 实时监控
训练过程中会显示：
- 📊 数据集统计信息
- 🎯 训练模式和参数
- 📈 训练指标变化
- ⏱️ 预计完成时间

### 日志文件
- `logs/training_log_YYYYMMDD_HHMMSS.json`：训练配置和结果
- `logs/error_log_YYYYMMDD_HHMMSS.json`：错误信息（如有）

### 模型文件
- `models/latest/best.pt`：最佳模型（推荐使用）
- `models/latest/last.pt`：最后一轮模型

## 📋 完整工作流程示例

### 示例1：首次使用

```bash
# 1. 准备视频
copy /path/to/your/videos/* C:\Users\Administrator\Desktop\AIGolf\videos\

# 2. 视频标注
cd C:\Users\Administrator\Desktop\AIGolf\yolo_dataset_full
python start_video_annotation.py

# 3. 自动训练
python train_yolo_auto.py

# 4. 查看结果
ls models/latest/best.pt
```

### 示例2：增量训练

```bash
# 1. 添加新视频
copy /path/to/new/videos/* C:\Users\Administrator\Desktop\AIGolf\videos\

# 2. 标注新视频（系统会跳过已处理的视频）
python start_video_annotation.py

# 3. 增量训练（系统会自动检测到新数据）
python train_yolo_auto.py
```

### 示例3：批量处理

```bash
# 处理多个视频目录
for video_dir in video_batch_1 video_batch_2 video_batch_3; do
    copy $video_dir/* C:\Users\Administrator\Desktop\AIGolf\videos\
    python start_video_annotation.py
    python train_yolo_auto.py
done
```

## 🚨 故障排除

### 常见问题

**Q: 提示"未找到视频文件"**
- A: 检查 `C:\Users\Administrator\Desktop\AIGolf\videos\` 中是否有视频文件
- A: 确保视频格式为支持的格式 (.mp4, .avi, .mov, .mkv, .wmv)

**Q: 视频标注窗口无法显示**
- A: 检查OpenCV是否正确安装：`pip install opencv-python`
- A: 确保系统支持图形界面显示

**Q: 训练提示"未找到新数据"**
- A: 确保已完成视频标注步骤
- A: 检查 `C:\Users\Administrator\Desktop\AIGolf\dataset\` 中是否有图像和标注文件

**Q: 内存不足**
- A: 减少每个视频的最大帧数：`--max_frames 30`
- A: 增加帧间隔：`--frame_interval 20`
- A: 关闭其他占用内存的程序

**Q: 训练速度慢**
- A: 检查是否使用GPU：确保CUDA环境配置正确
- A: 减少批次大小（系统会自动调整）

### 系统测试

```bash
# 运行系统测试
cd C:\Users\Administrator\Desktop\AIGolf\yolo_dataset_full
python test_system.py
```

### 调试模式

```bash
# 启用详细日志
set YOLO_DEBUG=1
python train_yolo_auto.py
```

## 🔄 数据流程图

```
视频文件 → 视频标注系统 → 数据集 → 训练系统 → 模型 + 已训练数据归档
   ↓           ↓            ↓         ↓        ↓
videos/   → dataset/   → processed/ → models/ + trained_data/
```

## 📞 技术支持

如遇到问题，请：

1. 查看 `logs/` 目录中的详细日志
2. 运行 `python test_system.py` 检查系统状态
3. 检查系统要求和依赖
4. 参考故障排除指南

## 🎉 开始使用

现在您可以开始使用这个强大的完整训练系统了！

### 快速启动命令

```bash
# 一键启动视频标注
cd C:\Users\Administrator\Desktop\AIGolf\yolo_dataset_full
python start_video_annotation.py

# 一键启动训练
python train_yolo_auto.py
```

### 完整流程

1. 将视频放入 `C:\Users\Administrator\Desktop\AIGolf\videos\`
2. 运行 `python start_video_annotation.py`
3. 运行 `python train_yolo_auto.py`
4. 在 `models/latest/best.pt` 找到您的模型

祝您训练愉快！🏌️‍♂️ 