# api.py
# ---------------------------------------------------
import os
import sys
import tempfile
import traceback
from datetime import datetime
import csv
import numpy as np
import shutil
import json
import pandas as pd

from flask import Flask, request, jsonify
import torch
import cv2

from assistant_transparent_lines import generate_all_assistant_lines
from config import STAGE_MAP, ERROR_CHECK_STAGES, BODY_POINT_NAMES
from utils import combine_stages_to_video, extract_golf_swing_video, generate_rotated_video, convert_to_h264, ensure_file_permissions, combine_img_to_video, Timer, set_timing_enabled
from video_processor import (
    extract_keypoints_from_video,
    VideoProcessor,
    compute_stage_intervals,
    fill_frames_with_intervals
)
from swing_analyzer import SwingAnalyzer

from keypoint_3d_processor import Keypoint3DProcessor

from frame_analysis import analyze_frame
from multiprocessing import Pool
from functools import lru_cache
from golf_club_detector import add_golf_club_to_keypoints

log_filename = f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join("logs", log_filename)
os.makedirs("logs", exist_ok=True)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass  # 兼容 Python flush 机制

sys.stdout = Logger()
sys.stderr = sys.stdout  # 错误输出也写入日志
app = Flask(__name__)



# 声明全局变量
standard_model = None
analyzer = None
error_model = None
error_classes = []
stage2valid = {}
model_type = "basic"


analyzer = SwingAnalyzer(window_size=7)
    

def safe_delete_file(file_path):
    """
    安全地删除文件，处理文件被占用的情况
    
    参数:
        file_path: 要删除的文件路径
    """
    if not os.path.exists(file_path):
        return

    # 尝试多次释放和删除文件
    for i in range(3):
        try:
            # 强制垃圾回收，释放资源
            import gc
            gc.collect()
            
            # 在Windows上，尝试强制关闭所有打开的文件句柄
            if sys.platform == 'win32':
                # 先关闭所有视频捕获对象
                cv2.destroyAllWindows()
                
                # 等待系统释放文件
                import time
                time.sleep(0.5)
            
            # 尝试删除文件
            os.remove(file_path)
            print(f"[INFO] 成功删除临时文件: {file_path}")
            return
        except Exception as e:
            if i == 2:  # 最后一次尝试
                print(f"[WARN] 无法删除临时文件 {file_path}: {str(e)}")
                # 注册在程序退出时删除
                import atexit
                atexit.register(lambda p=file_path: os.remove(p) if os.path.exists(p) else None)
            else:
                # 等待更长时间后重试
                import time
                time.sleep(1)

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
    
    cap = cv2.VideoCapture(video_path)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    need_rotate = (width > height)  # 判断是否需要旋转
    
    # 确保帧索引在有效范围内
    start_frame = max(0, start_frame) if start_frame is not None else 0
    end_frame = min(total_frame_count - 1, end_frame) if end_frame is not None else total_frame_count - 1
    frame_count = end_frame - start_frame + 1
    
    print(f"[INFO] 正在保存视频帧到 {output_dir}，帧范围：{start_frame}-{end_frame}，共 {frame_count} 帧")
    
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
        cv2.imwrite(os.path.join(output_dir, out_name), frame, image_quality)
    
    cap.release()
    return frame_count, need_rotate

def create_video_from_images(img_folder, output_path, fps=30.0, img_pattern="frame*.jpg"):
    """
    从图片序列创建视频
    
    参数:
        img_folder: 图片文件夹路径
        output_path: 输出视频路径
        fps: 帧率
        img_pattern: 图片匹配模式
        
    返回:
        success: 是否成功
    """
    try:
        print(f"[INFO] 开始从图片创建视频: {output_path}")
        result = combine_img_to_video(
            img_folder=img_folder,
            output_path=output_path,
            fps=fps,
            img_pattern=img_pattern
        )
        
        success = result["status"] == "success"
        if success:
            print(f"[INFO] 成功从图片合成视频: {output_path}")
        else:
            print(f"[WARN] 合成视频失败: {result.get('message', '未知错误')}")
        
        return success
    except Exception as e:
        print(f"[WARN] 视频合成过程中出错: {str(e)}")
        return False

def ensure_h264_encoding(video_path):
    """
    确保视频使用H264编码
    
    参数:
        video_path: 视频文件路径
        
    返回:
        success: 是否成功
    """
    try:
        if not os.path.exists(video_path):
            print(f"[WARN] 视频文件不存在: {video_path}")
            return False
            
        print(f"[INFO] 尝试确保视频使用H264编码: {video_path}")
        convert_to_h264(video_path, overwrite_input=True)
        ensure_file_permissions(video_path)
        return True
    except Exception as e:
        print(f"[WARN] 视频转换为H264过程中出错: {str(e)}")
        return False

@app.route("/process_video", methods=["POST"])
def process_video_and_detect():
    """
    接口说明：处理视频并进行错误检测
    """
    Timer.reset()  # 重置计时器
    Timer.start("total")  # 开始总计时
    
    if "video" not in request.files:
        return jsonify({"error": "未上传视频文件"}), 400

    video_file = request.files["video"]
    base_name = os.path.splitext(video_file.filename)[0]
    
    # 使用临时文件保存上传的视频
    tmp_video_path = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_video_path = tmp.name
        video_file.save(tmp_video_path)
    
    # 处理上传的视频文件
    try:
        # 1. 提取关键点
        Timer.start("extract_keypoints")
        keypoint_data = extract_keypoints_from_video(tmp_video_path)
        Timer.end("extract_keypoints")
        
        frame_count = keypoint_data.shape[0]
        if frame_count == 0:
            return jsonify({"error": "视频中未提取到关键点"}), 400

        # 获取视频帧率
        cap = cv2.VideoCapture(tmp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # 2. 阶段分类
        Timer.start("stage_classification")
        keypoint_data_flat = keypoint_data.view(frame_count, -1)
        
        # 使用新的基于物理特性的阶段分类方法，并获取旋转类型信息
        stage_indices, rotation_type = analyzer.identify_swing_stages(keypoint_data, top_k=1)
        
        # 如果想使用原始方法，可以取消下面的注释
        # stage_indices = analyzer.find_swing_stages(keypoint_data_flat, top_k=3)
        Timer.end("stage_classification")

        # 3. 构建输出目录
        Timer.start("setup_directories")
        output_folder = os.path.join("./resultData", base_name)
        csv_folder = os.path.join(output_folder, "csv")
        img_folder = os.path.join(output_folder, "img")
        video_folder = os.path.join(output_folder, "video")
        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(video_folder, exist_ok=True)
        Timer.end("setup_directories")
        
        # 4. 视频处理
        Timer.start("video_processing")
        # 设置临时输出路径，不直接写入original.mp4
        temp_output_path = os.path.join(video_folder, "temp_extracted.mp4")
        
        # 确定剪辑范围
        start_frame, end_frame = extract_golf_swing_video(stage_indices, tmp_video_path, temp_output_path, start_phase='1', end_phase='8', fps=fps)
        
        # 完成后删除临时视频文件
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except Exception as e:
                print(f"[WARN] 无法删除临时视频文件: {str(e)}")
        
        if start_frame is not None and start_frame > 0:
            # 调整stage_indices中的帧索引以适应剪切后的视频
            stage_indices = {k: [v_item - start_frame for v_item in v if v_item >= start_frame] 
                            for k, v in stage_indices.items()}
            # 检查列表是否为空，为空则设为[0]，否则取最小值
            if not stage_indices['0']:
                stage_indices['0'] = [0]
            else:
                stage_indices['0'] = [min(stage_indices['0'])]
            print(f"[INFO] 已调整阶段索引，减去起始帧 {start_frame}")
        else:
            start_frame = 0
        
        # 4. 将剪裁后的视频帧保存为图片
        img_all_dir = os.path.join(img_folder, "all")

        # 计算实际需要提取的帧数
        extract_frame_count = end_frame - start_frame + 1 if start_frame is not None and end_frame is not None else None
        print(f"[INFO] 需要提取的视频帧数: {extract_frame_count}")

        frame_count, need_rotate = extract_frames_to_images(
            video_path=tmp_video_path,
            output_dir=img_all_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps
        )

        # 验证提取的帧数是否正确
        if frame_count != extract_frame_count:
            print(f"[WARN] 提取的帧数({frame_count})与预期({extract_frame_count})不一致")
        
        # 5. 基于图片合成H264视频
        h264_video_path = os.path.join(video_folder, "original_h264.mp4")
        create_video_from_images(
            img_folder=img_all_dir,
            output_path=h264_video_path,
            fps=fps,
            img_pattern="frame*.jpg"
        )
        
        # 确保视频使用H264编码
        h264_output_path = os.path.join(video_folder, "original.mp4")
        convert_to_h264(h264_video_path, h264_output_path)
        
        try:
            # 如果转换成功，可以删除中间文件
            if os.path.exists(h264_output_path) and os.path.getsize(h264_output_path) > 0:
                os.remove(h264_video_path)
                print(f"[INFO] 成功创建H264编码视频: {h264_output_path}")
            else:
                # 如果转换失败，使用原始文件
                os.rename(h264_video_path, h264_output_path)
                print(f"[WARN] H264转换可能不成功，使用原始文件: {h264_output_path}")
        except Exception as e:
            print(f"[ERROR] 处理视频文件时出错: {str(e)}")
            
        # 6. 重新处理视频获取关键点
        print(f"[INFO] 从剪裁后的视频重新提取关键点数据")
        keypoint_data = extract_keypoints_from_video(h264_output_path)
        if keypoint_data.shape[0] == 0:
            return jsonify({"error": "从剪裁后的视频中未提取到关键点"}), 400
        print(f"[INFO] 成功提取关键点数据，形状: {keypoint_data.shape}")

        # 🆕 添加球杆检测信息到关键点数据
        try:
            print(f"[INFO] 开始添加球杆检测信息...")
            enhanced_keypoint_data = add_golf_club_to_keypoints(
                existing_keypoints=keypoint_data.numpy(),  # 转换为numpy数组
                video_path=h264_output_path,
                confidence=0.2  # 使用较低的置信度阈值以提高检测率
            )
            # 转换回torch tensor
            keypoint_data = torch.from_numpy(enhanced_keypoint_data).float()
            print(f"[INFO] 成功添加球杆检测信息，增强后形状: {keypoint_data.shape}")
        except Exception as golf_err:
            print(f"[WARN] 添加球杆检测信息失败: {str(golf_err)}")
            print(f"[INFO] 继续使用原始关键点数据，形状: {keypoint_data.shape}")
            # 保持原有的keypoint_data不变，确保系统正常运行

        # 7. 如果使用增强模型，提取动态特征
        if model_type == "enhanced":
            print(f"[INFO] 使用增强错误检测模型进行分析，提取动态特征...")
            # 提取3D姿态和动态特征
            keypoint_np = keypoint_data.numpy()
            dynamic_features = Keypoint3DProcessor.extract_dynamic_features(keypoint_np, fps)
            
            # 保存部分动态特征到CSV文件（可选）
            velocity_csv = os.path.join(csv_folder, "velocity.csv")
            with open(velocity_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "joint", "velocity_x", "velocity_y", "velocity_z", "magnitude"])
                
                # 只取部分关键关节点保存速度数据
                key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]  # 肩、肘、腕、髋、膝
                
                for i in range(len(dynamic_features["velocities"])):
                    frame_idx = i + 1  # 第一帧没有速度，从第二帧开始
                    for j in key_joints:
                        v = dynamic_features["velocities"][i, j]
                        mag = np.linalg.norm(v)
                        writer.writerow([frame_idx, BODY_POINT_NAMES[j], v[0], v[1], v[2], mag])

        # 8. 保存关键点数据到keypoints.pt
        keypoints_pt_path = os.path.join(output_folder, "keypoints.pt")
        print(f"[INFO] 保存关键点数据到: {keypoints_pt_path}")
        torch.save(keypoint_data, keypoints_pt_path)

        # 9. 保存阶段索引信息，为后续处理提供便利
        stage_indices_path = os.path.join(output_folder, "stage_indices.json")
        with open(stage_indices_path, 'w') as f:
            json.dump(stage_indices, f)
        print(f"[INFO] 保存阶段索引到: {stage_indices_path}")

        # 10. 调用 VideoProcessor.handle_video，将 CSV 文件保存为 allData.csv，并将骨骼图保存至 img_folder
        points_err_list, stageid_err_list = VideoProcessor.handle_video(
            video_path=h264_output_path,  # 使用h264编码的视频
            stage_indices=stage_indices,
            output_folder=output_folder,
            keypoint_data=keypoint_data,
            analyzer=analyzer,
            csv_filename="allData.csv",     
            img_folder=img_folder,
            rotation_type=rotation_type           # 传递旋转类型信息
        )
        

        # 14. 尝试生成辅助线透明图片
        Timer.start("generate_assistant_lines")
        try:
            print(f"[INFO] 开始生成辅助线透明PNG图片...")
            lines_result = generate_all_assistant_lines(base_name)
            
            if "error" not in lines_result:
                print(f"[INFO] 辅助线透明图片生成完成: {lines_result.get('stance_line_count', 0)}帧")
                
                # 确保生成的图片有正确的权限
                stance_folder = lines_result.get("stance_line_folder", "")
                if os.path.exists(stance_folder):
                    for file in os.listdir(stance_folder):
                        if file.endswith(".png"):
                            ensure_file_permissions(os.path.join(stance_folder, file))
            else:
                print(f"[WARN] 辅助线透明图片生成失败: {lines_result.get('error', '未知错误')}")
        except Exception as lines_err:
            print(f"[WARN] 生成辅助线透明图片过程中出错: {str(lines_err)}")
            # 这里只记录警告，不影响正常流程
        Timer.end("generate_assistant_lines")

        Timer.end("total")  # 结束总计时
        Timer.print_stats()  # 打印统计信息
        
        # 返回结果时包含性能统计
        return jsonify({
            "msg": "视频处理与错误检测完成",
            "stage_indices": stage_indices,
            "output_folder": output_folder,
            "model_type": model_type,
            "performance_stats": Timer.get_stats()  # 添加性能统计信息
        })
    except Exception as e:
        Timer.end("total")  # 确保在发生异常时也能记录总时间
        Timer.print_stats()
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # 安全地删除临时文件
        try:
            if os.path.exists(tmp_video_path):
                import gc
                gc.collect()
                os.remove(tmp_video_path)
        except Exception as del_err:
            print(f"[WARN] 无法删除临时文件 {tmp_video_path}: {str(del_err)}")


@app.route("/convert_videos_to_h264", methods=["POST"])
def convert_videos_to_h264_api():
    """
    接口说明：将指定的视频文件转换为H264编码
    JSON 示例：
    {
      "session_id": "WeChat_20250411113108",
      "video_files": ["result.mp4", "original.mp4"]
    }
    """
    from utils import convert_to_h264
    
    data = request.json if request.is_json else {}
    session_id = data.get("session_id", "")
    video_files = data.get("video_files", ["result.mp4", "original.mp4"])
    
    if not session_id:
        return jsonify({"error": "缺少必要的会话ID"}), 400
    
    try:
        # 构建关键路径
        session_folder = os.path.join("./resultData", session_id)
        video_folder = os.path.join(session_folder, "video")
        
        if not os.path.exists(video_folder):
            return jsonify({"error": f"找不到视频文件夹: {video_folder}"}), 404
        
        # 逐个处理视频文件
        results = {}
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            
            if not os.path.exists(video_path):
                results[video_file] = {
                    "status": "error",
                    "message": f"找不到视频文件: {video_path}"
                }
                continue
            
            # 尝试转换视频
            success = convert_to_h264(
                input_video_path=video_path,
                overwrite_input=True
            )
            
            results[video_file] = {
                "status": "success" if success else "error",
                "message": f"视频已成功转换为H264编码" if success else "视频转换失败"
            }
        
        return jsonify({
            "status": "ok",
            "message": "视频转换处理完成",
            "results": results
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/check_swing_conditions", methods=["POST"])
def check_swing_conditions_api():
    """
    检查挥杆动作是否符合指定条件的API端点
    
    输入参数:
        video_name: 视频文件名（不含扩展名）
        output_dir: (可选) 输出目录，默认为results/视频名
        
    返回:
        JSON对象，包含:
            success: 是否成功
            message: 成功或失败的消息
            report_file: 报告文件路径（如果成功）
    """
    try:
        # 获取请求参数
        data = request.get_json()
        
        # 检查必要参数
        if 'video_name' not in data:
            return jsonify({
                'success': False,
                'message': '缺少必要参数: video_name'
            }), 400
        
        video_name = data['video_name']
        output_dir = data.get('output_dir', None)
        
        # 导入挥杆检查模块
        from golf_swing_checker import check_swing
        
        # 执行挥杆检查
        success, report_file = check_swing(video_name, output_dir, generate_visualizations=True, csv_format=True)
        
        if success:
            return jsonify({
                'success': True,
                'message': '成功生成挥杆检查报告',
                'report_file': report_file
            })
        else:
            return jsonify({
                'success': False,
                'message': '生成挥杆检查报告失败'
            }), 500
            
    except Exception as e:
        print(f"[ERR] 挥杆检查API出错: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'服务器错误: {str(e)}'
        }), 500

def save_to_csv(data, csv_path):
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # 设定默认端口和host
    port = 5000  
    host = '0.0.0.0'
    
    # 获取命令行参数
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    # 创建必要的目录
    os.makedirs("./resultData", exist_ok=True)
    
    # 重定向输出
    sys.stdout = Logger()
    sys.stderr = Logger()
    
    # 启动服务器
    print(f"[INFO] 服务器启动于 http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
