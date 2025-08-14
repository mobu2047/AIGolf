# frame_analysis.py
import cv2
import numpy as np
import mediapipe as mp

# 以下参数可根据实际情况调整
PERSON_MIN_AREA_RATIO = 0.1  # 人体所占面积低于10%时，认为人物太小，提示“请靠近”
PERSON_MAX_AREA_RATIO = 0.5  # 人体所占面积超过50%时，认为人物太大，提示“请远离”
HORIZONTAL_OFFSET_THRESHOLD = 0.3  # 人体中心水平偏离图像中心超过10%时，提示“请靠左”或“请靠右”
BORDER_MARGIN_RATIO = 0.01  # 若人体边界离图像边缘小于1%图像尺寸，则认为未完整入镜


def analyze_frame(image_path: str) -> dict:
    """
    分析单帧图片中的人体关键点，判断人体是否完整入镜以及尺寸和水平位置情况。
    返回一个字典，包含：
      - complete: 是否完整入镜（布尔值）
      - size_message: 针对尺寸的建议（例如“请靠近”或“请远离”，否则为“合适”）
      - position_message: 针对水平位置的建议（例如“请靠左”或“请靠右”，否则为“居中”）
      - bbox: 人体边界框 (x, y, w, h)
      - area_ratio: 人体边界框面积占图像面积比例
      - offset_ratio: 水平偏差比例
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "无法读取图片"}
    h, w, _ = image.shape

    # 利用 MediaPipe Pose 进行关键点检测
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return {"error": "未检测到人体关键点"}

    # 提取关键点坐标（转换为像素坐标）
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append((landmark.x * w, landmark.y * h))
    keypoints = np.array(keypoints)  # (33,2)

    # 计算人体边界框
    min_xy = keypoints.min(axis=0)
    max_xy = keypoints.max(axis=0)
    bbox_x = int(min_xy[0])
    bbox_y = int(min_xy[1])
    bbox_w = int(max_xy[0] - min_xy[0])
    bbox_h = int(max_xy[1] - min_xy[1])
    bbox = (bbox_x, bbox_y, bbox_w, bbox_h)

    # 判断是否完整入镜：若边界框离图像边缘小于一定比例，则认为人物未完整入镜
    margin_w = w * BORDER_MARGIN_RATIO
    margin_h = h * BORDER_MARGIN_RATIO
    complete = True
    if bbox_x <= margin_w or bbox_y <= margin_h or (bbox_x + bbox_w) >= (w - margin_w) or (bbox_y + bbox_h) >= (
            h - margin_h):
        complete = False

    # 计算边界框面积占比
    bbox_area = bbox_w * bbox_h
    image_area = w * h
    area_ratio = bbox_area / image_area

    size_message = "合适"
    if area_ratio < PERSON_MIN_AREA_RATIO:
        size_message = "请靠近"
    elif area_ratio > PERSON_MAX_AREA_RATIO:
        size_message = "请远离"
    if not complete:
        size_message = "请远离"

    # 判断人体水平位置
    bbox_center_x = bbox_x + bbox_w / 2.0
    image_center_x = w / 2.0
    offset_ratio = (bbox_center_x - image_center_x) / image_center_x  # 正表示偏右
    position_message = "居中"
    if offset_ratio < -HORIZONTAL_OFFSET_THRESHOLD:
        position_message = "请靠右"
    elif offset_ratio > HORIZONTAL_OFFSET_THRESHOLD:
        position_message = "请靠左"

    return {
        "complete": complete,
        "size_message": size_message,
        "position_message": position_message,
        "bbox": bbox,
        "area_ratio": area_ratio,
        "offset_ratio": offset_ratio
    }

if __name__ == "__main__":
    print(analyze_frame('pianjin.jpg'))