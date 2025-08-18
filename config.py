# config.py
# -------------------
# 主要存放 KEYPOINT_WEIGHTS, MEDIAPIPE_POSE_CONNECTIONS, STAGE_MAP 等全局配置。

KEYPOINT_WEIGHTS = {
    'default': {
        'shoulders': 1.5,
        'arms': 1.2,
        'hips': 1.0,
        'legs': 0.8,
        'position_weight': 0.5,
        'angle_weight': 0.5,
        'angle_config': {
            'left_elbow': 1.0,
            'right_elbow': 1.0,
            'left_shoulder': 1.0,
            'right_shoulder': 1.0,
            'left_knee': 1.0,
            'right_knee': 1.0,
            'spine_angle': 1.0,
            'pelvic_rotation': 1.0,
            'shoulder_tilt': 1.0
        }
    },
}

MEDIAPIPE_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]

STAGE_MAP = {
    8: "8",  # 收杆阶段
    7: "7",  # 随杆阶段
    6: "6",  # 击球阶段
    5: "5",  # 下杆阶段
    4: "4",  # 顶点阶段
    3: "3",  # 上杆阶段
    2: "2",  # 起杆阶段
    1: "1",  # 引杆阶段
    0: "0"   # 准备阶段
}

BODY_POINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index"
]

# 统一可调阈值参数
CHECK_PARAMS = {
    "front": {
        "torso_sway": {
            # 基准垂直线向外侧偏移比例(相对肩宽)
            "line_outer_offset_ratio": 0.02,
            # 躯干越界允许容差(相对肩宽)
            "torso_tolerance_ratio": 0.05,
        },
        "hip_rotation": {
            # 髋部旋转角允许范围(度)
            "min_deg": 0.0,
            "max_deg": 25.0,
        },
        "shaft_range": {
            # 杆身角允许范围(度)
            "min_deg": 150.0,
            "max_deg": 180.0,
        },
    }
    ,
    "side": {
        "head_k_line": {
            "ref_point": "nose",
            "up_sw": 0.05,
            "down_sw": 0.10,
            "smooth_win": 5
        },
        "hip_line": {
            "use_center": True,
            "forward_sw": 0.08,
            "backward_sw": 0.06,
            "smooth_win": 5
        },
        "knee_bend": {
            "mode": "range",
            "min_deg": 150.0,
            "max_deg": 175.0,
            "delta_deg": 15.0
        },
        "feet_line": {
            "point": "foot_index",
            "max_deg": 2.5
        },
        "swing_path": {
            "use_baseline": "shaft",
            "lower_rel": -20.0,
            "upper_rel": 20.0
        }
    }
}

# 每个检测项的生效阶段表达式（支持不连续），pX 映射为阶段 X-1
# 例如 p1-p7 => 阶段 0..6
CONDITION_STAGE_RULES = {
    "front": {
        "躯干偏移检测": ["p1-p8"],
        "髋部旋转检测": ["p1-p7"],
        "杆身范围检测": ["p6"],
    },
    "side": {
        "头部K线检测": ["p1-p7"],
        "臀线检测": ["p1-p7"],
        "膝盖弯曲检测": ["p1-p7"],
        "双脚连线水平检测": ["p1"],
        "挥杆轨迹检测": ["p1-p8"]
    }
}
MARGIN_CONFIG = {
    "6": 1,         # impact 阶段前后扩展 5 帧
    "4": 1    # top of swing 阶段前后扩展 5 帧
}

ERROR_CHECK_STAGES = [0,1,2,3,4,5,6,7,8]