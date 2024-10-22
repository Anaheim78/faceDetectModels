import cv2
import dlib
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from collections import deque

# 初始化 dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 初始化 MediaPipe
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

# 平滑化參數
SMOOTHING_WINDOW = 5
last_landmarks = None
landmark_queue = deque(maxlen=SMOOTHING_WINDOW)
detection_failed_count = 0
MAX_FAILED_FRAMES = 5

def smooth_landmarks(current_landmarks):
    """平滑化特徵點位置"""
    global last_landmarks, detection_failed_count
    
    if current_landmarks is None:
        detection_failed_count += 1
        if detection_failed_count > MAX_FAILED_FRAMES:
            last_landmarks = None
            return None
        return last_landmarks
    
    detection_failed_count = 0
    if last_landmarks is None:
        last_landmarks = current_landmarks
        return current_landmarks
    
    # 平滑化處理
    smoothed_landmarks = current_landmarks
    alpha = 0.7  # 平滑係數
    for i in range(68):
        try:
            current_point = (current_landmarks.part(i).x, current_landmarks.part(i).y)
            last_point = (last_landmarks.part(i).x, last_landmarks.part(i).y)
            smoothed_x = int(alpha * current_point[0] + (1 - alpha) * last_point[0])
            smoothed_y = int(alpha * current_point[1] + (1 - alpha) * last_point[1])
            smoothed_landmarks.parts()[i].x = smoothed_x
            smoothed_landmarks.parts()[i].y = smoothed_y
        except:
            continue
    
    last_landmarks = smoothed_landmarks
    return smoothed_landmarks

def draw_mediapipe_landmarks(frame, landmarks):
    """繪製 MediaPipe 的特徵點（半透明藍色）"""
    overlay = frame.copy()
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
        for landmark in landmarks
    ])
    
    custom_drawing_spec = mp.solutions.drawing_styles.DrawingSpec(
        color=(255, 200, 100),  # 淺藍色
        thickness=1,
        circle_radius=1
    )
    
    solutions.drawing_utils.draw_landmarks(
        image=overlay,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=custom_drawing_spec,
        connection_drawing_spec=custom_drawing_spec)
    
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def draw_dlib_landmarks(frame, landmarks):
    """繪製 dlib 的下顎特徵點"""
    if landmarks is None:
        return
        
    overlay = frame.copy()
    
    # 下顎特徵點
    jaw_points = list(range(1, 17))
    tmj_points = [1, 15]
    
    # 繪製顳顎關節點（藍色）
    for pt in tmj_points:
        try:
            x = landmarks.part(pt).x
            y = landmarks.part(pt).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        except:
            continue
    
    # 繪製下顎輪廓（綠色）
    for i in range(len(jaw_points)-1):
        try:
            x1 = landmarks.part(jaw_points[i]).x
            y1 = landmarks.part(jaw_points[i]).y
            x2 = landmarks.part(jaw_points[i+1]).x
            y2 = landmarks.part(jaw_points[i+1]).y
            
            cv2.circle(frame, (x1, y1), 3, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except:
            continue
    
    # 最後一個點
    try:
        x = landmarks.part(jaw_points[-1]).x
        y = landmarks.part(jaw_points[-1]).y
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    except:
        pass

# 打開攝像頭
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 創建視窗
cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Detection', 1280, 720)

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # MediaPipe 檢測
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect(mp_image)
        
        # Dlib 檢測
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if face_landmarker_result.face_landmarks:
            draw_mediapipe_landmarks(frame, face_landmarker_result.face_landmarks[0])
        
        # 只處理第一個檢測到的臉
        if len(faces) > 0:
            landmarks = predictor(gray, faces[0])
            smoothed_landmarks = smooth_landmarks(landmarks)
            if smoothed_landmarks is not None:
                draw_dlib_landmarks(frame, smoothed_landmarks)
        
        # 添加狀態指示
        if detection_failed_count > 0:
            cv2.putText(frame, "Stabilizing...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 顯示結果
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            break

cap.release()
cv2.destroyAllWindows()