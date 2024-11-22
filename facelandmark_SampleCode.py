import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# 設定方法
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 自定義繪圖樣式
def get_custom_face_mesh_style():
    return mp.solutions.drawing_styles.DrawingSpec(
        color=(152, 223, 138),  # 淺綠色
        thickness=1,
        circle_radius=1
    )

def get_custom_face_connections_style():
    return mp.solutions.drawing_styles.DrawingSpec(
        color=(194, 220, 243),  # 淺藍色
        thickness=1
    )

# 人臉偵測設定
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

# 執行人臉偵測
with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)               # 讀取攝影鏡頭
    while True:
        ret, frame = cap.read()             # 讀取影片的每一幀
        if not ret:
            print("Cannot receive frame")   
            break
            
        w = frame.shape[1]                  # 畫面寬度
        h = frame.shape[0]                  # 畫面高度
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect(mp_image)

        face_landmarks_list = face_landmarker_result.face_landmarks
        annotated_image = np.copy(frame)

        # 處理每個偵測到的臉
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # 繪製臉部座標點
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            # 繪製所有臉部網格
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=get_custom_face_mesh_style(),
                connection_drawing_spec=get_custom_face_connections_style())

            # 繪製臉部輪廓
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=get_custom_face_mesh_style(),
                connection_drawing_spec=get_custom_face_connections_style())

        cv2.imshow('Face Landmarks', annotated_image)
        if cv2.waitKey(10) == ord('q'):     # 每一毫秒更新一次，直到按下 q 結束
            break

    cap.release()
    cv2.destroyAllWindows()