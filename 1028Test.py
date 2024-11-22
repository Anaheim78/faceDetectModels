# face_tracker.py
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from datetime import datetime

class FaceTracker:
    def __init__(self):
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=self.VisionRunningMode.IMAGE)
        
        self.landmarker = self.FaceLandmarker.create_from_options(self.options)
        self.cap = cv2.VideoCapture(0)
        
        # 關鍵點索引
        self.LEFT_CHEEK = 234  # 左顴骨
        self.RIGHT_CHEEK = 454 # 右顴骨
        self.NOSE_TIP = 4     # 鼻尖
        self.CHIN = 152       # 下巴
        
        # 記錄數據
        self.is_recording = False
        self.video_writer = None
        self.movement_data = []
        self.cycle_count = 0
        self.start_time = None

    def start_recording(self):
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = f'recording_{timestamp}.mp4'
            self.excel_path = f'movement_data_{timestamp}.xlsx'
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, 20.0, (width, height))
                
            self.movement_data = []
            self.start_time = datetime.now()
            self.is_recording = True
            return self.video_path

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # 儲存數據到Excel
            if self.movement_data:
                df = pd.DataFrame(self.movement_data)
                df.to_excel(self.excel_path, index=False)
                return self.video_path, self.excel_path
        return None, None

    def calculate_metrics(self, landmarks):
        if not landmarks:
            return None
            
        # 取得關鍵點座標
        cheek_left = np.array([landmarks[self.LEFT_CHEEK].x, landmarks[self.LEFT_CHEEK].y])
        cheek_right = np.array([landmarks[self.RIGHT_CHEEK].x, landmarks[self.RIGHT_CHEEK].y])
        nose = np.array([landmarks[self.NOSE_TIP].x, landmarks[self.NOSE_TIP].y])
        chin = np.array([landmarks[self.CHIN].x, landmarks[self.CHIN].y])
        
        # 計算各項指標
        now = datetime.now()
        elapsed_time = (now - self.start_time).total_seconds() if self.start_time else 0
        
        metrics = {
            'timestamp': now.strftime("%H:%M:%S.%f"),
            'elapsed_time': elapsed_time,
            'left_cheek_x': cheek_left[0],
            'left_cheek_y': cheek_left[1],
            'right_cheek_x': cheek_right[0],
            'right_cheek_y': cheek_right[1],
            'nose_x': nose[0],
            'nose_y': nose[1],
            'chin_x': chin[0],
            'chin_y': chin[1]
        }
        
        if self.is_recording:
            self.movement_data.append(metrics)
            
        return metrics

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_results = self.landmarker.detect(mp_image)
        
        if face_results.face_landmarks:
            landmarks = face_results.face_landmarks[0]
            metrics = self.calculate_metrics(landmarks)
            
            if metrics:
                # 繪製關鍵點
                h, w = frame.shape[:2]
                cv2.circle(frame, (int(metrics['left_cheek_x']*w), int(metrics['left_cheek_y']*h)), 5, (0,255,0), -1)
                cv2.circle(frame, (int(metrics['right_cheek_x']*w), int(metrics['right_cheek_y']*h)), 5, (0,255,0), -1)
                cv2.circle(frame, (int(metrics['nose_x']*w), int(metrics['nose_y']*h)), 5, (0,255,0), -1)
                cv2.circle(frame, (int(metrics['chin_x']*w), int(metrics['chin_y']*h)), 5, (0,255,0), -1)
                
                # 顯示錄影狀態
                if self.is_recording:
                    cv2.putText(frame, "Recording...", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.video_writer.write(frame)
        
        return frame

    def release(self):
        self.stop_recording()
        self.cap.release()