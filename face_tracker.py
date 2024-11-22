# face_tracker.py
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from datetime import datetime
from movement_analyzer import MovementAnalyzer

class FaceTracker:
    def __init__(self):
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        # 臉部關鍵點索引
        self.LEFT_EYE = 33
        self.RIGHT_EYE = 263
        self.NOSE_TIP = 4
        self.CHIN = 152
        
        self.setup_face_detection()
        self.reset_recording()
        
    def setup_face_detection(self):
        """初始化人臉偵測"""
        self.options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path='face_landmarker.task'),
            running_mode=self.VisionRunningMode.IMAGE
        )
        self.landmarker = self.FaceLandmarker.create_from_options(self.options)
        self.cap = cv2.VideoCapture(0)
        
    def reset_recording(self):
        """重置錄影相關變數"""
        self.is_recording = False
        self.video_writer = None
        self.movement_data = []
        self.start_time = None
        
    def start_recording(self):
        """開始錄影"""
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = f'recording_{timestamp}.mp4'
            self.excel_path = f'movement_data_{timestamp}.xlsx'
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, 30.0, (width, height))
                
            self.movement_data = []
            self.start_time = datetime.now()
            self.is_recording = True
            return self.video_path
            
    def stop_recording(self):
        """停止錄影並分析數據"""
        if self.is_recording:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            if self.movement_data:
                analyzer = MovementAnalyzer()
                
                # 儲存原始數據
                df_raw = pd.DataFrame(self.movement_data)
                df_raw.to_excel(self.excel_path, index=False)
                
                # 分析數據
                df_analysis = analyzer.analyze_movement(self.movement_data)
                analysis_path = self.excel_path.replace('.xlsx', '_analysis.xlsx')
                df_analysis.to_excel(analysis_path, index=False)
                
                return self.video_path, self.excel_path, analysis_path
                
        return None, None, None
        
    def get_frame(self):
        """獲取並處理每一幀影像"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_results = self.landmarker.detect(mp_image)
        
        if face_results.face_landmarks:
            landmarks = face_results.face_landmarks[0]
            
            # 記錄關鍵點位置
            if self.is_recording:
                data = {
                    'timestamp': datetime.now().strftime("%H:%M:%S.%f"),
                    'elapsed_time': (datetime.now() - self.start_time).total_seconds(),
                    'left_eye_x': landmarks[self.LEFT_EYE].x,
                    'left_eye_y': landmarks[self.LEFT_EYE].y,
                    'right_eye_x': landmarks[self.RIGHT_EYE].x,
                    'right_eye_y': landmarks[self.RIGHT_EYE].y,
                    'nose_x': landmarks[self.NOSE_TIP].x,
                    'nose_y': landmarks[self.NOSE_TIP].y,
                    'chin_x': landmarks[self.CHIN].x,
                    'chin_y': landmarks[self.CHIN].y
                }
                self.movement_data.append(data)
            
            # 繪製關鍵點
            self.draw_landmarks(frame, landmarks)
            
            if self.is_recording:
                self.video_writer.write(frame)
                
        return frame
        
    def draw_landmarks(self, frame, landmarks):
        """繪製臉部關鍵點"""
        h, w = frame.shape[:2]
        for idx in [self.LEFT_EYE, self.RIGHT_EYE, self.NOSE_TIP, self.CHIN]:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
        if self.is_recording:
            cv2.putText(frame, "Recording...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                       
    def release(self):
        """釋放資源"""
        self.stop_recording()
        self.cap.release()