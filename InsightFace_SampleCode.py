import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np

def initialize_face_analyzer():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def track_landmarks():
    cap = cv2.VideoCapture(0)
    face_analyzer = initialize_face_analyzer()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces = face_analyzer.get(frame)
        
        for face in faces:
            landmarks = face.landmark_2d_106
            
            # 顯示所有點和它們的索引
            for i, point in enumerate(landmarks):
                pt = tuple(point.astype(int))
                # 畫點
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)
                # 顯示點的索引編號
                cv2.putText(frame, str(i), pt, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                          (255, 0, 0), 1)
            
        cv2.imshow('All Landmarks', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_landmarks()