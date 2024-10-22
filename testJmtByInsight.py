import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np

def initialize_face_analyzer():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def track_jaw():
    cap = cv2.VideoCapture(0)
    face_analyzer = initialize_face_analyzer()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        faces = face_analyzer.get(frame)
        
        for face in faces:
            landmarks = face.landmark_2d_106
            # 下顎的關鍵點
            jaw_points = landmarks[:17]
            
            # 把點轉成整數座標的array
            points = np.array([point.astype(np.int32) for point in jaw_points])
            
            # 畫點
            for point in points:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
            
            # 直接用points畫線
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, (0, 255, 0), 2)
            
        cv2.imshow('Jaw Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_jaw()