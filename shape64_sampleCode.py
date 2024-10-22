import cv2
import dlib
import numpy as np

# 初始化 dlib 的人臉檢測器和特徵點預測器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\plus1\\Desktop\\JowMoveDetection\\py20241010\\maindible\\shape_predictor_68_face_landmarks.dat')

# 定義所有連接關係
FACE_CONNECTIONS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # Jaw
    [17, 18, 19, 20, 21],  # Left eyebrow
    [22, 23, 24, 25, 26],  # Right eyebrow
    [27, 28, 29, 30],  # Nose bridge
    [30, 31, 32, 33, 34, 35],  # Lower nose
    [36, 37, 38, 39, 40, 41, 36],  # Left eye
    [42, 43, 44, 45, 46, 47, 42],  # Right eye
    [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48],  # Outer lip
    [60, 61, 62, 63, 64, 65, 66, 67, 60]  # Inner lip
]

# 打開攝像頭
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 設置寬度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 設置高度

# 創建一個命名視窗並設置為可調整大小
cv2.namedWindow('Facial Landmarks', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Facial Landmarks', 1280, 720)  # 設置視窗大小

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 將影像轉為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 檢測人臉
    faces = detector(gray)
    
    for face in faces:
        # 檢測特徵點
        landmarks = predictor(gray, face)
        
        # # 繪製所有特徵點，統一使用綠色
        # for i in range(68):
        #     x = landmarks.part(i).x
        #     y = landmarks.part(i).y
        #     cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # 增大點的大小為3
        
        # # 繪製所有連接線，統一使用綠色
        # for connection in FACE_CONNECTIONS:
        #     points = []
        #     for i in connection:
        #         point = landmarks.part(i)
        #         points.append([point.x, point.y])
        #     points = np.array(points, dtype=np.int32)
        #     cv2.polylines(frame, [points], False, (0, 255, 0), 2)  # 增加線條粗細為2
    ##

      # 顳顎關節的特徵點（例如：1 和 15）
        tmj_points = [1, 15]
        # 下顎輪廓特徵點（1 到 17）
        jaw_points = list(range(1, 17))
        # 繪製顳顎關節特徵點
        for pt in tmj_points:
            x = landmarks.part(pt).x
            y = landmarks.part(pt).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        # 繪製下顎輪廓
        for pt in jaw_points:
            x = landmarks.part(pt).x
            y = landmarks.part(pt).y
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            # 若要連接下顎輪廓點，可以加入線條
            if pt < 16:
                x2 = landmarks.part(pt + 1).x
                y2 = landmarks.part(pt + 1).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
    ##
    # 顯示結果
    cv2.imshow('Facial Landmarks', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # 按 'ESC' 鍵退出
        break

cap.release()
cv2.destroyAllWindows()