import cv2
import dlib

# 加載dlib的人臉檢測器
detector = dlib.get_frontal_face_detector()

# 加載圖像
image = cv2.imread("path/to/your/image.jpg")  # 將此路徑替換為您的圖片路徑
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用dlib檢測人臉
faces = detector(gray)

# 在圖像上繪製矩形框
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 顯示結果
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
