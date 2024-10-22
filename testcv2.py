import cv2
import numpy as np

# 創建一個簡單的圖像
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.putText(img, "Test", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 嘗試顯示
try:
    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.imshow("Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"錯誤: {e}")