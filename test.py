import sys
import cv2
import numpy as np
import insightface
import onnxruntime

def test_environment():
    """測試所有必要的套件是否正確安裝和運作"""
    
    print("Python 版本:", sys.version)
    print("\n套件版本:")
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("ONNX Runtime:", onnxruntime.__version__)
    print("InsightFace:", insightface.__version__)
    
    print("\n測試攝像頭...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("攝像頭工作正常")
            # 測試numpy陣列操作
            print(f"影像大小: {frame.shape}")
        else:
            print("無法讀取影像")
        cap.release()
    else:
        print("無法開啟攝像頭")
    
    print("\n測試InsightFace...")
    try:
        app = insightface.app.FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace 初始化成功")
    except Exception as e:
        print("InsightFace 初始化失敗:", str(e))

if __name__ == "__main__":
    test_environment()