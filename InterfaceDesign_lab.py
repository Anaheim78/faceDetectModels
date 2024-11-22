import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import deque
import time
import os

# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

# Global variables
measuring = False
start_button = None
end_button = None
reset_button = None
save_button = None
jaw_movements = deque(maxlen=100)
deviation_history = deque(maxlen=100)
time_history = deque(maxlen=100)
max_mouth_opening = 0
min_mouth_opening = float('inf')
max_deviation = 0

# Thresholds
DEVIATION_THRESHOLD = 0.5  # percentage
OPENING_THRESHOLD = 30  # percentage, threshold for large mouth opening

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color=(0, 255, 0)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color

    def draw(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), self.color, -1)
        cv2.putText(img, self.text, (self.x + 5, self.y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def is_clicked(self, x, y):
        return self.x < x < self.x + self.width and self.y < y < self.y + self.height

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global measuring, start_button, end_button, reset_button, save_button, jaw_movements, deviation_history, time_history, max_mouth_opening, min_mouth_opening, max_deviation
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_button.is_clicked(x, y):
            measuring = True
            print("Start measuring")
        elif end_button.is_clicked(x, y):
            measuring = False
            print("End measuring")
        elif reset_button.is_clicked(x, y):
            jaw_movements.clear()
            deviation_history.clear()
            time_history.clear()
            max_mouth_opening = 0
            min_mouth_opening = float('inf')
            max_deviation = 0
            print("Reset data")
        elif save_button.is_clicked(x, y):
            save_image(param)

def detect_teeth_edges(frame, face_landmarks):
    try:
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get mouth region
        mouth_top = max(0, int(face_landmarks[13].y * frame.shape[0]))
        mouth_bottom = min(frame.shape[0], int(face_landmarks[14].y * frame.shape[0]))
        mouth_left = max(0, int(face_landmarks[78].x * frame.shape[1]))
        mouth_right = min(frame.shape[1], int(face_landmarks[308].x * frame.shape[1]))
        
        # Check if mouth region is valid
        if mouth_top >= mouth_bottom or mouth_left >= mouth_right:
            return None, None

        # Apply edge detection to mouth region
        mouth_roi = gray[mouth_top:mouth_bottom, mouth_left:mouth_right]
        edges = cv2.Canny(mouth_roi, 100, 200)
        
        # Find the upper and lower teeth edges
        upper_half = edges[:edges.shape[0]//2]
        lower_half = edges[edges.shape[0]//2:]
        
        if upper_half.size > 0 and upper_half.sum(axis=1).size > 0:
            upper_edge = mouth_top + np.argmax(upper_half.sum(axis=1))
        else:
            upper_edge = mouth_top

        if lower_half.size > 0 and lower_half.sum(axis=1).size > 0:
            lower_edge = mouth_top + edges.shape[0]//2 + np.argmax(lower_half.sum(axis=1))
        else:
            lower_edge = mouth_bottom
        
        return upper_edge, lower_edge
    except Exception as e:
        print(f"Error in detect_teeth_edges: {e}")
        return None, None

def calculate_relative_jaw_deviation(landmarks):
    # 定義固定點（例如：兩眼之間的點）
    left_eye = np.array([landmarks[33].x, landmarks[33].y])
    right_eye = np.array([landmarks[263].x, landmarks[263].y])
    fixed_point = (left_eye + right_eye) / 2

    # 定義下巴點
    chin_point = np.array([landmarks[152].x, landmarks[152].y])

    # 計算面部中線向量
    face_midline = np.array([0, 1])  # 垂直向下的單位向量

    # 計算下巴點到固定點的向量
    chin_vector = chin_point - fixed_point

    # 將chin_vector投影到面部中線上
    projection = np.dot(chin_vector, face_midline) * face_midline

    # 計算偏移向量（投影向量與實際向量之間的差）
    deviation_vector = chin_vector - projection

    # 計算偏移量（向量的x分量）和方向
    deviation_amount = deviation_vector[0]
    deviation_direction = "右" if deviation_amount > 0 else "左"

    return deviation_amount, deviation_direction

def calculate_metrics(frame, landmarks):
    try:
        # Calculate mouth opening
        upper_lip = landmarks[13]  # Upper lip center
        lower_lip = landmarks[14]  # Lower lip center
        left_mouth = landmarks[78]  # Left mouth corner
        right_mouth = landmarks[308]  # Right mouth corner
        
        mouth_height = lower_lip.y - upper_lip.y
        mouth_width = right_mouth.x - left_mouth.x
        opening_ratio = (mouth_height / mouth_width) * 100 if mouth_width > 0 else 0

        # Detect teeth edges
        upper_edge, lower_edge = detect_teeth_edges(frame, landmarks)
        
        # Calculate relative jaw deviation
        deviation_amount, deviation_direction = calculate_relative_jaw_deviation(landmarks)
        
        if upper_edge is None or lower_edge is None:
            return opening_ratio, deviation_amount, deviation_direction, None, None

        # Calculate upper and lower jaw lines
        philtrum_point = landmarks[164]  # Philtrum point
        chin_point = landmarks[152]  # Chin center point
        upper_jaw_line = (philtrum_point, (upper_lip.x, upper_edge / frame.shape[0]))
        lower_jaw_line = ((lower_lip.x, lower_edge / frame.shape[0]), chin_point)
        
        return opening_ratio, deviation_amount, deviation_direction, upper_jaw_line, lower_jaw_line
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        return 0, 0, "N/A", None, None

# Draw waveforms (fixed Y-axis range)
def draw_waveforms(jaw_data, deviation_data, time_data, size=(320, 720)):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(size[0]/100, size[1]/100), dpi=100)
    
    # Upper graph: Jaw opening
    ax1.plot(time_data, jaw_data)
    ax1.set_title('Jaw Opening Trajectory')
    ax1.set_ylabel('Opening Ratio (%)')
    ax1.set_ylim(0, 135)  # Fixed Y-axis range
    
    # Lower graph: Deviation trajectory
    ax2.plot(time_data, deviation_data)
    ax2.set_title('Deviation Trajectory')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Deviation (%)')
    ax2.set_ylim(-15, 15)  # Fixed Y-axis range to ±15%
    ax2.axhline(y=0, color='r', linestyle='--')
    
    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape((size[1], size[0], 3))
    plt.close(fig)
    return img

def save_image(img):
    if not os.path.exists('saved_images'):
        os.makedirs('saved_images')
    filename = f'saved_images/face_analysis_{time.strftime("%Y%m%d-%H%M%S")}.png'
    cv2.imwrite(filename, img)
    print(f"Image saved as {filename}")

def main():
    global start_button, end_button, reset_button, save_button, max_mouth_opening, min_mouth_opening, max_deviation

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        cv2.namedWindow('Face Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Analysis', 1280, 720)
        cv2.setMouseCallback('Face Analysis', mouse_callback, param=None)

        start_button = Button(10, 10, 80, 30, "Start", (0, 255, 0))
        end_button = Button(100, 10, 80, 30, "End", (0, 0, 255))
        reset_button = Button(190, 10, 80, 30, "Reset", (255, 255, 0))
        save_button = Button(280, 10, 80, 30, "Save", (255, 0, 255))

        start_time = time.time()
        last_time = start_time

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame")
                break

            frame = cv2.resize(frame, (960, 720))
            
            try:
                current_time = time.time() - start_time
                fps = int(1 / (current_time - last_time)) if current_time != last_time else 0
                last_time = current_time

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                face_landmarker_result = landmarker.detect(mp_image)
                
                display = np.zeros((720, 1280, 3), dtype=np.uint8)
                display[0:720, 0:960] = frame

                # Draw Frankfurt plane guide lines
                cv2.line(display, (0, 360), (960, 360), (255, 255, 255, 128), 1)
                cv2.line(display, (480, 0), (480, 720), (255, 255, 255, 128), 1)

                if not face_landmarker_result.face_landmarks:
                    cv2.putText(display, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    face_landmarks = face_landmarker_result.face_landmarks[0]
                    
                    # Draw facial feature points
                    for landmark in face_landmarks:
                        x = int(landmark.x * 960)
                        y = int(landmark.y * 720)
                        cv2.circle(display, (x, y), 2, (0, 255, 0), -1)

                    # Calculate metrics
                    opening_ratio, deviation_amount, deviation_direction, upper_jaw_line, lower_jaw_line = calculate_metrics(frame, face_landmarks)
                    
                    # Draw upper and lower jaw lines
                    if upper_jaw_line and lower_jaw_line:
                        cv2.line(display, 
                                 (int(upper_jaw_line[0].x * 960), int(upper_jaw_line[0].y * 720)),
                                 (int(upper_jaw_line[1][0] * 960), int(upper_jaw_line[1][1] * 720)),
                                 (255, 0, 0), 2)
                        cv2.line(display, 
                                 (int(lower_jaw_line[0][0] * 960), int(lower_jaw_line[0][1] * 720)),
                                 (int(lower_jaw_line[1].x * 960), int(lower_jaw_line[1].y * 720)),
                                 (0, 0, 255), 2)
                    
                    if measuring:
                        jaw_movements.append(opening_ratio)
                        time_history.append(current_time)
                        if opening_ratio > OPENING_THRESHOLD:
                            deviation_history.append(deviation_amount * 100)  # Convert to percentage
                        else:
                            deviation_history.append(0)
                        max_mouth_opening = max(max_mouth_opening, opening_ratio)
                        min_mouth_opening = min(min_mouth_opening, opening_ratio)
                        max_deviation = max(max_deviation, abs(deviation_amount * 100))  # Convert to percentage

                    # Right side: waveforms
                    if jaw_movements and deviation_history:
                        waveforms = draw_waveforms(list(jaw_movements), list(deviation_history), list(time_history), (320, 720))
                        display[0:720, 960:1280] = waveforms

                    # Create semi-transparent background
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 480), (400, 720), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)


                        # Display data
                    info_text = [
                        f"Max Opening: {max_mouth_opening:.2f}%",
                        f"Min Opening: {min_mouth_opening:.2f}%",
                        f"Current Opening: {opening_ratio:.2f}%",
                        f"Current Deviation: {deviation_amount * 100:.2f}%",
                        f"Max Deviation: {max_deviation:.2f}%",
                        f"Deviation Direction: {deviation_direction}"
                    ]
                    for i, text in enumerate(info_text):
                        cv2.putText(display, text, (10, 500 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Draw buttons
                start_button.draw(display)
                end_button.draw(display)
                reset_button.draw(display)
                save_button.draw(display)

                # Display FPS
                cv2.putText(display, f"FPS: {fps}", (880, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.setMouseCallback('Face Analysis', mouse_callback, param=display)
                cv2.imshow('Face Analysis', display)

            except Exception as e:
                print(f"Processing error: {e}")

            if cv2.waitKey(10) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()