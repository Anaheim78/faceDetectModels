# main_window.py
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                           QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                           QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from face_tracker import FaceTracker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('咀嚼行為分析')
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        self.initTracker()
        
    def initUI(self):
        """初始化使用者介面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # 左側顯示區域
        self.display_area = QLabel()
        self.display_area.setStyleSheet("background-color: black;")
        self.display_area.setMinimumSize(800, 600)
        layout.addWidget(self.display_area)

        # 右側控制區域
        button_layout = QVBoxLayout()
        
        self.start_button = QPushButton('開始錄影')
        self.stop_button = QPushButton('停止錄影')
        self.stop_button.setEnabled(False)
        
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        
        button_container = QWidget()
        button_container.setLayout(button_layout)
        layout.addWidget(button_container)

    def initTracker(self):
        """初始化臉部追蹤"""
        self.tracker = FaceTracker()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms ≈ 33fps

    def start_recording(self):
        """開始錄影"""
        video_path = self.tracker.start_recording()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        QMessageBox.information(self, '開始錄影', 
                              f'錄影已開始\n儲存至: {video_path}')

    def stop_recording(self):
        """停止錄影"""
        video_path, excel_path, analysis_path = self.tracker.stop_recording()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if all([video_path, excel_path, analysis_path]):
            QMessageBox.information(
                self, 
                '停止錄影',
                f'錄影已停止\n'
                f'影片: {video_path}\n'
                f'原始數據: {excel_path}\n'
                f'分析報告: {analysis_path}'
            )

    def update_frame(self):
        """更新畫面"""
        frame = self.tracker.get_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.display_area.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.display_area.size(), Qt.KeepAspectRatio))
    
    def closeEvent(self, event):
        """關閉視窗事件"""
        self.tracker.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())