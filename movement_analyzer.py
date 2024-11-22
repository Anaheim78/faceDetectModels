# movement_analyzer.py
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd

class MovementAnalyzer:
    def __init__(self):
        self.fps = 30
        self.pixel_to_mm = 0.264583
        self.window_size = 5

    def calculate_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def calculate_uci(self, chin_x_coords):
        """計算單側咀嚼指數 (UCI)"""
        # 計算左右側移動
        movements = np.diff(chin_x_coords)
        right_moves = np.sum(movements > 0)
        left_moves = np.sum(movements < 0)
        
        if (right_moves + left_moves) == 0:
            return 0
            
        return (right_moves - left_moves) / (right_moves + left_moves)

    def calculate_hssi(self, chin_x_coords):
        """計算習慣性側咬指數 (HSSI)"""
        movements = np.diff(chin_x_coords)
        score = 0
        consecutive_right = 0
        consecutive_left = 0
        
        for mov in movements:
            if mov > 0:  # 右側移動
                consecutive_right += 1
                if consecutive_left > 0:
                    score += 0  # 左側切換
                consecutive_left = 0
            elif mov < 0:  # 左側移動
                consecutive_left += 1
                consecutive_right = 0
                
        # 評分
        if consecutive_right > consecutive_left:
            if consecutive_left == 0:
                score += 1  # 純右側
            else:
                score += 0.5  # 右側為主
                
        return score / len(movements) if len(movements) > 0 else 0

    def analyze_motion_type(self, x_coords, y_coords):
        """分析運動類型（圓周/直線）"""
        linear_count = 0
        circular_count = 0
        
        for i in range(2, len(x_coords)):
            v1 = np.array([x_coords[i-1] - x_coords[i-2], 
                          y_coords[i-1] - y_coords[i-2]])
            v2 = np.array([x_coords[i] - x_coords[i-1], 
                          y_coords[i] - y_coords[i-1]])
            
            dot_product = np.dot(v1, v2)
            cross_product = np.cross(v1, v2)
            angle = np.abs(np.degrees(np.arctan2(cross_product, dot_product)))
            
            if angle < 30:
                linear_count += 1
            else:
                circular_count += 1

        total_count = linear_count + circular_count
        if total_count == 0:
            return 0, 0, 0

        linear_ratio = linear_count / total_count * 100
        circular_ratio = circular_count / total_count * 100
        circular_frequency = circular_count / (total_count / self.fps)

        return linear_ratio, circular_ratio, circular_frequency

    def detect_chewing_cycles(self, df):
        """檢測咀嚼週期"""
        eye_distance = self.calculate_distance(
            df['left_eye_x'], df['left_eye_y'],
            df['right_eye_x'], df['right_eye_y']
        )
        
        nose_chin_distance = self.calculate_distance(
            df['nose_x'], df['nose_y'],
            df['chin_x'], df['chin_y']
        )
        
        ratio = nose_chin_distance / eye_distance
        smoothed_ratio = self.moving_average(ratio, self.window_size)
        
        cycles = []
        cycle_start = None
        in_cycle = False
        threshold = np.mean(smoothed_ratio)
        
        for i in range(1, len(smoothed_ratio)):
            if not in_cycle and smoothed_ratio[i] > threshold:
                cycle_start = i
                in_cycle = True
            elif in_cycle and smoothed_ratio[i] < threshold:
                if cycle_start is not None:
                    cycles.append({
                        'start': cycle_start,
                        'end': i,
                        'duration': (i - cycle_start) / self.fps
                    })
                in_cycle = False
        
        return cycles

    def moving_average(self, data, window_size):
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean()

    def analyze_movement(self, movement_data):
        df = pd.DataFrame(movement_data)
        
        cycles = self.detect_chewing_cycles(df)
        cycle_durations = [cycle['duration'] for cycle in cycles]
        total_time = df['elapsed_time'].max()
        
        linear_ratio, circular_ratio, circular_freq = self.analyze_motion_type(
            df['chin_x'].values, 
            df['chin_y'].values
        )
        
        # 計算UCI和HSSI
        uci = self.calculate_uci(df['chin_x'].values)
        hssi = self.calculate_hssi(df['chin_x'].values)
        
        results = {
            '咀嚼時間(s)': total_time,
            '循環次數(n)': len(cycles),
            '循環頻率(n/s)': len(cycles) / total_time if total_time > 0 else 0,
            '平均週期時間(s)': np.mean(cycle_durations) if cycle_durations else 0,
            '直線運動比例(%)': linear_ratio,
            '圓周運動比例(%)': circular_ratio,
            '圓周運動頻率(Hz)': circular_freq,
            '單側咀嚼指數(UCI)': uci,
            '習慣性側咬指數(HSSI)': hssi
        }
        
        return pd.DataFrame([results])