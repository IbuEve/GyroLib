import numpy as np
from collections import deque
import time
from scipy.signal import savgol_filter

class MovingAverageFilter:
    """移動平均フィルタクラス（単純移動平均 + 指数移動平均）"""
    
    def __init__(self, window_size=10, alpha=0.1, filter_type='sma'):
        """
        Args:
            window_size: 単純移動平均の窓サイズ
            alpha: 指数移動平均の平滑化係数 (0-1)
            filter_type: 'sma'(単純移動平均) or 'ema'(指数移動平均)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.filter_type = filter_type
        
        # 各データタイプ用のバッファ（SMA用）
        self.quaternion_buffers = {
            'w': deque(maxlen=window_size),
            'x': deque(maxlen=window_size),
            'y': deque(maxlen=window_size),
            'z': deque(maxlen=window_size)
        }
        self.accel_buffers = {
            'x': deque(maxlen=window_size),
            'y': deque(maxlen=window_size),
            'z': deque(maxlen=window_size)
        }
        self.gyro_buffers = {
            'x': deque(maxlen=window_size),
            'y': deque(maxlen=window_size),
            'z': deque(maxlen=window_size)
        }
        self.velocity_buffers = {
            'x': deque(maxlen=window_size),
            'y': deque(maxlen=window_size),
            'z': deque(maxlen=window_size)
        }
        
        # EMA用の前回値
        self.ema_values = {
            'quaternion': {'w': None, 'x': None, 'y': None, 'z': None},
            'acceleration': {'x': None, 'y': None, 'z': None},
            'gyroscope': {'x': None, 'y': None, 'z': None},
            'velocity': {'x': None, 'y': None, 'z': None}
        }
    
    def _apply_sma(self, buffer):
        """単純移動平均を適用"""
        if len(buffer) == 0:
            return 0.0
        return sum(buffer) / len(buffer)
    
    def _apply_ema(self, new_value, prev_value):
        """指数移動平均を適用"""
        if prev_value is None:
            return new_value
        return self.alpha * new_value + (1 - self.alpha) * prev_value
    
    def filter_raw_data(self, parsed_data):
        """生データ（クォータニオン、加速度、ジャイロ）をフィルタリング"""
        quat = parsed_data['quaternion']
        accel = parsed_data['acceleration']
        gyro = parsed_data['gyroscope']
        
        if self.filter_type == 'sma':
            # バッファに追加
            for axis in ['w', 'x', 'y', 'z']:
                if axis in quat:
                    self.quaternion_buffers[axis].append(quat[axis])
            
            for axis in ['x', 'y', 'z']:
                self.accel_buffers[axis].append(accel[axis])
                self.gyro_buffers[axis].append(gyro[axis])
            
            # SMA適用
            filtered_data = {
                'quaternion': {
                    'w': self._apply_sma(self.quaternion_buffers['w']),
                    'x': self._apply_sma(self.quaternion_buffers['x']),
                    'y': self._apply_sma(self.quaternion_buffers['y']),
                    'z': self._apply_sma(self.quaternion_buffers['z'])
                },
                'acceleration': {
                    'x': self._apply_sma(self.accel_buffers['x']),
                    'y': self._apply_sma(self.accel_buffers['y']),
                    'z': self._apply_sma(self.accel_buffers['z'])
                },
                'gyroscope': {
                    'x': self._apply_sma(self.gyro_buffers['x']),
                    'y': self._apply_sma(self.gyro_buffers['y']),
                    'z': self._apply_sma(self.gyro_buffers['z'])
                },
                'button': parsed_data['button']
            }
            
        elif self.filter_type == 'ema':
            # EMA適用
            filtered_data = {
                'quaternion': {
                    'w': self._apply_ema(quat['w'], self.ema_values['quaternion']['w']),
                    'x': self._apply_ema(quat['x'], self.ema_values['quaternion']['x']),
                    'y': self._apply_ema(quat['y'], self.ema_values['quaternion']['y']),
                    'z': self._apply_ema(quat['z'], self.ema_values['quaternion']['z'])
                },
                'acceleration': {
                    'x': self._apply_ema(accel['x'], self.ema_values['acceleration']['x']),
                    'y': self._apply_ema(accel['y'], self.ema_values['acceleration']['y']),
                    'z': self._apply_ema(accel['z'], self.ema_values['acceleration']['z'])
                },
                'gyroscope': {
                    'x': self._apply_ema(gyro['x'], self.ema_values['gyroscope']['x']),
                    'y': self._apply_ema(gyro['y'], self.ema_values['gyroscope']['y']),
                    'z': self._apply_ema(gyro['z'], self.ema_values['gyroscope']['z'])
                },
                'button': parsed_data['button']
            }
            
            # EMA値を更新
            self.ema_values['quaternion']['w'] = filtered_data['quaternion']['w']
            self.ema_values['quaternion']['x'] = filtered_data['quaternion']['x']
            self.ema_values['quaternion']['y'] = filtered_data['quaternion']['y']
            self.ema_values['quaternion']['z'] = filtered_data['quaternion']['z']
            self.ema_values['acceleration']['x'] = filtered_data['acceleration']['x']
            self.ema_values['acceleration']['y'] = filtered_data['acceleration']['y']
            self.ema_values['acceleration']['z'] = filtered_data['acceleration']['z']
            self.ema_values['gyroscope']['x'] = filtered_data['gyroscope']['x']
            self.ema_values['gyroscope']['y'] = filtered_data['gyroscope']['y']
            self.ema_values['gyroscope']['z'] = filtered_data['gyroscope']['z']
        
        return filtered_data
    
    def filter_velocity(self, velocity):
        """速度データをフィルタリング"""
        if self.filter_type == 'sma':
            for i, axis in enumerate(['x', 'y', 'z']):
                self.velocity_buffers[axis].append(velocity[i])
            
            return [
                self._apply_sma(self.velocity_buffers['x']),
                self._apply_sma(self.velocity_buffers['y']),
                self._apply_sma(self.velocity_buffers['z'])
            ]
            
        elif self.filter_type == 'ema':
            filtered_velocity = [
                self._apply_ema(velocity[0], self.ema_values['velocity']['x']),
                self._apply_ema(velocity[1], self.ema_values['velocity']['y']),
                self._apply_ema(velocity[2], self.ema_values['velocity']['z'])
            ]
            
            # EMA値を更新
            self.ema_values['velocity']['x'] = filtered_velocity[0]
            self.ema_values['velocity']['y'] = filtered_velocity[1]
            self.ema_values['velocity']['z'] = filtered_velocity[2]
            
            return filtered_velocity
        

class SavitzkyGolayFilter:
    """Savitzky-Golayフィルタクラス"""
    
    def __init__(self, window_length=7, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder
        
        # 各データタイプ用のバッファ
        self.quaternion_buffers = {
            'w': deque(maxlen=window_length),
            'x': deque(maxlen=window_length),
            'y': deque(maxlen=window_length),
            'z': deque(maxlen=window_length)
        }
        self.accel_buffers = {
            'x': deque(maxlen=window_length),
            'y': deque(maxlen=window_length),
            'z': deque(maxlen=window_length)
        }
        self.gyro_buffers = {
            'x': deque(maxlen=window_length),
            'y': deque(maxlen=window_length),
            'z': deque(maxlen=window_length)
        }
        self.velocity_buffers = {
            'x': deque(maxlen=window_length),
            'y': deque(maxlen=window_length),
            'z': deque(maxlen=window_length)
        }
    
    def _apply_filter(self, buffer):
        """単一バッファにフィルタを適用"""
        if len(buffer) < self.window_length:
            return buffer[-1] if buffer else 0.0
        
        data = np.array(buffer)
        filtered = savgol_filter(data, self.window_length, self.polyorder)
        return filtered[-1]  # 最新値を返す
    
    def filter_raw_data(self, parsed_data):
        """生データ（クォータニオン、加速度、ジャイロ）をフィルタリング"""
        # バッファに追加
        quat = parsed_data['quaternion']
        accel = parsed_data['acceleration']
        gyro = parsed_data['gyroscope']
        
        for axis in ['w', 'x', 'y', 'z']:
            if axis in quat:
                self.quaternion_buffers[axis].append(quat[axis])
        
        for axis in ['x', 'y', 'z']:
            self.accel_buffers[axis].append(accel[axis])
            self.gyro_buffers[axis].append(gyro[axis])
        
        # フィルタ適用
        filtered_data = {
            'quaternion': {
                'w': self._apply_filter(self.quaternion_buffers['w']),
                'x': self._apply_filter(self.quaternion_buffers['x']),
                'y': self._apply_filter(self.quaternion_buffers['y']),
                'z': self._apply_filter(self.quaternion_buffers['z'])
            },
            'acceleration': {
                'x': self._apply_filter(self.accel_buffers['x']),
                'y': self._apply_filter(self.accel_buffers['y']),
                'z': self._apply_filter(self.accel_buffers['z'])
            },
            'gyroscope': {
                'x': self._apply_filter(self.gyro_buffers['x']),
                'y': self._apply_filter(self.gyro_buffers['y']),
                'z': self._apply_filter(self.gyro_buffers['z'])
            },
            'button': parsed_data['button']
        }
        
        return filtered_data
    
    def filter_velocity(self, velocity):
        """速度データをフィルタリング"""
        for i, axis in enumerate(['x', 'y', 'z']):
            self.velocity_buffers[axis].append(velocity[i])
        
        return [
            self._apply_filter(self.velocity_buffers['x']),
            self._apply_filter(self.velocity_buffers['y']),
            self._apply_filter(self.velocity_buffers['z'])
        ]