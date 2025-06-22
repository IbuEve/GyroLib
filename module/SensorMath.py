import numpy as np
import math
from collections import deque
import time

class SensorMath:
    """センサーデータの数学的変換を行う汎用クラス"""
    
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """クォータニオンを回転行列に変換"""
        if isinstance(q, dict):
            w, x, y, z = q['w'], q['x'], q['y'], q['z']
        else:
            w, x, y, z = q
            
        # 正規化
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
    
    @staticmethod
    def transform_to_global(local_vector, quaternion):
        """ローカル座標系からグローバル座標系に変換"""
        if isinstance(quaternion, dict):
            q_list = [quaternion['w'], quaternion['x'], quaternion['y'], quaternion['z']]
        else:
            q_list = quaternion
            
        R = SensorMath.quaternion_to_rotation_matrix(q_list)
        result = R @ np.array(local_vector)
        return result.tolist()
    
    @staticmethod
    def transform_to_local(global_vector, quaternion):
        """グローバル座標系からローカル座標系に変換"""
        if isinstance(quaternion, dict):
            w, x, y, z = quaternion['w'], quaternion['x'], quaternion['y'], quaternion['z']
        else:
            w, x, y, z = quaternion
            
        # 共役クォータニオンを使用してグローバル→ローカル変換
        q_conjugate = [w, -x, -y, -z]
        R = SensorMath.quaternion_to_rotation_matrix(q_conjugate)
        result = R @ np.array(global_vector)
        return result.tolist()
    
    @staticmethod
    def quaternion_to_euler_angles(quaternion):
        """クォータニオンをオイラー角に変換"""
        if isinstance(quaternion, dict):
            w, x, y, z = quaternion['w'], quaternion['x'], quaternion['y'], quaternion['z']
        else:
            w, x, y, z = quaternion
        
        # ロール (X軸周り)
        roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        
        # ピッチ (Y軸周り)
        sin_pitch = 2*(w*y - z*x)
        if abs(sin_pitch) >= 1:
            pitch = math.copysign(math.pi/2, sin_pitch)
        else:
            pitch = math.asin(sin_pitch)
        
        # ヨー (Z軸周り)
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        return roll, pitch, yaw
    
    @staticmethod
    def body_angular_velocity_to_euler_rates(local_gyro, quaternion):
        """ボディフレームの角速度をオイラー角の角速度に変換"""
        roll, pitch, yaw = SensorMath.quaternion_to_euler_angles(quaternion)
        
        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        tan_pitch = math.tan(pitch)
        
        # ジンバルロック回避
        if abs(cos_pitch) < 1e-6:
            cos_pitch = 1e-6 if cos_pitch >= 0 else -1e-6
        
        # 変換行列
        transformation_matrix = np.array([
            [1, sin_roll * tan_pitch, cos_roll * tan_pitch],
            [0, cos_roll, -sin_roll],
            [0, sin_roll / cos_pitch, cos_roll / cos_pitch]
        ])
        
        euler_rates = transformation_matrix @ np.array(local_gyro)
        return euler_rates.tolist()
    
    @staticmethod
    def get_gravity_corrected_angular_velocity(local_gyro, quaternion):
        """重力方向のみを考慮した角速度補正"""
        gravity_global = [0, 0, -1]
        gravity_local = SensorMath.transform_to_local(gravity_global, quaternion)
        
        gx, gy, gz = local_gyro
        
        # 簡易実装：Z軸（重力軸）周りの回転はそのまま
        vertical_rotation = gz
        horizontal_x = gx
        horizontal_y = gy
        
        return [horizontal_x, horizontal_y, vertical_rotation]
    
    @staticmethod
    def lowcut_integration(previous_velocity, current_accel, dt, q=0.99):
        """論文ベースのローカットフィルター付き積分"""
        return q * previous_velocity + dt * current_accel
    
    @staticmethod
    def calculate_kinetic_energy(velocity, angular_velocity, mass=1.0, moment_of_inertia=1.0):
        """剛体の運動エネルギーを計算"""
        # 並進運動エネルギー
        v_squared = sum(v**2 for v in velocity)
        translational_energy = 0.5 * mass * v_squared
        
        # 回転運動エネルギー
        omega_squared = sum(w**2 for w in angular_velocity)
        rotational_energy = 0.5 * moment_of_inertia * omega_squared
        
        total_energy = translational_energy + rotational_energy
        
        return {
            'total': total_energy,
            'translational': translational_energy,
            'rotational': rotational_energy
        }
    
    @staticmethod
    def trapezoidal_integration(current_velocity, current_accel, prev_accel, dt):
        """
        台形公式による速度積分
        
        Args:
            current_velocity: 現在の速度 [vx, vy, vz]
            current_accel: 現在の加速度 [ax, ay, az]
            prev_accel: 前回の加速度 [ax, ay, az] (None可)
            dt: 時間ステップ
            
        Returns:
            新しい速度 [vx, vy, vz]
        """
        decay_factor = 0.95
        v0 = np.array(current_velocity)
        a_current = np.array(current_accel)
        
        if prev_accel is None:
            # 初回時はオイラー法でフォールバック
            return (v0 + a_current * dt).tolist()
        
        a_prev = np.array(prev_accel)
        
        # 台形公式: v(t+dt) = v(t) + dt * (a(t) + a(t+dt)) / 2
        a_average = (a_prev + a_current) / 2
        new_velocity = decay_factor * v0 + dt * a_average
        
        return new_velocity.tolist()

from module.Filter import SavitzkyGolayFilter, MovingAverageFilter
class SensorDataProcessor:
    """センサーデータの処理と統合を行うクラス（閾値処理付き）"""
    
    def __init__(self, filter_type='sma', gyro_threshold=50.0, accel_threshold=0.15, **filter_params):
        self.last_time = None
        self.velocity = [0.0, 0.0, 0.0]  # [vx, vy, vz]
        self.filter_type = filter_type
        self.prev_accel = None  # RK4用
        self.accel_history = deque(maxlen=4)

        # 閾値設定
        self.gyro_threshold = gyro_threshold
        self.accel_threshold = accel_threshold
        
        if filter_type == 'savgol':
            window_length = filter_params.get('window_length', 7)
            polyorder = filter_params.get('polyorder', 2)
            self.filter = SavitzkyGolayFilter(window_length, polyorder)
            
        elif filter_type in ['sma', 'ema']:
            window_size = filter_params.get('window_size', 20)
            alpha = filter_params.get('alpha', 0.5)
            self.filter = MovingAverageFilter(window_size, alpha, filter_type)
            
        else:
            self.filter = None
    
    def _apply_gyro_threshold(self, gyro_values):
        """角速度に閾値処理を適用"""
        processed_gyro = []
        for value in gyro_values:
            if abs(value) < self.gyro_threshold:
                processed_gyro.append(0.0)
            else:
                processed_gyro.append(value)
        return processed_gyro
    
    def _apply_accel_threshold(self, accel_values):
        """加速度に閾値処理を適用"""
        processed_accel = []
        for value in accel_values:
            if abs(value) < self.accel_threshold:
                processed_accel.append(0.0)
            else:
                processed_accel.append(value)
        return processed_accel
        
    def process_sensor_data(self, parsed_data, current_time=None, filter_stage='raw'):
        """
        センサーデータを処理して統合結果を返す（閾値処理付き）
        
        Args:
            parsed_data: パースされたセンサーデータ
            current_time: 現在時刻（Noneの場合は自動取得）
        
        Returns:
            dict: 処理済みデータ
        """
        if current_time is None:
            current_time = time.time()
            
        if self.filter and filter_stage in ['raw', 'both']:
            parsed_data = self.filter.filter_raw_data(parsed_data)
        else:
            parsed_data = parsed_data
        
        # 生データの取得
        raw_local_accel = [
            parsed_data['acceleration']['x'],
            parsed_data['acceleration']['y'],
            parsed_data['acceleration']['z']
        ]
        raw_local_gyro = [
            parsed_data['gyroscope']['x'],
            parsed_data['gyroscope']['y'],
            parsed_data['gyroscope']['z']
        ]
        
        # 閾値処理を適用
        local_accel = self._apply_accel_threshold(raw_local_accel)
        local_gyro = self._apply_gyro_threshold(raw_local_gyro)
        
        quaternion = parsed_data['quaternion']
        
        # 座標変換（閾値処理済みデータを使用）
        global_accel = SensorMath.transform_to_global(local_accel, quaternion)
        euler_rates = SensorMath.body_angular_velocity_to_euler_rates(local_gyro, quaternion)
        gravity_corrected_gyro = SensorMath.get_gravity_corrected_angular_velocity(local_gyro, quaternion)
        
        # 速度計算（閾値処理済みの加速度を使用）
        if self.last_time is not None:
            dt = current_time - self.last_time
            self.velocity = SensorMath.trapezoidal_integration(
                self.velocity, global_accel, self.prev_accel, dt
            )
            self.prev_accel = global_accel.copy()
        self.last_time = current_time
        
        # エネルギー計算
        energy = SensorMath.calculate_kinetic_energy(self.velocity, euler_rates)
        
        return {
            'timestamp': current_time,
            'local_acceleration': local_accel,      # 閾値処理済み
            'global_acceleration': global_accel,    # 閾値処理済み
            'local_gyroscope': local_gyro,          # 閾値処理済み
            'raw_acceleration': raw_local_accel,    # 生データも保持
            'raw_gyroscope': raw_local_gyro,        # 生データも保持
            'euler_rates': euler_rates,
            'gravity_corrected_gyro': gravity_corrected_gyro,
            'velocity': self.velocity.copy(),
            'quaternion': quaternion,
            'button': parsed_data['button'],
            'energy': energy
        }
    
    def set_gyro_threshold(self, threshold):
        """角速度閾値を設定"""
        self.gyro_threshold = threshold
        print(f"角速度閾値を {threshold} に設定")
    
    def set_accel_threshold(self, threshold):
        """加速度閾値を設定"""
        self.accel_threshold = threshold
        print(f"加速度閾値を {threshold} に設定")
    
    def set_thresholds(self, gyro_threshold=None, accel_threshold=None):
        """両方の閾値を同時に設定"""
        if gyro_threshold is not None:
            self.gyro_threshold = gyro_threshold
        if accel_threshold is not None:
            self.accel_threshold = accel_threshold
        print(f"閾値設定: 角速度={self.gyro_threshold}, 加速度={self.accel_threshold}")
    
    def reset_velocity(self):
        """速度をリセット"""
        self.velocity = [0.0, 0.0, 0.0]