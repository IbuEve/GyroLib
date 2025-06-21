import numpy as np
import math
from collections import deque

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
    def lowcut_integration(previous_velocity, current_accel, dt, q=0.95):
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

class SensorDataProcessor:
    """センサーデータの処理と統合を行うクラス"""
    
    def __init__(self):
        self.last_time = None
        self.velocity = [0.0, 0.0, 0.0]  # [vx, vy, vz]
        
    def process_sensor_data(self, parsed_data, current_time=None):
        """
        センサーデータを処理して統合結果を返す
        
        Args:
            parsed_data: パースされたセンサーデータ
            current_time: 現在時刻（Noneの場合は自動取得）
        
        Returns:
            dict: 処理済みデータ
        """
        if current_time is None:
            current_time = time.time()
            
        # 入力データの取得
        local_accel = [
            parsed_data['acceleration']['x'],
            parsed_data['acceleration']['y'],
            parsed_data['acceleration']['z']
        ]
        local_gyro = [
            parsed_data['gyroscope']['x'],
            parsed_data['gyroscope']['y'],
            parsed_data['gyroscope']['z']
        ]
        quaternion = parsed_data['quaternion']
        
        # 座標変換
        global_accel = SensorMath.transform_to_global(local_accel, quaternion)
        euler_rates = SensorMath.body_angular_velocity_to_euler_rates(local_gyro, quaternion)
        gravity_corrected_gyro = SensorMath.get_gravity_corrected_angular_velocity(local_gyro, quaternion)
        
        # 速度計算
        if self.last_time is not None:
            dt = current_time - self.last_time
            self.velocity[0] = SensorMath.lowcut_integration(self.velocity[0], global_accel[0], dt)
            self.velocity[1] = SensorMath.lowcut_integration(self.velocity[1], global_accel[1], dt)
            self.velocity[2] = SensorMath.lowcut_integration(self.velocity[2], global_accel[2], dt)
        
        self.last_time = current_time
        
        # エネルギー計算
        energy = SensorMath.calculate_kinetic_energy(self.velocity, euler_rates)
        
        return {
            'timestamp': current_time,
            'local_acceleration': local_accel,
            'global_acceleration': global_accel,
            'local_gyroscope': local_gyro,
            'euler_rates': euler_rates,
            'gravity_corrected_gyro': gravity_corrected_gyro,
            'velocity': self.velocity.copy(),
            'quaternion': quaternion,
            'button': parsed_data['button'],
            'energy': energy
        }
    
    def reset_velocity(self):
        """速度をリセット"""
        self.velocity = [0.0, 0.0, 0.0]