import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import threading
import time

class SensorDataVisualizer:
    """センサーデータの可視化クラス"""
    
    def __init__(self, max_points=500):
        self.max_points = max_points
        self.data_lock = threading.Lock()
        self.data_updated = False
        
        # データ保存用
        self._init_data_storage()
        
        # グラフ設定
        self.fig = None
        self.axes = None
        self.animation = None
    
    def _init_data_storage(self):
        """データ保存用のdequeを初期化"""
        self.timestamps = deque(maxlen=self.max_points)
        self.global_accel_x = deque(maxlen=self.max_points)
        self.global_accel_y = deque(maxlen=self.max_points)
        self.global_accel_z = deque(maxlen=self.max_points)
        self.global_vel_x = deque(maxlen=self.max_points)
        self.global_vel_y = deque(maxlen=self.max_points)
        self.global_vel_z = deque(maxlen=self.max_points)
        self.global_gyro_x = deque(maxlen=self.max_points)
        self.global_gyro_y = deque(maxlen=self.max_points)
        self.global_gyro_z = deque(maxlen=self.max_points)
        self.local_gyro_x = deque(maxlen=self.max_points)
        self.local_gyro_y = deque(maxlen=self.max_points)
        self.local_gyro_z = deque(maxlen=self.max_points)
        self.energy_total = deque(maxlen=self.max_points)
    
    def add_data(self, processed_data):
        """処理済みデータを追加"""
        with self.data_lock:
            self.timestamps.append(processed_data['timestamp'])
            
            # 加速度
            global_accel = processed_data['global_acceleration']
            self.global_accel_x.append(global_accel[0])
            self.global_accel_y.append(global_accel[1])
            self.global_accel_z.append(global_accel[2])
            
            # 速度
            velocity = processed_data['velocity']
            self.global_vel_x.append(velocity[0])
            self.global_vel_y.append(velocity[1])
            self.global_vel_z.append(velocity[2])
            
            # 角速度（オイラー角レート）
            euler_rates = processed_data['euler_rates']
            self.global_gyro_x.append(euler_rates[0])
            self.global_gyro_y.append(euler_rates[1])
            self.global_gyro_z.append(euler_rates[2])
            
            # ローカル角速度
            local_gyro = processed_data['local_gyroscope']
            self.local_gyro_x.append(local_gyro[0])
            self.local_gyro_y.append(local_gyro[1])
            self.local_gyro_z.append(local_gyro[2])
            
            # エネルギー
            self.energy_total.append(processed_data['energy']['total'])
            
            self.data_updated = True
    
    def setup_plots(self, plot_config=None):
        """グラフの設定"""
        if plot_config is None:
            plot_config = {
                'rows': 4,
                'titles': [
                    'Global Acceleration (m/s²)',
                    'Global Velocity (m/s) - with Lowcut Filter',
                    'Euler Angular Rates (rad/s)',
                    'Local (Sensor) Angular Velocity (rad/s)'
                ],
                'ylims': [[-20, 20], [-5, 5], [-400, 400], [-400, 400]]
            }
        
        self.fig, self.axes = plt.subplots(
            plot_config['rows'], 1, 
            figsize=(12, 3 * plot_config['rows'])
        )
        self.fig.suptitle('GyroStick Time Series Data', fontsize=14)
        
        for i, (ax, title) in enumerate(zip(self.axes, plot_config['titles'])):
            ax.set_title(title)
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend(['X', 'Y', 'Z'])
            if i < len(plot_config['ylims']):
                ax.set_ylim(plot_config['ylims'][i])
        
        self.axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        return self.fig, self.axes
    
    def _update_plots(self, frame):
        """グラフ更新（内部メソッド）"""
        with self.data_lock:
            if not self.data_updated or len(self.timestamps) == 0:
                return
            
            # データをコピー
            time_data = list(self.timestamps)
            data_sets = [
                [list(self.global_accel_x), list(self.global_accel_y), list(self.global_accel_z)],
                [list(self.global_vel_x), list(self.global_vel_y), list(self.global_vel_z)],
                [list(self.global_gyro_x), list(self.global_gyro_y), list(self.global_gyro_z)],
                [list(self.local_gyro_x), list(self.local_gyro_y), list(self.local_gyro_z)]
            ]
            self.data_updated = False
        
        if time_data:
            start_time = time_data[0]
            relative_time = [(t - start_time) for t in time_data]
            
            colors = ['r-', 'g-', 'b-']
            labels = ['X', 'Y', 'Z']
            
            for ax, data_set in zip(self.axes, data_sets):
                ax.clear()
                for i, (data, color, label) in enumerate(zip(data_set, colors, labels)):
                    ax.plot(relative_time, data, color, label=label, linewidth=1.5)
                
                ax.grid(True)
                ax.legend()
            
            # タイトル更新
            self.axes[0].set_title('Global Acceleration (m/s²)')
            self.axes[1].set_title('Global Velocity (m/s) - with Lowcut Filter')
            self.axes[2].set_title('Euler Angular Rates (rad/s)')
            self.axes[3].set_title('Local (Sensor) Angular Velocity (rad/s)')
            
            # Y軸範囲設定
            ylims = [[-20, 20], [-5, 5], [-400, 400], [-400, 400]]
            for ax, ylim in zip(self.axes, ylims):
                ax.set_ylim(ylim)
            
            self.axes[-1].set_xlabel('Time (s)')
            self.fig.suptitle(f'GyroStick Time Series Data - Points: {len(time_data)}', fontsize=14)
    
    def start_animation(self, interval=100):
        """アニメーション開始"""
        if self.fig is None or self.axes is None:
            self.setup_plots()
        
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plots, 
            interval=interval, cache_frame_data=False
        )
        
        return self.animation
    
    def show(self):
        """グラフを表示"""
        plt.show()
    
    def save_data(self, filename):
        """データをファイルに保存"""
        import json
        
        with self.data_lock:
            data_dict = {
                'timestamps': list(self.timestamps),
                'global_acceleration': {
                    'x': list(self.global_accel_x),
                    'y': list(self.global_accel_y),
                    'z': list(self.global_accel_z)
                },
                'global_velocity': {
                    'x': list(self.global_vel_x),
                    'y': list(self.global_vel_y),
                    'z': list(self.global_vel_z)
                },
                'euler_rates': {
                    'x': list(self.global_gyro_x),
                    'y': list(self.global_gyro_y),
                    'z': list(self.global_gyro_z)
                },
                'local_gyroscope': {
                    'x': list(self.local_gyro_x),
                    'y': list(self.local_gyro_y),
                    'z': list(self.local_gyro_z)
                },
                'energy': list(self.energy_total)
            }
        
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        print(f"Data saved to {filename}")