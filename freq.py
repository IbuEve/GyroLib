import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from module.SensorMath import SensorDataProcessor
from module.SensorReceiver import GyroStickReceiver
from module.MotionAnalyzer import FrequencyAnalyzer  # 既存のFrequencyAnalyzerをインポート

class FrequencyGraphMonitorWithAnalyzer:
    """MotionAnalyzerを使用するリアルタイム周波数グラフ監視"""
    
    def __init__(self):
        # 受信・処理
        self.receiver = GyroStickReceiver()
        self.processor = SensorDataProcessor()
        
        # 既存のFrequencyAnalyzerを使用
        self.frequency_analyzer = FrequencyAnalyzer(
            thresholds={'prominence': 0.1, 'entropy': 0.8},  # 感度を上げる
            sampling_rate=100
        )
        
        # グラフ用データバッファ
        self.time_buffer = deque(maxlen=100)
        self.frequency_buffer = deque(maxlen=100)
        self.gyro_z_buffer = deque(maxlen=500)  # 表示用
        
        # グラフ設定
        self.setup_graphs()
        
        # コールバック設定
        self.receiver.set_data_callback(self._on_data_received)
        self.receiver.set_error_callback(self._on_error)
        self.frequency_analyzer.add_callback(self._on_frequency_detected)
        
        # 開始時刻
        self.start_time = time.time()
        self.data_updated = False
        
        # 最新の周波数
        self.latest_frequency = 0
    
    def setup_graphs(self):
        """グラフの初期設定"""
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 上段：ジャイロZ軸信号
        self.ax1.set_title('Gyro Z-axis Signal')
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Angular Velocity (rad/s)')
        self.ax1.grid(True)
        
        # 下段：検出周波数履歴
        self.ax2.set_title('Detected Frequency History (using MotionAnalyzer)')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Frequency (Hz)')
        self.ax2.grid(True)
        self.ax2.set_ylim(0, 10)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def _on_data_received(self, parsed_data):
        """データ受信時の処理"""
        try:
            current_time = time.time()
            relative_time = current_time - self.start_time
            
            # センサーデータ処理
            processed_data = self.processor.process_sensor_data(parsed_data, current_time)
            
            # ジャイロZ軸データを保存（表示用）
            gyro_z = processed_data['local_gyroscope'][2]
            self.gyro_z_buffer.append(gyro_z)
            
            # FrequencyAnalyzerで解析
            frequency_result = self.frequency_analyzer.process(processed_data, current_time)
            
            # 時間と周波数を履歴に追加
            self.time_buffer.append(relative_time)
            
            # 検出された場合は周波数を更新、されなかった場合は0
            if frequency_result['detected']:
                self.latest_frequency = frequency_result['best_frequency']
                self.frequency_buffer.append(self.latest_frequency)
            else:
                self.frequency_buffer.append(0)
            
            self.data_updated = True
                
        except Exception as e:
            print(f"エラー: {e}")
    
    def _on_frequency_detected(self, result):
        """FrequencyAnalyzerからのコールバック"""
        frequency = result['best_frequency']
        confidence = result['best_confidence']
        print(f"周波数検出: {frequency:.2f} Hz (信頼度: {confidence:.3f})")
    
    def _on_error(self, error_msg):
        """エラー処理"""
        print(f"受信エラー: {error_msg}")
    
    def update_graphs(self):
        """グラフ更新"""
        if not self.data_updated or len(self.time_buffer) < 2:
            return
        
        try:
            time_data = list(self.time_buffer)
            frequency_data = list(self.frequency_buffer)
            
            # 上段：ジャイロZ軸信号
            self.ax1.clear()
            self.ax1.set_title('Gyro Z-axis Signal (Recent 5 seconds)')
            self.ax1.set_xlabel('Time (s)')
            self.ax1.set_ylabel('Angular Velocity (rad/s)')
            self.ax1.grid(True)
            
            if len(self.gyro_z_buffer) > 0 and len(time_data) > 0:
                # 最新5秒分のデータを表示
                recent_samples = min(500, len(self.gyro_z_buffer))
                gyro_data = list(self.gyro_z_buffer)[-recent_samples:]
                gyro_time = np.linspace(
                    time_data[-1] - recent_samples/100, 
                    time_data[-1], 
                    recent_samples
                )
                
                self.ax1.plot(gyro_time, gyro_data, 'b-', linewidth=1)
                
                # Y軸の範囲を動的調整
                if gyro_data:
                    y_max = max(abs(min(gyro_data)), abs(max(gyro_data)))
                    if y_max > 0:
                        self.ax1.set_ylim(-y_max*1.1, y_max*1.1)
            
            # 下段：周波数履歴（MotionAnalyzerの結果）
            self.ax2.clear()
            self.ax2.set_title('Detected Frequency History (using MotionAnalyzer)')
            self.ax2.set_xlabel('Time (s)')
            self.ax2.set_ylabel('Frequency (Hz)')
            self.ax2.grid(True)
            
            # 有効な周波数のみプロット
            valid_frequencies = []
            valid_times = []
            for t, f in zip(time_data, frequency_data):
                if f > 0.5:  # 0.5Hz以上のみ
                    valid_times.append(t)
                    valid_frequencies.append(f)
            
            if valid_times:
                self.ax2.plot(valid_times, valid_frequencies, 'ro-', markersize=4, linewidth=2)
                
                # 最新の周波数を強調表示
                if valid_frequencies:
                    latest_freq = valid_frequencies[-1]
                    latest_time = valid_times[-1]
                    self.ax2.plot(latest_time, latest_freq, 'go', markersize=8)
                    self.ax2.text(latest_time, latest_freq + 0.3, 
                                 f'{latest_freq:.2f}Hz', 
                                 ha='center', fontsize=12, fontweight='bold')
            
            # Y軸の範囲設定
            if valid_frequencies:
                max_freq = max(valid_frequencies)
                self.ax2.set_ylim(0, max(10, max_freq + 1))
            else:
                self.ax2.set_ylim(0, 10)
            
            # X軸の範囲設定
            if time_data:
                latest_time = time_data[-1]
                self.ax1.set_xlim(max(0, latest_time - 5), latest_time + 0.5)
                self.ax2.set_xlim(max(0, latest_time - 30), latest_time + 1)
            
            # グラフ描画
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            self.data_updated = False
            
        except Exception as e:
            print(f"グラフ更新エラー: {e}")
    
    def start(self):
        """監視開始"""
        print("=== MotionAnalyzer使用 周波数グラフ監視 ===")
        print("FrequencyAnalyzerクラスを使用して周波数検出")
        print("上段: ジャイロZ軸信号")
        print("下段: 検出周波数履歴")
        print("")
        
        self.receiver.start_receiving()
        
        # グラフ更新ループ
        try:
            while True:
                self.update_graphs()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """監視停止"""
        self.receiver.stop_receiving()
        plt.close('all')
        print("監視停止")
    
    def configure_analyzer(self, **kwargs):
        """FrequencyAnalyzerの設定変更"""
        self.frequency_analyzer.thresholds.update(kwargs)
        print(f"FrequencyAnalyzer設定更新: {kwargs}")

# 実行
if __name__ == "__main__":
    monitor = FrequencyGraphMonitorWithAnalyzer()
    
    # FrequencyAnalyzerの感度調整
    monitor.configure_analyzer(
        prominence=0.15,  # より感度を上げる
        entropy=0.7       # エントロピー閾値を緩める
    )
    
    try:
        monitor.start()
        
    except KeyboardInterrupt:
        print("\n終了中...")
        monitor.stop()
        print("プログラム終了")