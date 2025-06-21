import numpy as np
import time
from collections import deque
from abc import ABC, abstractmethod
from module.SensorMath import SensorDataProcessor
from module.SensorReceiver import GyroStickReceiver

# ベースクラス
class MotionAnalyzer(ABC):
    """動作解析の基底クラス"""
    
    def __init__(self):
        self.callbacks = []
    
    def add_callback(self, callback):
        """結果通知用コールバックを追加"""
        self.callbacks.append(callback)
    
    def notify(self, result):
        """結果を通知"""
        for callback in self.callbacks:
            callback(result)
    
    @abstractmethod
    def process(self, processed_data, timestamp):
        """データ処理の抽象メソッド"""
        pass

# 個別分析クラス
class EnergyDetector(MotionAnalyzer):
    """エネルギーレベル検出クラス"""
    
    def __init__(self, thresholds=None):
        super().__init__()
        self.thresholds = thresholds or {
            'velocity_x': 0.1, 'velocity_y': 0.1, 'velocity_z': 0.1,
            'angular_x': 40000.0, 'angular_y': 40000.0, 'angular_z': 40000.0
        }
    
    def process(self, processed_data, timestamp):
        velocity = processed_data['velocity']
        euler_rates = processed_data['euler_rates']
        
        # エネルギー計算
        vel_energies = {axis: vel**2 for axis, vel in zip(['x','y','z'], velocity)}
        angular_energies = {axis: rate**2 for axis, rate in zip(['x','y','z'], euler_rates)}
        
        # 閾値チェック
        active_channels = []
        for axis in ['x', 'y', 'z']:
            if vel_energies[axis] > self.thresholds[f'velocity_{axis}']:
                active_channels.append(f'vel_{axis}')
            if angular_energies[axis] > self.thresholds[f'angular_{axis}']:
                active_channels.append(f'ang_{axis}')
        
        result = {
            'type': 'energy',
            'timestamp': timestamp,
            'high_energy': len(active_channels) > 0,
            'active_channels': active_channels,
            'vel_energies': vel_energies,
            'angular_energies': angular_energies
        }
        
        self.notify(result)
        return result

class FrequencyAnalyzer(MotionAnalyzer):
    """周波数・周期性解析クラス"""
    
    def __init__(self, thresholds=None, sampling_rate=100):
        super().__init__()
        self.thresholds = thresholds or {'prominence': 0.7, 'entropy': 0.3}
        self.sampling_rate = sampling_rate
        self.window_sizes = [64, 128, 256, 512, 1024]
        
        # データバッファ
        self.gyro_buffers = {axis: deque(maxlen=1024) for axis in ['x', 'y', 'z']}
        self.accel_buffers = {axis: deque(maxlen=1024) for axis in ['x', 'y', 'z']}
    
    def process(self, processed_data, timestamp):
        # バッファ更新
        self._update_buffers(processed_data)
        
        # 周期性検出
        result = self._detect_periodicity(timestamp)
        
        if result['detected']:
            self.notify(result)
        
        return result
    
    def _update_buffers(self, processed_data):
        gyro = processed_data['local_gyroscope']
        accel = processed_data['global_acceleration']
        
        for i, axis in enumerate(['x', 'y', 'z']):
            self.gyro_buffers[axis].append(gyro[i])
            self.accel_buffers[axis].append(accel[i])
    
    def _detect_periodicity(self, timestamp):
        # 既存のFFT解析ロジックをここに移動
        results = {
            'type': 'frequency',
            'timestamp': timestamp,
            'detected': False,
            'best_frequency': 0,
            'best_confidence': 0
        }
        
        # ... FFT解析処理 ...
        
        return results

class JerkDetector(MotionAnalyzer):
    """ジャーク・スナップ検出クラス"""
    
    def __init__(self, thresholds=None):
        super().__init__()
        self.thresholds = thresholds or {
            'trigger_threshold': 1000.0,
            'zero_threshold': 100.0,
            'minimum_duration': 0.03
        }
        
        # ジャーク計算用バッファ
        self.accel_history = deque(maxlen=10)
        self.timestamp_history = deque(maxlen=10)
        
        # 状態管理
        self.jerk_state = {
            'is_high_jerk': False,
            'high_jerk_start_time': None,
            'last_snap_time': None
        }
    
    def process(self, processed_data, timestamp):
        accel = processed_data['global_acceleration']
        accel_magnitude = np.sqrt(sum(a**2 for a in accel))
        
        # 履歴更新
        self.accel_history.append(accel_magnitude)
        self.timestamp_history.append(timestamp)
        
        # ジャーク計算
        result = None
        if len(self.accel_history) >= 2:
            jerk_magnitude = self._calculate_jerk()
            result = self._detect_snap(jerk_magnitude, timestamp)
        
        return result
    
    def _calculate_jerk(self):
        dt = self.timestamp_history[-1] - self.timestamp_history[-2]
        if dt > 0:
            jerk = (self.accel_history[-1] - self.accel_history[-2]) / dt
            return abs(jerk)
        return 0
    
    def _detect_snap(self, jerk_magnitude, timestamp):
        # 既存のスナップ検出ロジック
        trigger_threshold = self.thresholds['trigger_threshold']
        zero_threshold = self.thresholds['zero_threshold']
        
        if not self.jerk_state['is_high_jerk'] and jerk_magnitude > trigger_threshold:
            self.jerk_state['is_high_jerk'] = True
            self.jerk_state['high_jerk_start_time'] = timestamp
            
            result = {
                'type': 'jerk_start',
                'timestamp': timestamp,
                'jerk_magnitude': jerk_magnitude
            }
            self.notify(result)
            return result
            
        elif self.jerk_state['is_high_jerk'] and jerk_magnitude < zero_threshold:
            duration = timestamp - self.jerk_state['high_jerk_start_time']
            
            if duration >= self.thresholds['minimum_duration']:
                result = {
                    'type': 'snap_detected',
                    'timestamp': timestamp,
                    'duration': duration,
                    'interval': timestamp - self.jerk_state['last_snap_time'] if self.jerk_state['last_snap_time'] else None
                }
                
                self.jerk_state['last_snap_time'] = timestamp
                self.notify(result)
            
            self.jerk_state['is_high_jerk'] = False
            self.jerk_state['high_jerk_start_time'] = None
            
            return result
        
        return None

# 統合パイプライン
class MotionAnalysisPipeline:
    """動作解析パイプライン（コーディネーター）"""
    
    def __init__(self):
        # 受信・基本処理
        self.receiver = GyroStickReceiver()
        self.processor = SensorDataProcessor()
        
        # 分析クラス
        self.energy_detector = EnergyDetector()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.jerk_detector = JerkDetector()
        
        # 結果統合用
        self.results_history = deque(maxlen=1000)
        
        # コールバック設定
        self.receiver.set_data_callback(self._on_data_received)
        self._setup_analysis_callbacks()
    
    def _setup_analysis_callbacks(self):
        """各分析クラスの結果通知を設定"""
        self.energy_detector.add_callback(self._on_energy_result)
        self.frequency_analyzer.add_callback(self._on_frequency_result)
        self.jerk_detector.add_callback(self._on_jerk_result)
    
    def _on_data_received(self, parsed_data):
        """データ受信時の処理"""
        current_time = time.time()
        processed_data = self.processor.process_sensor_data(parsed_data, current_time)
        
        # 各分析を並列実行
        energy_result = self.energy_detector.process(processed_data, current_time)
        frequency_result = self.frequency_analyzer.process(processed_data, current_time)
        jerk_result = self.jerk_detector.process(processed_data, current_time)
        
        # 結果の統合判定
        self._integrate_results(energy_result, frequency_result, jerk_result, current_time)
    
    def _integrate_results(self, energy, frequency, jerk, timestamp):
        """結果を統合して高次判定"""
        # 例：高エネルギー + 周期性 = 振動状態
        if energy['high_energy'] and frequency['detected']:
            print(f"[{timestamp:.3f}s] 振動状態検出: {frequency['best_frequency']:.2f}Hz")
        
        # 結果履歴に保存
        self.results_history.append({
            'timestamp': timestamp,
            'energy': energy,
            'frequency': frequency,
            'jerk': jerk
        })
    
    def _on_energy_result(self, result):
        """エネルギー検出結果の処理"""
        if result['high_energy']:
            print(f"高エネルギー: {result['active_channels']}")
    
    def _on_frequency_result(self, result):
        """周波数解析結果の処理"""
        print(f"周期性検出: {result['best_frequency']:.2f}Hz")
    
    def _on_jerk_result(self, result):
        """ジャーク検出結果の処理"""
        if result['type'] == 'snap_detected':
            print(f"★ スナップ検出! ★ (持続時間: {result['duration']:.3f}s)")
    
    def start(self):
        print("モジュラー動作解析パイプライン開始")
        self.receiver.start_receiving()
    
    def stop(self):
        print("パイプライン停止")
        self.receiver.stop_receiving()
    
    # 設定メソッド
    def configure_energy_detector(self, **kwargs):
        self.energy_detector.thresholds.update(kwargs)
    
    def configure_frequency_analyzer(self, **kwargs):
        self.frequency_analyzer.thresholds.update(kwargs)
    
    def configure_jerk_detector(self, **kwargs):
        self.jerk_detector.thresholds.update(kwargs)

# 使用例
if __name__ == "__main__":
    pipeline = MotionAnalysisPipeline()
    
    # 各分析器の設定
    pipeline.configure_energy_detector(velocity_x=0.1, angular_z=40000.0)
    pipeline.configure_frequency_analyzer(prominence=0.7, entropy=0.3)
    pipeline.configure_jerk_detector(trigger_threshold=1000.0)
    
    try:
        pipeline.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pipeline.stop()