import numpy as np
import time
from collections import deque
from abc import ABC, abstractmethod
from scipy.signal import find_peaks
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
    """周波数・周期性解析クラス（パラボリック補間付き）"""
    
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
        """周期性検出処理"""
        results = {
            'type': 'frequency',
            'timestamp': timestamp,
            'detected': False,
            'best_frequency': 0,
            'best_confidence': 0,
            'gyro_results': {},
            'accel_results': {}
        }
        
        # ジャイロと加速度の各軸をFFT解析
        sensors = {
            'gyro': self.gyro_buffers,
            'accel': self.accel_buffers
        }
        
        all_detections = []
        
        for sensor_type, buffers in sensors.items():
            sensor_results = {}
            
            for axis in ['x', 'y', 'z']:
                axis_result = self._analyze_axis(buffers[axis], f"{sensor_type}_{axis}")
                sensor_results[axis] = axis_result
                
                if axis_result['detected']:
                    all_detections.append(axis_result)
            
            results[f'{sensor_type}_results'] = sensor_results
        
        # 最良の検出結果を選択
        if all_detections:
            best_detection = max(all_detections, key=lambda x: x['confidence'])
            results['detected'] = True
            results['best_frequency'] = best_detection['frequency']
            results['best_confidence'] = best_detection['confidence']
        
        return results
    
    def _analyze_axis(self, buffer_data, axis_name):
        """単一軸のFFT解析（周波数精密化付き）"""
        if len(buffer_data) < 64:
            return {'detected': False, 'reason': 'insufficient_data'}
        
        best_result = {'detected': False, 'confidence': 0}
        
        for window_size in self.window_sizes:
            if len(buffer_data) < window_size:
                continue
            
            recent_data = list(buffer_data)[-window_size:]
            
            # FFT実行
            fft_result = np.fft.fft(recent_data)
            frequencies = np.fft.fftfreq(window_size, 1/self.sampling_rate)
            power = np.abs(fft_result)**2
            
            # 正の周波数のみ
            pos_mask = frequencies > 0
            freq_pos = frequencies[pos_mask]
            power_pos = power[pos_mask]
            
            if len(power_pos) == 0:
                continue
            
            # ピーク検出
            peak_idx = np.argmax(power_pos)
            
            # パラボリック補間で周波数を精密化
            refined_frequency = self._refine_frequency(freq_pos, power_pos, peak_idx)
            
            # 信頼性指標計算
            prominence = self._calculate_prominence(freq_pos, power_pos, refined_frequency)
            entropy = self._calculate_entropy(power_pos)
            
            # 閾値チェック
            if (prominence > self.thresholds['prominence'] and 
                entropy < self.thresholds['entropy']):
                
                confidence = prominence * (1 - entropy)
                
                if confidence > best_result['confidence']:
                    best_result = {
                        'detected': True,
                        'frequency': refined_frequency,
                        'raw_frequency': freq_pos[peak_idx],  # 元の周波数も保存
                        'prominence': prominence,
                        'entropy': entropy,
                        'confidence': confidence,
                        'window_size': window_size,
                        'axis': axis_name
                    }
        
        return best_result
    
    def _refine_frequency(self, frequencies, power, peak_idx):
        """パラボリック補間で周波数を精密化"""
        if peak_idx == 0 or peak_idx >= len(power) - 1:
            return frequencies[peak_idx]
        
        # 隣接する3点でパラボリック補間
        y1, y2, y3 = power[peak_idx-1], power[peak_idx], power[peak_idx+1]
        
        # パラボリックピークの位置
        a = (y1 - 2*y2 + y3) / 2
        if a != 0:
            peak_offset = (y1 - y3) / (4 * a)
            refined_idx = peak_idx + peak_offset
            
            # 補間された周波数
            freq_resolution = frequencies[1] - frequencies[0]
            refined_frequency = frequencies[peak_idx] + peak_offset * freq_resolution
            
            return refined_frequency
        
        return frequencies[peak_idx]
    
    def _calculate_prominence(self, frequencies, power, peak_freq, bandwidth=0.1):
        """ピーク突出度の計算"""
        peak_mask = np.abs(frequencies - peak_freq) <= bandwidth
        peak_power = np.sum(power[peak_mask])
        total_power = np.sum(power)
        return peak_power / total_power if total_power > 0 else 0
    
    def _calculate_entropy(self, power):
        """スペクトルエントロピーの計算"""
        power_norm = power / (np.sum(power) + 1e-10)
        entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10))
        max_entropy = np.log2(len(power))
        return entropy / max_entropy if max_entropy > 0 else 1

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
        """スナップ検出処理"""
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
            active_channels_str = ', '.join(result['active_channels'])
            print(f"高エネルギー検出: {active_channels_str}")
    
    def _on_frequency_result(self, result):
        """周波数解析結果の処理（精密化された周波数表示）"""
        refined_freq = result['best_frequency']
        confidence = result['best_confidence']
        print(f"周期性検出: {refined_freq:.4f}Hz (信頼度: {confidence:.3f})")
    
    def _on_jerk_result(self, result):
        """ジャーク検出結果の処理"""
        if result and result.get('type') == 'snap_detected':
            duration = result.get('duration', 0)
            interval = result.get('interval')
            print(f"★ スナップ検出! ★ (持続時間: {duration:.3f}s)")
            if interval:
                print(f"  前回スナップからの間隔: {interval:.3f}s")
        elif result and result.get('type') == 'jerk_start':
            jerk_mag = result.get('jerk_magnitude', 0)
            print(f"高ジャーク開始: {jerk_mag:.1f}")
    
    def start(self):
        """パイプライン開始"""
        print("=== モジュラー動作解析パイプライン（周波数精密化版） ===")
        print("機能:")
        print("- エネルギー検出")
        print("- 周波数・周期性解析（パラボリック補間付き）")
        print("- ジャーク・スナップ検出")
        print("")
        self.receiver.start_receiving()
    
    def stop(self):
        """パイプライン停止"""
        print("パイプライン停止")
        self.receiver.stop_receiving()
    
    # 設定メソッド
    def configure_energy_detector(self, **kwargs):
        """エネルギー検出器設定"""
        self.energy_detector.thresholds.update(kwargs)
        print(f"エネルギー検出器設定更新: {kwargs}")
    
    def configure_frequency_analyzer(self, **kwargs):
        """周波数解析器設定"""
        self.frequency_analyzer.thresholds.update(kwargs)
        print(f"周波数解析器設定更新: {kwargs}")
    
    def configure_jerk_detector(self, **kwargs):
        """ジャーク検出器設定"""
        self.jerk_detector.thresholds.update(kwargs)
        print(f"ジャーク検出器設定更新: {kwargs}")

# 使用例
if __name__ == "__main__":
    pipeline = MotionAnalysisPipeline()
    
    # 各分析器の設定
    pipeline.configure_energy_detector(
        velocity_x=0.1, 
        velocity_y=0.1,
        velocity_z=0.1,
        angular_x=40000.0,
        angular_y=40000.0,
        angular_z=40000.0
    )
    
    pipeline.configure_frequency_analyzer(
        prominence=0.7, 
        entropy=0.3
    )
    
    pipeline.configure_jerk_detector(
        trigger_threshold=800.0,
        zero_threshold=100.0,
        minimum_duration=0.03
    )
    
    try:
        pipeline.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n終了中...")
        pipeline.stop()
        print("プログラム終了")