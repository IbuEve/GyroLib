import numpy as np
import time
import threading
import random
import math
from module.SensorReceiver import GyroStickReceiver
from module.SensorMath import SensorDataProcessor

class SmoothPeriodicLED:
    """なめらかでスムーズな周期的LED制御"""
    
    def __init__(self, receiver):
        self.receiver = receiver
        
        # 周波数関連（スムーズ変化）
        self.min_frequency = 0.5
        self.max_frequency = 1.5
        self.target_frequency = 0.8
        self.current_frequency = 0.4
        self.frequency_smoothing = 0.98  # 0.95-0.99（高いほどなめらか）
        
        # 色関連（スムーズ変化）
        self.target_color = [0, 255, 0]  # RGB
        self.current_color = [0.0, 255.0, 0.0]  # float for smooth interpolation
        self.color_smoothing = 0.96
        
        # 明度関連（スムーズ変化）
        self.min_brightness = 20
        self.max_brightness = 255
        self.brightness_smoothing = 0.85  # 明度は少し早めに変化
        self.current_brightness = 100.0
        
        # 時間管理
        self.start_time = time.time()
        self.running = False
        
        # なめらかな値変化のためのフィルタ
        self.frequency_filter = ExponentialMovingAverage(alpha=1-self.frequency_smoothing)
        self.color_filters = [
            ExponentialMovingAverage(alpha=1-self.color_smoothing),
            ExponentialMovingAverage(alpha=1-self.color_smoothing),
            ExponentialMovingAverage(alpha=1-self.color_smoothing)
        ]
        self.brightness_filter = ExponentialMovingAverage(alpha=1-self.brightness_smoothing)
        
        # 更新スレッド
        self.update_thread = None
        
        # ターゲット値の変更タイマー
        self.last_target_change = time.time()
        self.target_change_interval = random.uniform(2.0, 5.0)  # 2-5秒でターゲット変更
        
    def start(self):
        """LED制御開始"""
        self.running = True
        self.start_time = time.time()
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        print("スムーズLED制御開始")
    
    def stop(self):
        """LED制御停止"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=0.5)
        self.receiver.led_on(0, 0, 0)
        print("スムーズLED制御停止")
    
    def _update_targets(self):
        """ターゲット値をなめらかに変更"""
        current_time = time.time()
        
        if current_time - self.last_target_change > self.target_change_interval:
            # 新しいターゲット周波数（なめらかに変化）
            frequency_change = random.uniform(-0.5, 0.5)
            self.target_frequency = max(self.min_frequency, 
                                      min(self.max_frequency, 
                                          self.target_frequency + frequency_change))
            
            # 新しいターゲット色（なめらかに変化）
            self.target_color = self._generate_smooth_color(self.target_frequency)
            
            # 次の変更時間
            self.target_change_interval = random.uniform(3.0, 8.0)
            self.last_target_change = current_time
            
            print(f"新ターゲット - 周波数: {self.target_frequency:.1f}Hz, 色: {self.target_color}")
    
    def _generate_smooth_color(self, frequency):
        """周波数からなめらかな色を生成"""
        # 周波数を0-1に正規化
        normalized_freq = (frequency - self.min_frequency) / (self.max_frequency - self.min_frequency)
        
        # HSVベースでなめらかな色変化
        hue = normalized_freq * 240  # 0(赤) → 240(青)
        saturation = 1.0
        value = 1.0
        
        # HSV to RGB conversion
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs((h % 2) - 1))
        m = value - c
        
        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        else:
            r, g, b = x, 0, c
        
        return [int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)]
    
    def _update_loop(self):
        """メインの更新ループ"""
        update_rate = 60  # 60Hz更新でなめらか
        
        while self.running:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # ターゲット値を更新
            self._update_targets()
            
            # 周波数をなめらかに変化
            self.current_frequency = self.frequency_filter.update(self.target_frequency)
            
            # 色をなめらかに変化
            for i in range(3):
                self.current_color[i] = self.color_filters[i].update(self.target_color[i])
            
            # 正弦波で基本明度を計算（なめらか）
            phase = 2 * math.pi * self.current_frequency * elapsed_time
            sin_value = math.sin(phase)
            
            # 正弦波を0-1に正規化
            normalized_sin = (sin_value + 1) / 2
            
            # ターゲット明度計算
            brightness_range = self.max_brightness - self.min_brightness
            target_brightness = self.min_brightness + brightness_range * normalized_sin
            
            # 明度をなめらかに変化
            self.current_brightness = self.brightness_filter.update(target_brightness)
            
            # 最終的な色を計算（なめらかな明度適用）
            brightness_ratio = self.current_brightness / 255.0
            final_color = [
                int(self.current_color[0] * brightness_ratio),
                int(self.current_color[1] * brightness_ratio),
                int(self.current_color[2] * brightness_ratio)
            ]
            
            # LED設定
            self.receiver.led_on(*final_color)
            
            # デバッグ出力（0.5秒おき）
            if int(elapsed_time * 2) % 2 == 0 and abs(sin_value) > 0.9:
                peak_status = "PEAK" if sin_value > 0 else "valley"
                print(f"周波数: {self.current_frequency:.1f}Hz | 明度: {int(self.current_brightness)} | RGB: {final_color} | {peak_status}")
            
            time.sleep(1.0 / update_rate)

class ExponentialMovingAverage:
    """指数移動平均によるなめらかな値の変化"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class SmoothPipeline:
    """なめらかなLED制御パイプライン"""
    
    def __init__(self):
        self.receiver = GyroStickReceiver()
        self.processor = SensorDataProcessor()
        self.led_controller = SmoothPeriodicLED(self.receiver)
        
        # センサーコールバック（使用しないが設定）
        self.receiver.set_data_callback(self._on_data_received)
        self.receiver.set_error_callback(self._on_error)
    
    def _on_data_received(self, parsed_data):
        """データ受信（無視）"""
        pass
    
    def _on_error(self, error_msg):
        """エラー処理"""
        print(f"受信エラー: {error_msg}")
    
    def start(self):
        """開始"""
        print("=== なめらかな周期的LED制御 ===")
        print("特徴:")
        print("- 指数移動平均によるなめらかな値変化")
        print("- 60Hz更新でスムーズな描画")
        print("- HSVベースの自然な色変化")
        print("- 段階的でない連続的な変化")
        print(f"周波数範囲: {self.led_controller.min_frequency}-{self.led_controller.max_frequency}Hz")
        print(f"明度範囲: {self.led_controller.min_brightness}-{self.led_controller.max_brightness}")
        print("")
        
        self.receiver.start_receiving()
        self.led_controller.start()
    
    def stop(self):
        """停止"""
        print("停止中...")
        self.led_controller.stop()
        self.receiver.stop_receiving()
        print("停止完了")
    
    def configure_smoothness(self, frequency_smooth=None, color_smooth=None, brightness_smooth=None):
        """なめらかさの調整"""
        if frequency_smooth is not None:
            self.led_controller.frequency_smoothing = frequency_smooth
            self.led_controller.frequency_filter.alpha = 1 - frequency_smooth
            print(f"周波数なめらかさ更新: {frequency_smooth}")
        
        if color_smooth is not None:
            self.led_controller.color_smoothing = color_smooth
            for filter in self.led_controller.color_filters:
                filter.alpha = 1 - color_smooth
            print(f"色なめらかさ更新: {color_smooth}")
        
        if brightness_smooth is not None:
            self.led_controller.brightness_smoothing = brightness_smooth
            self.led_controller.brightness_filter.alpha = 1 - brightness_smooth
            print(f"明度なめらかさ更新: {brightness_smooth}")

# 使用例
if __name__ == "__main__":
    pipeline = SmoothPipeline()
    
    # なめらかさ調整（0.9-0.99が推奨、高いほどなめらか）
    pipeline.configure_smoothness(
        frequency_smooth=0.98,  # 周波数変化のなめらかさ
        color_smooth=0.96,      # 色変化のなめらかさ  
        brightness_smooth=0.90  # 明度変化のなめらかさ
    )
    
    try:
        pipeline.start()
        
        print("実行中... (Ctrl+C で停止)")
        print("なめらかさレベル:")
        print("  0.90-0.95 = 普通のなめらかさ")
        print("  0.95-0.98 = 高いなめらかさ")  
        print("  0.98-0.99 = 極めてなめらか")
        print("")
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n終了中...")
        pipeline.stop()
        print("プログラム終了")