import numpy as np
import time
import threading
from module.SensorReceiver import GyroStickReceiver
from module.SensorMath import SensorDataProcessor

class SimpleVelocityLED:
    """速度の二乗和で緑色LEDの明るさを制御するシンプルクラス（二次関数正規化）"""
    
    def __init__(self, receiver):
        self.receiver = receiver
        self.velocity_scale = 2.0  # 速度スケール調整
        self.curve_power = 2.0     # カーブの強さ（1.0=線形, 2.0=二次, 3.0=三次）
    
    def update_led(self, velocity):
        """速度から緑色LEDの明るさを更新（二次関数的正規化）"""
        # 速度の二乗和を計算
        velocity_squared_sum = sum(v**2 for v in velocity)
        
        # まず線形正規化（0-1の範囲）
        linear_normalized = min(velocity_squared_sum / (self.velocity_scale**2), 1.0)
        
        # 二次関数的に変換（小さい値は小さく、大きい値はグイっと上がる）
        curved_normalized = linear_normalized ** self.curve_power
        
        # 0-255に変換
        brightness = int(curved_normalized * 255)
        
        # 緑色で設定（R=0, G=brightness, B=0）
        self.receiver.led_on(0, brightness, 0)
        
        return brightness, linear_normalized, curved_normalized

class SimplePipeline:
    """シンプルな速度LED制御パイプライン"""
    
    def __init__(self):
        self.receiver = GyroStickReceiver()
        self.processor = SensorDataProcessor()
        self.led_controller = SimpleVelocityLED(self.receiver)
        
        self.receiver.set_data_callback(self._on_data_received)
        self.receiver.set_error_callback(self._on_error)
    
    def _on_data_received(self, parsed_data):
        """データ受信時の処理"""
        try:
            current_time = time.time()
            
            # センサーデータ処理
            processed_data = self.processor.process_sensor_data(parsed_data, current_time)
            
            # 速度取得
            velocity = processed_data['velocity']
            
            # LED更新
            brightness, linear, curved = self.led_controller.update_led(velocity)
            
            # デバッグ出力（値が変化した時のみ）
            if brightness > 5:  # 明度が一定以上の時のみ表示
                velocity_sum = sum(v**2 for v in velocity)
                print(f"速度²和: {velocity_sum:.3f} | 線形: {linear:.3f} → 曲線: {curved:.3f} | 明度: {brightness}")
                
        except Exception as e:
            print(f"データ処理エラー: {e}")
    
    def _on_error(self, error_msg):
        """エラー処理"""
        print(f"受信エラー: {error_msg}")
    
    def start(self):
        """開始"""
        print("=== シンプル速度LED制御（二次関数正規化） ===")
        print("速度の二乗和 → 二次関数変換 → 緑色の明度")
        print(f"速度スケール: {self.led_controller.velocity_scale}")
        print(f"カーブ強度: {self.led_controller.curve_power}")
        print("小さい値は抑制、大きい値はグイっと強調")
        print("")
        
        self.receiver.start_receiving()
    
    def stop(self):
        """停止"""
        print("停止中...")
        self.receiver.led_on(0, 0, 0)  # LED消灯
        self.receiver.stop_receiving()
        print("停止完了")
    
    def set_velocity_scale(self, scale):
        """速度スケール調整"""
        self.led_controller.velocity_scale = scale
        print(f"速度スケール更新: {scale}")
    
    def set_curve_power(self, power):
        """カーブの強さ調整"""
        self.led_controller.curve_power = power
        print(f"カーブ強度更新: {power} ({'線形' if power == 1.0 else '二次関数的' if power == 2.0 else '強いカーブ'})")

# 使用例
if __name__ == "__main__":
    pipeline = SimplePipeline()
    
    # 設定調整
    pipeline.set_velocity_scale(2)  # 感度調整
    pipeline.set_curve_power(2.5)     # カーブ強度（1.0=線形, 2.0=二次, 3.0=三次, etc.）
    
    try:
        pipeline.start()
        
        print("実行中... (Ctrl+C で停止)")
        print("curve_power説明:")
        print("  1.0 = 線形（従来通り）")
        print("  2.0 = 二次関数（小さい値抑制、大きい値強調）")
        print("  3.0 = 三次関数（さらに強いカーブ）")
        print("")
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n終了中...")
        pipeline.stop()
        print("プログラム終了")