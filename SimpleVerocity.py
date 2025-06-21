import numpy as np
import time
import threading
from module.SensorReceiver import GyroStickReceiver
from module.SensorMath import SensorDataProcessor

class MovementBasedLED:
    """移動量で色、速度で明度を制御するLED"""
    
    def __init__(self, receiver):
        self.receiver = receiver
        self.velocity_scale = 2.0
        self.curve_power = 2.0
        
        # 移動量関連
        self.accumulated_distance = 0.0  # 累積移動距離
        self.distance_scale = 10.0       # 距離スケール（色変化の感度）
        self.distance_decay = 0.995      # 距離の減衰係数（長時間静止で徐々にリセット）
        
        # 前回の時刻と位置（移動距離計算用）
        self.last_time = None
        self.last_velocity = np.array([0.0, 0.0, 0.0])
        
        # 静止判定
        self.stationary_threshold = 0.1  # この値以下は静止とみなす
        self.stationary_duration = 0.0   # 静止している時間
        
    def update_led(self, velocity, current_time):
        """速度と時間から移動量を更新し、LEDを制御"""
        current_velocity = np.array(velocity)
        
        # 移動距離を計算（台形積分）
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                # 平均速度で移動距離を計算
                avg_velocity = (current_velocity + self.last_velocity) / 2
                distance_increment = np.linalg.norm(avg_velocity) * dt
                
                # 累積移動距離を更新
                self.accumulated_distance += distance_increment
                
                # 静止判定と距離減衰
                current_speed = np.linalg.norm(current_velocity)
                if current_speed < self.stationary_threshold:
                    self.stationary_duration += dt
                    # 長時間静止していたら距離を徐々に減衰
                    if self.stationary_duration > 2.0:  # 2秒以上静止
                        self.accumulated_distance *= self.distance_decay
                else:
                    self.stationary_duration = 0.0
        
        # 前回の値を保存
        self.last_time = current_time
        self.last_velocity = current_velocity.copy()
        
        # 速度の大きさから明度を計算（従来通り）
        velocity_squared_sum = sum(v**2 for v in velocity)
        linear_normalized = min(velocity_squared_sum / (self.velocity_scale**2), 1.0)
        curved_normalized = linear_normalized ** self.curve_power
        brightness = int(curved_normalized * 255)
        
        # 移動距離から色を決定
        color = self._get_color_from_distance(self.accumulated_distance)
        
        # 明度を色に適用
        final_color = tuple(int(c * brightness / 255) for c in color)
        
        # LED設定
        self.receiver.led_on(*final_color)
        
        return brightness, self.accumulated_distance, final_color
    
    def _get_color_from_distance(self, distance):
        """累積移動距離から色を決定"""
        # 距離を正規化（0-1の範囲に循環）
        normalized_distance = (distance / self.distance_scale) % 1.0
        
        # HSVベースで色を決定（距離に応じて色相が変化）
        hue = normalized_distance * 360  # 0-360度
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
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    def reset_distance(self):
        """累積移動距離をリセット"""
        self.accumulated_distance = 0.0
        print("移動距離リセット")

class MovementPipeline:
    """移動量ベースLED制御パイプライン"""
    
    def __init__(self):
        self.receiver = GyroStickReceiver()
        self.processor = SensorDataProcessor()
        self.led_controller = MovementBasedLED(self.receiver)
        
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
            brightness, distance, final_color = self.led_controller.update_led(velocity, current_time)
            
            # デバッグ出力
            if brightness > 5 or distance > 0.1:
                velocity_magnitude = np.linalg.norm(velocity)
                print(f"速度: {velocity_magnitude:.2f} | 累積距離: {distance:.2f} | 明度: {brightness} | RGB: {final_color}")
                
        except Exception as e:
            print(f"データ処理エラー: {e}")
    
    def _on_error(self, error_msg):
        """エラー処理"""
        print(f"受信エラー: {error_msg}")
    
    def start(self):
        """開始"""
        print("=== 移動量ベース色変化LED制御 ===")
        print("特徴:")
        print("- 累積移動距離 → 色の変化（HSV色相環）")
        print("- 速度の大きさ → 明度の変化")
        print("- 長時間静止 → 距離の自動減衰")
        print("- 継続的な同方向移動 → 低周波的な色変化")
        print(f"速度スケール: {self.led_controller.velocity_scale}")
        print(f"距離スケール: {self.led_controller.distance_scale}")
        print(f"距離減衰: {self.led_controller.distance_decay}")
        print("")
        
        self.receiver.start_receiving()
    
    def stop(self):
        """停止"""
        print("停止中...")
        self.receiver.led_on(0, 0, 0)
        self.receiver.stop_receiving()
        print("停止完了")
    
    def configure(self, velocity_scale=None, distance_scale=None, 
                 distance_decay=None, curve_power=None):
        """設定変更"""
        if velocity_scale is not None:
            self.led_controller.velocity_scale = velocity_scale
            print(f"速度スケール更新: {velocity_scale}")
        
        if distance_scale is not None:
            self.led_controller.distance_scale = distance_scale
            print(f"距離スケール更新: {distance_scale}")
        
        if distance_decay is not None:
            self.led_controller.distance_decay = distance_decay
            print(f"距離減衰更新: {distance_decay}")
        
        if curve_power is not None:
            self.led_controller.curve_power = curve_power
            print(f"カーブ強度更新: {curve_power}")
    
    def reset_distance(self):
        """移動距離リセット"""
        self.led_controller.reset_distance()

# 使用例
if __name__ == "__main__":
    pipeline = MovementPipeline()
    
    # 設定調整
    pipeline.configure(
        velocity_scale=1.5,      # 速度感度（小さいほど敏感）
        distance_scale=8.0,      # 距離感度（小さいほど早く色変化）
        distance_decay=0.99,    # 減衰率（1.0に近いほど減衰が遅い）
        curve_power=2.0          # 明度カーブ
    )
    
    try:
        pipeline.start()
        
        print("実行中... (Ctrl+C で停止)")
        print("動作説明:")
        print("  同方向への継続移動 → 色が順次変化")
        print("  急激な動き → 明るく点灯")
        print("  ゆっくりな動き → 暗く点灯")
        print("  静止状態 → 距離が徐々に減衰")
        print("  'r' + Enter → 距離リセット")
        print("")
        
        # キーボード入力スレッド（距離リセット用）
        def input_thread():
            while True:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'r':
                        pipeline.reset_distance()
                    elif user_input == 'q':
                        break
                except:
                    break
        
        input_handler = threading.Thread(target=input_thread)
        input_handler.daemon = True
        input_handler.start()
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n終了中...")
        pipeline.stop()
        print("プログラム終了")