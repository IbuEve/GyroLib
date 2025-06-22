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
        self.distance_scale = 1.0       # 距離スケール（緑色時の明度用）
        self.distance_decay = 0.99      # 距離の減衰係数
        
        # 前回の時刻と位置（移動距離計算用）
        self.last_time = None
        self.last_velocity = np.array([0.0, 0.0, 0.0])
        
        # 静止判定
        self.stationary_threshold = 0.3  # この値以下は静止とみなす
        self.stationary_duration = 0.0   # 静止している時間
        
        self.color_history = []  # 過去の色factor履歴
        self.history_length = 10  # 何ステップ前まで保持するか（ハイパラ）
        
    def update_led(self, velocity, current_time):
        """速度と時間から移動量を更新し、LEDを制御"""
        current_velocity = np.array(velocity)
                # 現在の速度
        current_speed = np.linalg.norm(current_velocity)
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
                is_stationary = current_speed < self.stationary_threshold

                if not is_stationary:
                    # 動いている時：通常の速度ベース色計算
                    color_factor = min(1, current_speed / self.velocity_scale)
                    
                    # 履歴に追加
                    self.color_history.append(color_factor)
                    if len(self.color_history) > self.history_length:
                        self.color_history.pop(0)  # 古いものを削除
                else:
                    # 静止中はnステップ前の色を使用
                    if len(self.color_history) >= self.history_length:
                        color_factor = self.color_history[0]  # history_length前の色
                    elif len(self.color_history) > 0:
                        color_factor = self.color_history[0]  # 履歴の最古
                    else:
                        color_factor = 0  # 履歴がない場合は緑
                if current_speed < self.stationary_threshold:
                    self.stationary_duration += dt
                    # 長時間静止していたら距離を徐々に減衰
                    if self.stationary_duration > 0.5:
                        self.accumulated_distance *= self.distance_decay
                else:
                    self.stationary_duration = 0.0
        
        # 前回の値を保存
        self.last_time = current_time
        self.last_velocity = current_velocity.copy()
        

        
        # 速度から基本明度を計算（従来通り）
        velocity_squared_sum = sum(v**2 for v in velocity)
        linear_normalized = min(velocity_squared_sum / (self.velocity_scale**2), 1.0)
        curved_normalized = linear_normalized ** self.curve_power
        base_brightness = curved_normalized
        
        
        # 緑から赤のグラデーション
        red_component = int(color_factor * 255)
        green_component = int((1 - color_factor) * 255)
        base_color = (red_component, green_component, 0)
        
        # 緑色時（遅い動き）は累積距離で明度を補強
        if color_factor < 0.5:  # 緑が強い時
            distance_brightness = min(1, self.accumulated_distance / self.distance_scale)
            # 緑の強さに応じて距離による明度補強を適用
            green_ratio = 1 - color_factor * 2  # 0.5以下を0-1に正規化
            final_brightness = base_brightness + (distance_brightness * green_ratio * 0.5)
            final_brightness = min(1, final_brightness)
        else:
            final_brightness = base_brightness
        
        # 最終的な明度を適用
        brightness_value = int(final_brightness * 255)
        final_color = tuple(int(c * brightness_value / 255) for c in base_color)
        
        # LED設定
        self.receiver.led_on(*final_color)
        
        return brightness_value, self.accumulated_distance, final_color
    
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
        print("=== 速度ベース色変化LED制御 ===")
        print("特徴:")
        print("- 速度が速い → 赤色（明度は速度）")
        print("- 速度が遅い → 緑色（明度は速度+累積距離）")
        print("- 長時間静止 → 距離の自動減衰")
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
        velocity_scale=1.5,        # 速度感度（小さいほど敏感）
        distance_scale=3.0,      # 距離感度（緑色時の明度補強用）
        distance_decay=0.98,     # 減衰率（1.0に近いほど減衰が遅い）
        curve_power=1.0          # 明度カーブ
    )
    
    try:
        pipeline.start()
        
        print("実行中... (Ctrl+C で停止)")
        print("動作説明:")
        print("  速い動き → 赤色で明るく点灯")
        print("  ゆっくりな動き → 緑色で点灯（累積距離で明度補強）")
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