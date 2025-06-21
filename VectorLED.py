import numpy as np
import time
import math
from module.SensorReceiver import GyroStickReceiver
from module.SensorMath import SensorDataProcessor

class RollControlledDirectionLED:
    """スティックの方向で色、ロール角で明度を制御するLED"""
    
    def __init__(self, receiver):
        self.receiver = receiver
    
    def quaternion_to_rotation_matrix(self, quaternion):
        """クォータニオンを回転行列に変換"""
        w, x, y, z = quaternion['w'], quaternion['x'], quaternion['y'], quaternion['z']
        
        # 正規化
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
    
    def quaternion_to_roll(self, quaternion):
        """クォータニオンからロール角を計算"""
        w, x, y, z = quaternion['w'], quaternion['x'], quaternion['y'], quaternion['z']
        
        # ロール角 (X軸周りの回転)
        roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        
        # ラジアンから度に変換
        roll_degrees = math.degrees(roll)
        
        return roll_degrees
    
    def calculate_brightness_from_roll(self, roll_degrees):
        """ロール角から明度を計算"""
        # ロール角の絶対値を取得
        abs_roll = abs(roll_degrees)
        
        # 0度で0、90度で1になるような計算
        # sin関数を使って滑らかに変化
        brightness = math.sin(math.radians(abs_roll))
        
        # 0-1の範囲にクランプ
        brightness = max(0, min(1, brightness))
        
        return brightness
    
    def update_led(self, quaternion):
        """方向とロール角からLEDを制御"""
        # 回転行列を計算
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        
        # X軸方向ベクトル（スティックが指す方向）
        x_direction = rotation_matrix[:, 0]
        
        # ロール角を計算
        roll_degrees = self.quaternion_to_roll(quaternion)
        
        # ロール角から明度を計算
        brightness = self.calculate_brightness_from_roll(roll_degrees)
        
        # 方向ベクトルを基本RGB値に変換
        base_r = int((x_direction[0] + 1) * 127.5)  # -1~1 を 0~255に
        base_g = int((x_direction[1] + 1) * 127.5)
        base_b = int((x_direction[2] + 1) * 127.5)
        
        # 明度を適用
        r = int(base_r * brightness)
        g = int(base_g * brightness)
        b = int(base_b * brightness)
        
        # 0-255の範囲にクランプ
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        # LED設定
        self.receiver.led_on(r, g, b)
        
        return (r, g, b), x_direction, roll_degrees, brightness

class RollPipeline:
    """ロール制御方向LED制御パイプライン"""
    
    def __init__(self):
        self.receiver = GyroStickReceiver()
        self.processor = SensorDataProcessor()
        self.led_controller = RollControlledDirectionLED(self.receiver)
        
        self.receiver.set_data_callback(self._on_data_received)
        self.receiver.set_error_callback(self._on_error)
    
    def _on_data_received(self, parsed_data):
        """データ受信時の処理"""
        try:
            current_time = time.time()
            
            # センサーデータ処理
            processed_data = self.processor.process_sensor_data(parsed_data, current_time)
            
            # 姿勢取得
            quaternion = processed_data['quaternion']
            
            # LED更新
            rgb_color, x_direction, roll_degrees, brightness = self.led_controller.update_led(quaternion)
            
            # デバッグ出力
            brightness_status = "消灯" if brightness < 0.1 else "点灯" if brightness > 0.9 else "調光"
            print(f"方向: [{x_direction[0]:.2f}, {x_direction[1]:.2f}, {x_direction[2]:.2f}] | "
                  f"ロール: {roll_degrees:.1f}° | 明度: {brightness:.2f} | RGB: {rgb_color} | {brightness_status}")
                
        except Exception as e:
            print(f"データ処理エラー: {e}")
    
    def _on_error(self, error_msg):
        """エラー処理"""
        print(f"受信エラー: {error_msg}")
    
    def start(self):
        """開始"""
        print("=== ロール制御方向LED ===")
        print("スティックの向き → RGB色")
        print("ロール角 → 明度制御")
        print("")
        print("色マッピング:")
        print("  X方向 → R値")
        print("  Y方向 → G値")
        print("  Z方向 → B値")
        print("")
        print("明度制御:")
        print("  ロール 0° (水平) → 消灯")
        print("  ロール ±45° → 中間の明るさ")
        print("  ロール ±90° (垂直) → 最大の明るさ")
        print("")
        
        self.receiver.start_receiving()
    
    def stop(self):
        """停止"""
        print("停止中...")
        self.receiver.led_on(0, 0, 0)
        self.receiver.stop_receiving()
        print("停止完了")

# 使用例
if __name__ == "__main__":
    pipeline = RollPipeline()
    
    try:
        pipeline.start()
        
        print("実行中... (Ctrl+C で停止)")
        print("操作:")
        print("  スティックを色々な方向に向ける → 色が変わる")
        print("  スティックを回転させる → 明度が変わる")
        print("  水平に持つ → 消灯")
        print("  垂直に立てる → 点灯")
        print("")
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n終了中...")
        pipeline.stop()
        print("プログラム終了")