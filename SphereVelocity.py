import numpy as np
import time
import threading
from module.SensorReceiver import GyroStickReceiver
from module.SensorMath import SensorDataProcessor, SensorMath

class DirectionBasedLED:
    """スティックの指す方向（Xベクトル）で色、速度で明度を制御するLED"""
    
    def __init__(self, receiver):
        self.receiver = receiver
        self.velocity_scale = 2.0
        self.curve_power = 2.0
        
        # 基準色（デフォルト姿勢時の色）
        self.base_color = [123, 123, 123]  # RGB基準値
        
        # 色の変動範囲
        self.color_range = 132  # ±132の範囲で変化（123±132 = -9~255の範囲）
        
    def quaternion_to_rotation_matrix(self, quaternion):
        """クォータニオンを回転行列に変換"""
        if isinstance(quaternion, dict):
            w, x, y, z = quaternion['w'], quaternion['x'], quaternion['y'], quaternion['z']
        else:
            w, x, y, z = quaternion
        
        # 正規化
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
    
    def update_led(self, velocity, quaternion):
        """速度と姿勢からLEDを制御"""
        # 速度の大きさから明度を計算（従来通り）
        velocity_squared_sum = sum(v**2 for v in velocity)
        linear_normalized = min(velocity_squared_sum / (self.velocity_scale**2), 1.0)
        curved_normalized = linear_normalized ** self.curve_power
        brightness_ratio = curved_normalized  # 0-1の範囲
        
        # クォータニオンから回転行列を計算
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        
        # スティックのX軸方向ベクトル（スティックが指す方向）
        x_direction = rotation_matrix[:, 0]  # [x, y, z]
        
        # 方向ベクトルから色を計算
        direction_color = self._get_color_from_direction(x_direction)
        
        # 明度を適用
        final_color = tuple(int(c * brightness_ratio) for c in direction_color)
        
        # LED設定
        self.receiver.led_on(*final_color)
        
        return final_color, x_direction, brightness_ratio
    
    def _get_color_from_direction(self, direction_vector):
        """方向ベクトルから色を決定"""
        x, y, z = direction_vector
        
        # 方向ベクトル（-1~1）を基準色からの変動に変換
        r = self.base_color[0] + int(x * self.color_range)
        g = self.base_color[1] + int(y * self.color_range)
        b = self.base_color[2] + int(z * self.color_range)
        
        # 0-255の範囲にクランプ
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return (r, g, b)

class DirectionPipeline:
    """方向ベース色変化LED制御パイプライン"""
    
    def __init__(self):
        self.receiver = GyroStickReceiver()
        self.processor = SensorDataProcessor()
        self.led_controller = DirectionBasedLED(self.receiver)
        
        self.receiver.set_data_callback(self._on_data_received)
        self.receiver.set_error_callback(self._on_error)
    
    def _on_data_received(self, parsed_data):
        """データ受信時の処理"""
        try:
            current_time = time.time()
            
            # センサーデータ処理
            processed_data = self.processor.process_sensor_data(parsed_data, current_time)
            
            # 速度と姿勢を取得
            velocity = processed_data['velocity']
            quaternion = processed_data['quaternion']
            
            # LED更新
            final_color, x_direction, brightness = self.led_controller.update_led(velocity, quaternion)
            
            # デバッグ出力
            if brightness > 0.02:  # 明度が一定以上の時のみ表示
                velocity_magnitude = np.linalg.norm(velocity)
                print(f"速度: {velocity_magnitude:.2f} | X方向: [{x_direction[0]:.2f}, {x_direction[1]:.2f}, {x_direction[2]:.2f}] | 明度: {brightness:.2f} | RGB: {final_color}")
                
        except Exception as e:
            print(f"データ処理エラー: {e}")
    
    def _on_error(self, error_msg):
        """エラー処理"""
        print(f"受信エラー: {error_msg}")
    
    def start(self):
        """開始"""
        print("=== 方向ベース色変化LED制御 ===")
        print("特徴:")
        print("- スティックの指す方向(X軸) → 色の変化")
        print("- 速度の大きさ → 明度の変化")
        print("- 天球探索のような色空間マッピング")
        print("")
        print("色マッピング:")
        print(f"  基準色: RGB({self.led_controller.base_color[0]}, {self.led_controller.base_color[1]}, {self.led_controller.base_color[2]})")
        print(f"  変動範囲: ±{self.led_controller.color_range}")
        print("  X方向 +1.0 → R成分 +132")
        print("  Y方向 +1.0 → G成分 +132") 
        print("  Z方向 +1.0 → B成分 +132")
        print("")
        print("操作:")
        print("  スティックを色々な方向に向けて色空間を探索")
        print("  動かすと明るく、静止すると暗く")
        print("")
        
        self.receiver.start_receiving()
    
    def stop(self):
        """停止"""
        print("停止中...")
        self.receiver.led_on(0, 0, 0)
        self.receiver.stop_receiving()
        print("停止完了")
    
    def configure(self, velocity_scale=None, base_color=None, 
                 color_range=None, curve_power=None):
        """設定変更"""
        if velocity_scale is not None:
            self.led_controller.velocity_scale = velocity_scale
            print(f"速度スケール更新: {velocity_scale}")
        
        if base_color is not None:
            self.led_controller.base_color = base_color
            print(f"基準色更新: RGB{base_color}")
        
        if color_range is not None:
            self.led_controller.color_range = color_range
            print(f"色変動範囲更新: ±{color_range}")
        
        if curve_power is not None:
            self.led_controller.curve_power = curve_power
            print(f"カーブ強度更新: {curve_power}")
    
    def show_direction_map(self):
        """方向と色の対応表を表示"""
        print("\n=== 方向→色マッピング例 ===")
        directions = [
            ([1, 0, 0], "右向き"),
            ([-1, 0, 0], "左向き"),
            ([0, 1, 0], "上向き"),
            ([0, -1, 0], "下向き"),
            ([0, 0, 1], "手前向き"),
            ([0, 0, -1], "奥向き"),
            ([0.707, 0.707, 0], "右上"),
            ([-0.707, -0.707, 0], "左下"),
        ]
        
        for direction, name in directions:
            color = self.led_controller._get_color_from_direction(direction)
            print(f"  {name:8s}: 方向{direction} → RGB{color}")
        print("")

# 使用例
if __name__ == "__main__":
    pipeline = DirectionPipeline()
    
    # 設定調整
    pipeline.configure(
        velocity_scale=1.5,      # 速度感度
        base_color=[128, 128, 128],  # より中央の基準色
        color_range=127,         # 変動範囲（128±127 = 1~255）
        curve_power=2.0          # 明度カーブ
    )
    
    try:
        pipeline.start()
        
        # 方向マッピング表示
        pipeline.show_direction_map()
        
        print("実行中... (Ctrl+C で停止)")
        print("スティックを色々な方向に向けて色の変化を楽しんでください！")
        print("各方向での色を覚えて、天球色空間を探索しましょう。")
        print("")
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n終了中...")
        pipeline.stop()
        print("プログラム終了")