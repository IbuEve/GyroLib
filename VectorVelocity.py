import numpy as np
import time
import math
import threading
from collections import deque
from module.SensorReceiver import GyroStickReceiver
from module.SensorMath import SensorDataProcessor

def convert_to_led_rgb(color_ratio, brightness_scale):
    """
    色の割合と明度スケールからLED用RGB値に変換
    
    Args:
        color_ratio: (r_ratio, g_ratio, b_ratio) 各色の割合 0.0-1.0
        brightness_scale: 明度スケール 0.0-1.0
    
    Returns:
        (r, g, b): LED用RGB値 0-255
    """
    r_ratio, g_ratio, b_ratio = color_ratio
    
    # 割合を0-1の範囲にクランプ
    r_ratio = max(0.0, min(1.0, r_ratio))
    g_ratio = max(0.0, min(1.0, g_ratio))
    b_ratio = max(0.0, min(1.0, b_ratio))
    
    # 明度スケールを0-1の範囲にクランプ
    brightness_scale = max(0.0, min(1.0, brightness_scale))
    
    # RGB値を計算
    r = int(r_ratio * brightness_scale * 255)
    g = int(g_ratio * brightness_scale * 255)
    b = int(b_ratio * brightness_scale * 255)
    
    return (r, g, b)

class DirectionVelocityLED:
    """方向で色、速度で明度を制御するLED"""
    
    def __init__(self, receiver):
        self.receiver = receiver
        self.velocity_scale = 2.0
        self.curve_power = 2.0
        self.delay_seconds = 0.0  # 遅延時間（秒）
        
        # 遅延用のデータ履歴
        self.led_history = deque()  # (timestamp, led_rgb) のタプルを保存
        
        # 前回の時刻
        self.last_time = None
        self.last_velocity = np.array([0.0, 0.0, 0.0])
    
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

    def create_color_rotation_matrix(self, angle_degrees, axis='z'):
        """
        色空間での回転行列を作成
        
        Args:
            angle_degrees: 回転角度（度）
            axis: 回転軸 'x', 'y', 'z'
        """
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        if axis == 'x':  # R軸周りの回転（G-B平面）
            return np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        elif axis == 'y':  # G軸周りの回転（R-B平面）
            return np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        elif axis == 'z':  # B軸周りの回転（R-G平面）
            return np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])

    def direction_to_color_ratio(self, direction_vector, rotation_angle=20, rotation_axis='x'):
        """方向ベクトルを色の割合に変換"""
        # -1~1 の範囲を 0~1 に変換
        base_color = np.array([
            (direction_vector[0] + 1) * 0.5,
            (direction_vector[1] + 1) * 0.5,
            (direction_vector[2] + 1) * 0.5
        ])
    
        # 色空間で回転
        if rotation_angle != 0:
            rotation_matrix = self.create_color_rotation_matrix(rotation_angle, rotation_axis)
            rotated_color = rotation_matrix @ base_color
            # 0-1の範囲にクランプ
            rotated_color = np.clip(rotated_color, 0, 1)
        else:
            rotated_color = base_color
        
        return tuple(rotated_color)

    def velocity_to_brightness(self, velocity):
        """速度から明度スケールを計算"""
        # 速度の大きさから明度を計算
        velocity_squared_sum = sum(v**2 for v in velocity)
        linear_normalized = min(velocity_squared_sum / (self.velocity_scale**2), 1.0)
        curved_normalized = linear_normalized ** self.curve_power
        
        return curved_normalized
    
    def update_led(self, quaternion, velocity, current_time):
        """方向と速度からLEDを制御（遅延機能付き）"""
        # 回転行列を計算
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        
        # X軸方向ベクトル（スティックが指す方向）
        x_direction = rotation_matrix[:, 0]
        
        # 方向から色の割合を計算
        color_ratio = self.direction_to_color_ratio(x_direction)
        
        # 速度から明度スケールを計算
        brightness_scale = self.velocity_to_brightness(velocity)
        
        # LED用RGB値に変換
        led_rgb = convert_to_led_rgb(color_ratio, brightness_scale)
        
        # 遅延処理
        if self.delay_seconds > 0:
            # 現在のデータを履歴に追加
            self.led_history.append((current_time, led_rgb))
            
            # 古いデータを削除（遅延時間より古いもの）
            cutoff_time = current_time - self.delay_seconds * 2  # 余裕を持って削除
            while self.led_history and self.led_history[0][0] < cutoff_time:
                self.led_history.popleft()
            
            # 遅延時間前のデータを探す
            target_time = current_time - self.delay_seconds
            delayed_led_rgb = (0, 0, 0)  # デフォルト値
            
            for timestamp, rgb in self.led_history:
                if timestamp <= target_time:
                    delayed_led_rgb = rgb
                else:
                    break
            
            # 遅延されたLED値を適用
            self.receiver.led_on(*delayed_led_rgb)
            actual_led_rgb = delayed_led_rgb
        else:
            # 遅延なし
            self.receiver.led_on(*led_rgb)
            actual_led_rgb = led_rgb
        
        return actual_led_rgb, x_direction, color_ratio, brightness_scale

class DirectionVelocityPipeline:
    """方向と速度制御LED制御パイプライン"""
    
    def __init__(self, enable_data_save=False):
        self.receiver = GyroStickReceiver(enable_data_save=enable_data_save)
        self.processor = SensorDataProcessor()
        self.led_controller = DirectionVelocityLED(self.receiver)
        
        self.receiver.set_data_callback(self._on_data_received)
        self.receiver.set_error_callback(self._on_error)
        
        # キーボード入力用
        self.input_thread = None
        self.running = False
    
    def _on_data_received(self, parsed_data):
        """データ受信時の処理"""
        try:
            current_time = time.time()
            
            # センサーデータ処理
            processed_data = self.processor.process_sensor_data(parsed_data, current_time)
            
            # 姿勢と速度取得
            quaternion = processed_data['quaternion']
            velocity = processed_data['velocity']
            
            # LED更新
            led_rgb, direction, color_ratio, brightness = self.led_controller.update_led(
                quaternion, velocity, current_time
            )
            
            # デバッグ出力
            if brightness > 0.1:
                velocity_magnitude = np.linalg.norm(velocity)
                delay_status = f"遅延{self.led_controller.delay_seconds:.1f}s" if self.led_controller.delay_seconds > 0 else "即座"
                print(f"方向: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}] | "
                        f"速度: {velocity_magnitude:.2f} | "
                        f"色割合: ({color_ratio[0]:.2f}, {color_ratio[1]:.2f}, {color_ratio[2]:.2f}) | "
                        f"明度: {brightness:.2f} | RGB: {led_rgb} | {delay_status}")
                
        except Exception as e:
            print(f"データ処理エラー: {e}")
    
    def _on_error(self, error_msg):
        """エラー処理"""
        print(f"受信エラー: {error_msg}")
    
    def _keyboard_input_loop(self):
        """キーボード入力処理ループ"""
        print("\nキーボードコマンド:")
        print("  'l' + Enter → LED制御 オン/オフ切り替え")
        print("  'q' + Enter → 終了")
        print()
        
        while self.running:
            try:
                user_input = input().strip().lower()
                
                if user_input == 'l':
                    # LED制御切り替え
                    self.receiver.toggle_led_control()
                elif user_input == 'q':
                    # 終了
                    print("終了コマンドが入力されました")
                    self.stop()
                    break
                    
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"入力エラー: {e}")
    
    def start(self):
        """開始"""
        print("=== 方向×速度制御LED（遅延機能付き） ===")
        print("特徴:")
        print("- スティックの向き → RGB色の割合")
        print("- 速度の大きさ → 明度")
        print(f"速度スケール: {self.led_controller.velocity_scale}")
        print(f"明度カーブ: {self.led_controller.curve_power}")
        print(f"遅延時間: {self.led_controller.delay_seconds}秒")
        print("")
        
        self.running = True
        self.receiver.start_receiving()
        
        # キーボード入力スレッドを開始
        self.input_thread = threading.Thread(target=self._keyboard_input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
    
    def stop(self):
        """停止"""
        print("停止中...")
        self.running = False
        self.receiver.led_on(0, 0, 0)
        self.receiver.stop_receiving()
        print("停止完了")
    
    def configure(self, velocity_scale=None, curve_power=None, delay_seconds=None):
        """設定変更"""
        if velocity_scale is not None:
            self.led_controller.velocity_scale = velocity_scale
            print(f"速度スケール更新: {velocity_scale}")
        
        if curve_power is not None:
            self.led_controller.curve_power = curve_power
            print(f"カーブ強度更新: {curve_power}")
        
        if delay_seconds is not None:
            self.led_controller.delay_seconds = delay_seconds
            print(f"遅延時間更新: {delay_seconds}秒")

    # 使用例    
if __name__ == "__main__":
    pipeline = DirectionVelocityPipeline(enable_data_save=True)
    
    # 設定調整
    pipeline.configure(
        velocity_scale=0.8,        # 速度感度（小さいほど敏感）
        curve_power=2.5,         # 明度カーブ
        delay_seconds=0
    )
    
    try:
        pipeline.start()
        
        print("実行中... (Ctrl+C で停止)")
        
        # メインループ（キーボード入力は別スレッドで処理）
        while pipeline.running:
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n終了中...")
        pipeline.stop()
        print("プログラム終了")