import numpy as np
import time
import threading
from module.SensorReceiver import GyroStickReceiver
from module.SensorMath import SensorDataProcessor

class ZoneBasedLED:
    """ゾーンベースのLED制御"""
    
    def __init__(self, receiver):
        self.receiver = receiver
        
        # ゾーン設定
        self.calm_threshold = 0.5      # 落ち着きゾーンの上限速度
        self.intense_threshold = 1.5   # 激しいゾーンの下限速度
        
        # 各ゾーンの速度スケール（明度計算用）
        self.calm_velocity_scale = 0.8     # 落ち着きゾーン用
        self.intense_velocity_scale = 3.0  # 激しいゾーン用
        self.button_velocity_scale = 1.5   # ボタン押下時用
        
        # 明度カーブ
        self.curve_power = 1.5
        
        # 現在のゾーン状態
        self.current_zone = 'calm'  # 'calm', 'intense', 'transition'
        self.zone_lock_time = 0.0   # ゾーン固定時間
        self.zone_lock_duration = 0.3  # 最小ゾーン維持時間（秒）
        
        # 赤ゾーンからの減衰制御
        self.in_red_decay = False
        self.red_decay_start_time = None
        self.red_decay_duration = 1.0  # 赤からの減衰時間（秒）
        
    def _determine_zone(self, speed, current_time, button_pressed):
        """現在の速度とボタン状態からゾーンを決定"""
        
        # ボタンが押されている場合は白色優先
        if button_pressed:
            return 'button'
        
        # ゾーン固定時間中は変更しない
        if current_time - self.zone_lock_time < self.zone_lock_duration:
            if self.current_zone in ['calm', 'intense']:
                return self.current_zone
        
        # 赤ゾーン減衰中の特殊処理
        if self.in_red_decay:
            decay_elapsed = current_time - self.red_decay_start_time
            if decay_elapsed < self.red_decay_duration:
                # 減衰時間内は赤を維持
                return 'red_decay'
            else:
                # 減衰完了
                self.in_red_decay = False
                self.red_decay_start_time = None
        
        # 通常のゾーン判定
        if speed >= self.intense_threshold:
            new_zone = 'intense'
        elif speed <= self.calm_threshold:
            new_zone = 'calm'
        else:
            # 中間帯域：現在のゾーンを維持
            new_zone = self.current_zone if self.current_zone in ['calm', 'intense'] else 'calm'
        
        # ゾーン変更時の処理
        if new_zone != self.current_zone:
            # 激しいゾーンから落ち着きゾーンへの移行時
            if self.current_zone == 'intense' and new_zone == 'calm':
                # 赤ゾーン減衰を開始
                self.in_red_decay = True
                self.red_decay_start_time = current_time
                self.zone_lock_time = current_time
                return 'red_decay'
            else:
                self.zone_lock_time = current_time
        
        return new_zone
    
    def _calculate_brightness(self, speed, zone):
        """ゾーンと速度から明度を計算"""
        if zone == 'button':
            # ボタン押下時：専用スケールで速度に応じた明度
            normalized = min(speed / self.button_velocity_scale, 1.0)
            
        elif zone == 'calm':
            # 落ち着きゾーン：専用スケールで計算
            normalized = min(speed / self.calm_velocity_scale, 1.0)
            
        elif zone == 'intense':
            # 激しいゾーン：専用スケールで計算
            normalized = min(speed / self.intense_velocity_scale, 1.0)
            
        elif zone == 'red_decay':
            # 赤ゾーン減衰：時間経過で明度を下げる
            decay_elapsed = time.time() - self.red_decay_start_time
            decay_ratio = 1.0 - (decay_elapsed / self.red_decay_duration)
            decay_ratio = max(0.0, decay_ratio)
            
            # 速度ベースの明度も考慮
            speed_normalized = min(speed / self.intense_velocity_scale, 1.0)
            normalized = max(speed_normalized, decay_ratio * 0.5)  # 最低限の明度を保証
            
        else:
            normalized = 0.0
        
        # カーブ適用
        curved_brightness = normalized ** self.curve_power
        return curved_brightness
    
    def _get_zone_color(self, zone, brightness):
        """ゾーンから基本色を取得"""
        if zone == 'button':
            # 白色（速度に応じた明度）
            base_color = (255, 255, 255)
        elif zone == 'calm':
            # 緑色
            base_color = (0, 255, 0)
        elif zone in ['intense', 'red_decay']:
            # 赤色
            base_color = (255, 0, 0)
        else:
            base_color = (0, 0, 0)
        
        # 明度適用
        brightness_value = int(brightness * 255)
        final_color = tuple(int(c * brightness_value / 255) for c in base_color)
        
        return final_color, brightness_value
    
    def update_led(self, velocity, current_time, button_pressed):
        """LEDを更新"""
        speed = np.linalg.norm(velocity)
        
        # ゾーン決定
        zone = self._determine_zone(speed, current_time, button_pressed)
        
        # 明度計算
        brightness = self._calculate_brightness(speed, zone)
        
        # 色計算
        final_color, brightness_value = self._get_zone_color(zone, brightness)
        
        # ゾーン更新
        self.current_zone = zone
        
        # LED設定
        self.receiver.led_on(*final_color)
        
        return {
            'zone': zone,
            'speed': speed,
            'brightness': brightness_value,
            'color': final_color,
            'in_red_decay': self.in_red_decay
        }
    
    def configure(self, calm_threshold=None, intense_threshold=None,
                 calm_velocity_scale=None, intense_velocity_scale=None,
                 button_velocity_scale=None, curve_power=None, 
                 zone_lock_duration=None, red_decay_duration=None):
        """設定変更"""
        if calm_threshold is not None:
            self.calm_threshold = calm_threshold
            print(f"落ち着きゾーン閾値更新: {calm_threshold}")
        
        if intense_threshold is not None:
            self.intense_threshold = intense_threshold
            print(f"激しいゾーン閾値更新: {intense_threshold}")
        
        if calm_velocity_scale is not None:
            self.calm_velocity_scale = calm_velocity_scale
            print(f"落ち着きゾーン速度スケール更新: {calm_velocity_scale}")
        
        if intense_velocity_scale is not None:
            self.intense_velocity_scale = intense_velocity_scale
            print(f"激しいゾーン速度スケール更新: {intense_velocity_scale}")
        
        if button_velocity_scale is not None:
            self.button_velocity_scale = button_velocity_scale
            print(f"ボタン押下時速度スケール更新: {button_velocity_scale}")
        
        if curve_power is not None:
            self.curve_power = curve_power
            print(f"明度カーブ更新: {curve_power}")
        
        if zone_lock_duration is not None:
            self.zone_lock_duration = zone_lock_duration
            print(f"ゾーン固定時間更新: {zone_lock_duration}")
        
        if red_decay_duration is not None:
            self.red_decay_duration = red_decay_duration
            print(f"赤ゾーン減衰時間更新: {red_decay_duration}")

class ZoneBasedPipeline:
    """ゾーンベースLED制御パイプライン"""
    
    def __init__(self, enable_data_save=False):
        self.receiver = GyroStickReceiver(enable_data_save=enable_data_save)
        self.processor = SensorDataProcessor()
        self.led_controller = ZoneBasedLED(self.receiver)
        
        self.receiver.set_data_callback(self._on_data_received)
        self.receiver.set_error_callback(self._on_error)
        
        # キーボード入力用
        self.input_thread = None
        self.running = False
        
        # デバッグ用
        self.last_debug_time = 0
        self.debug_interval = 0.1  # デバッグ出力間隔（秒）
    
    def _on_data_received(self, parsed_data):
        """データ受信時の処理"""
        try:
            current_time = time.time()
            
            # センサーデータ処理
            processed_data = self.processor.process_sensor_data(parsed_data, current_time)
            
            # 速度とボタン状態取得
            velocity = processed_data['velocity']
            button_pressed = processed_data['button']
            
            # LED更新
            led_info = self.led_controller.update_led(velocity, current_time, button_pressed)
            
            # デバッグ出力（間隔制御）
            if current_time - self.last_debug_time >= self.debug_interval:
                if led_info['brightness'] > 5 or led_info['zone'] != 'calm':
                    zone_indicator = {
                        'calm': '🟢',
                        'intense': '🔴',
                        'red_decay': '🟠',
                        'button': '⚪'
                    }.get(led_info['zone'], '❓')
                    
                    decay_status = " [減衰中]" if led_info['in_red_decay'] else ""
                    button_status = " [ボタン]" if led_info['zone'] == 'button' else ""
                    
                    print(f"{zone_indicator} {led_info['zone'].upper()} | "
                          f"速度: {led_info['speed']:.2f} | "
                          f"明度: {led_info['brightness']} | "
                          f"RGB: {led_info['color']}{decay_status}{button_status}")
                
                self.last_debug_time = current_time
                
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
        print("=== ゾーンベースLED制御 ===")
        print("ゾーン構成:")
        print(f"  🟢 落ち着きゾーン: 速度 ≤ {self.led_controller.calm_threshold}")
        print(f"  🔴 激しいゾーン: 速度 ≥ {self.led_controller.intense_threshold}")
        print(f"  ⚪ ボタンゾーン: ボタン押下時（白色、速度連動）")
        print("")
        print("特徴:")
        print("  - 各ゾーンで独立した速度スケール")
        print("  - ボタン押下時も速度に応じた明度変化")
        print("  - 赤→緑移行時は赤色減衰で自然な変化")
        print("  - ゾーン切り替え時の安定化機能")
        print(f"  - 落ち着き速度スケール: {self.led_controller.calm_velocity_scale}")
        print(f"  - 激しい速度スケール: {self.led_controller.intense_velocity_scale}")
        print(f"  - ボタン速度スケール: {self.led_controller.button_velocity_scale}")
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
        self.receiver.led_on(0, 0, 0)  # 最後に確実に消灯
        self.receiver.stop_receiving()
        print("停止完了")
    
    def configure(self, **kwargs):
        """設定変更"""
        self.led_controller.configure(**kwargs)

# 使用例
if __name__ == "__main__":
    pipeline = ZoneBasedPipeline(enable_data_save=True)
    
    # 設定調整
    pipeline.configure(
        calm_threshold=0.7,           # 落ち着きゾーン上限
        intense_threshold=0.8,        # 激しいゾーン下限
        calm_velocity_scale=0.6,      # 落ち着きゾーン感度
        intense_velocity_scale=1.5,   # 激しいゾーン感度
        button_velocity_scale=1.5,    # ボタン押下時感度
        curve_power=2,              # 明度カーブ
        zone_lock_duration=0.8,       # ゾーン固定時間
        red_decay_duration=0.4        # 赤減衰時間
    )
    
    try:
        pipeline.start()
        
        print("実行中... (Ctrl+C で停止)")
        print("")
        print("動作説明:")
        print("  🟢 ゆっくり動く → 緑色で速度に応じた明度")
        print("  🔴 激しく動く → 赤色で速度に応じた明度")
        print("  🟠 赤→緑移行 → 赤色減衰（緑の帯域を回避）")
        print("  ⚪ ボタン押下 → 白色で速度に応じた明度")
        print("")
        
        # メインループ（キーボード入力は別スレッドで処理）
        while pipeline.running:
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n終了中...")
        pipeline.stop()
        print("プログラム終了")