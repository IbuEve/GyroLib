import threading
import time
import math
import numpy as np

class MotionLEDController:
    """動作連動LED制御クラス"""
    
    def __init__(self, receiver):
        self.receiver = receiver
        self.is_animating = False
        self.animation_thread = None
        self.animation_stop_flag = False
        self.current_mode = 'idle'
        
        # 幻想的な色パレット（低エネルギー用）
        self.fantasy_colors = [
            (100, 255, 200),  # 薄い緑
            (150, 255, 255),  # 水色  
            (100, 200, 255),  # 薄い青
            (150, 255, 180),  # 薄い青緑
        ]
        
        # アニメーション設定
        self.fantasy_frequency = 0.5  # 0.5Hz で色移動
        self.max_frequency = 3.0      # 最大周波数（これ以上は真っ赤）
        
    def process_motion_results(self, energy_result, frequency_result, jerk_result):
        """動作解析結果からLED表現を決定"""
        
        # 最優先: スナップフラッシュ
        if jerk_result and jerk_result['type'] == 'snap_detected':
            self.trigger_snap_flash()
            return
            
        # 周期性検出時：周波数連動脈動
        if frequency_result['detected']:
            detected_freq = frequency_result['best_frequency']
            self.start_frequency_pulse(detected_freq)
            return
            
        # 高エネルギー（周期性なし）：静的色
        if energy_result['high_energy']:
            color = self.energy_to_static_color(energy_result)
            self.set_static_color(color)
            return
            
        # 低エネルギー：幻想的な色移動
        self.start_fantasy_animation()
    
    def trigger_snap_flash(self):
        """スナップ時の白フラッシュ効果"""
        # 現在のアニメーションを一時停止
        self.stop_current_animation()
        
        def flash_effect():
            # 白フラッシュ
            self.receiver.led_on(255, 255, 255)
            time.sleep(0.05)  # 50ms 白く光る
            
            # フェードアウト
            for i in range(20):
                intensity = int(255 * (1 - i/20))
                self.receiver.led_on(intensity, intensity, intensity)
                time.sleep(0.01)  # 10ms間隔
            
            # 完全消灯
            self.receiver.led_on(0, 0, 0)
            time.sleep(0.1)
            
            # 元のモードに復帰
            self.resume_animation()
        
        # 別スレッドで実行
        flash_thread = threading.Thread(target=flash_effect)
        flash_thread.daemon = True
        flash_thread.start()
    
    def start_fantasy_animation(self):
        """幻想的な色移動アニメーション（低エネルギー時）"""
        if self.current_mode == 'fantasy':
            return  # 既に実行中
            
        self.stop_current_animation()
        self.current_mode = 'fantasy'
        self.animation_stop_flag = False
        
        def fantasy_loop():
            start_time = time.time()
            
            while not self.animation_stop_flag:
                current_time = time.time() - start_time
                
                # 0.5Hzで色を循環（2秒で1サイクル）
                cycle_position = (current_time * self.fantasy_frequency) % 1.0
                color_index = cycle_position * len(self.fantasy_colors)
                
                # 色の補間
                color = self.interpolate_colors(color_index)
                
                # LED設定
                self.receiver.led_on(*color)
                time.sleep(0.05)  # 20Hz更新
        
        self.animation_thread = threading.Thread(target=fantasy_loop)
        self.animation_thread.daemon = True
        self.animation_thread.start()
        self.is_animating = True
    
    def start_frequency_pulse(self, frequency):
        """周波数連動脈動アニメーション"""
        if self.current_mode == f'pulse_{frequency:.1f}':
            return  # 同じ周波数なら継続
            
        self.stop_current_animation()
        self.current_mode = f'pulse_{frequency:.1f}'
        self.animation_stop_flag = False
        
        def pulse_loop():
            start_time = time.time()
            base_color = self.frequency_to_color(frequency)
            
            while not self.animation_stop_flag:
                current_time = time.time() - start_time
                
                # 検出周波数で明度を変調
                pulse_phase = (current_time * frequency) % 1.0
                pulse_intensity = (math.sin(pulse_phase * 2 * math.pi) + 1) / 2
                
                # 明度変調（50%-100%の範囲）
                intensity_factor = 0.5 + 0.5 * pulse_intensity
                
                # 色に明度を適用
                pulsed_color = tuple(int(c * intensity_factor) for c in base_color)
                
                # LED設定
                self.receiver.led_on(*pulsed_color)
                time.sleep(0.02)  # 50Hz更新（滑らかな脈動）
        
        self.animation_thread = threading.Thread(target=pulse_loop)
        self.animation_thread.daemon = True
        self.animation_thread.start()
        self.is_animating = True
    
    def frequency_to_color(self, frequency):
        """周波数を色に変換（0Hz→幻想色、3Hz→赤）"""
        # 3Hz以上は真っ赤
        if frequency >= self.max_frequency:
            return (255, 0, 0)
        
        # 0-3Hzの範囲で補間
        ratio = frequency / self.max_frequency
        
        # ベース色（幻想的な色の平均）
        base_color = np.mean(self.fantasy_colors, axis=0)
        
        # 赤色への補間
        target_color = np.array([255, 0, 0])
        
        # 線形補間
        result_color = base_color * (1 - ratio) + target_color * ratio
        
        return tuple(int(c) for c in result_color)
    
    def interpolate_colors(self, color_index):
        """色パレットからスムーズに補間"""
        color_count = len(self.fantasy_colors)
        
        # 現在の色インデックス
        current_idx = int(color_index) % color_count
        next_idx = (current_idx + 1) % color_count
        
        # 補間比率
        t = color_index - int(color_index)
        
        # 線形補間
        current_color = np.array(self.fantasy_colors[current_idx])
        next_color = np.array(self.fantasy_colors[next_idx])
        
        interpolated = current_color * (1 - t) + next_color * t
        
        return tuple(int(c) for c in interpolated)
    
    def set_static_color(self, color):
        """静的な色を設定"""
        self.stop_current_animation()
        self.current_mode = 'static'
        self.receiver.led_on(*color)
    
    def energy_to_static_color(self, energy_result):
        """エネルギーレベルを静的な色に変換"""
        total_energy = sum(energy_result['vel_energies'].values()) + \
                      sum(energy_result['angular_energies'].values()) / 1000
        
        # エネルギーに応じて黄色→赤色
        if total_energy < 10:
            return (255, 255, 0)    # 黄色
        elif total_energy < 50:
            return (255, 128, 0)    # オレンジ
        else:
            return (255, 0, 0)      # 赤
    
    def stop_current_animation(self):
        """現在のアニメーションを停止"""
        self.animation_stop_flag = True
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.1)
        self.is_animating = False
    
    def resume_animation(self):
        """フラッシュ後にアニメーション復帰"""
        # 前のモードに応じて復帰
        if self.current_mode == 'fantasy':
            self.start_fantasy_animation()
        # その他の復帰処理...

# パイプラインに統合
class MotionAnalysisPipeline:
    def __init__(self):
        # ... 既存の初期化 ...
        self.led_controller = MotionLEDController(self.receiver)
    
    def _integrate_results(self, energy, frequency, jerk, timestamp):
        """結果統合 + LED制御"""
        # LED表現を更新
        self.led_controller.process_motion_results(energy, frequency, jerk)
        
        # ... 既存の統合処理 ...

# 使用例
if __name__ == "__main__":
    pipeline = MotionAnalysisPipeline()
    
    try:
        pipeline.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pipeline.led_controller.stop_current_animation()
        pipeline.stop()