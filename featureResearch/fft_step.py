import pandas as pd
import numpy as np
import time
import json
import os
from scipy.signal import find_peaks

def adaptive_frequency_detection(csv_file, column_name='local_gyro_z', sampling_rate=100):
    """直近のステップ数を変えながら順次周波数を計算"""
    
    # データ読み込み
    df = pd.read_csv(csv_file)
    signal = df[column_name].values
    
    print(f"=== Adaptive Frequency Detection: {column_name} ===")
    print(f"Total samples: {len(signal)}")
    print()
    
    # 結果を保存する辞書
    results = {
        'column_name': column_name,
        'total_samples': len(signal),
        'sampling_rate': sampling_rate,
        'step_analysis': [],
        'full_signal_analysis': {}
    }
    
    # 直近100, 200, 300, ..., 1000ステップで解析
    step_sizes = list(range(100, 1100, 100))  # [100, 200, 300, ..., 1000]
    
    for step_size in step_sizes:
        if step_size > len(signal):
            print(f"Step {step_size}: データ不足 (必要: {step_size}, 実際: {len(signal)})")
            results['step_analysis'].append({
                'step_size': step_size,
                'status': 'insufficient_data',
                'required': step_size,
                'actual': len(signal)
            })
            continue
            
        start_time = time.time()
        
        # 直近のデータを取得
        recent_data = signal[-step_size:]
        
        # FFT実行
        fft_result = np.fft.fft(recent_data)
        frequencies = np.fft.fftfreq(step_size, 1/sampling_rate)
        power = np.abs(fft_result)**2
        
        # 正の周波数のみ
        positive_indices = frequencies > 0  # DC成分も除外
        freq_pos = frequencies[positive_indices]
        power_pos = power[positive_indices]
        
        # ピーク検出
        peaks, _ = find_peaks(power_pos, height=np.max(power_pos)*0.01)
        
        # パワーでソート（降順）
        if len(peaks) > 0:
            peak_powers = power_pos[peaks]
            sorted_indices = np.argsort(peak_powers)[::-1]
            
            # 1st peak
            first_peak_idx = peaks[sorted_indices[0]]
            first_freq = freq_pos[first_peak_idx]
            first_power = float(power_pos[first_peak_idx])
            
            # 2nd peak
            if len(sorted_indices) > 1:
                second_peak_idx = peaks[sorted_indices[1]]
                second_freq = freq_pos[second_peak_idx]
                second_power = float(power_pos[second_peak_idx])
            else:
                second_freq = 0.0
                second_power = 0.0
        else:
            first_freq, first_power = 0.0, 0.0
            second_freq, second_power = 0.0, 0.0
        
        calc_time = (time.time() - start_time) * 1000  # ms
        
        # 結果を記録
        step_result = {
            'step_size': step_size,
            'first_peak': {
                'frequency': float(first_freq),
                'power': first_power
            },
            'second_peak': {
                'frequency': float(second_freq),
                'power': second_power
            },
            'calculation_time_ms': calc_time
        }
        results['step_analysis'].append(step_result)
        
        print(f"Step {step_size:4d}: 1st={first_freq:5.2f}Hz (power={first_power:.2e}), "
              f"2nd={second_freq:5.2f}Hz (power={second_power:.2e}), "
              f"time={calc_time:.2f}ms")
    
    print()
    
    # 全体でFFT
    print("=== Full Signal Analysis ===")
    start_time = time.time()
    
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
    power = np.abs(fft_result)**2
    
    positive_indices = frequencies > 0
    freq_pos = frequencies[positive_indices]
    power_pos = power[positive_indices]
    
    peaks, _ = find_peaks(power_pos, height=np.max(power_pos)*0.01)
    
    if len(peaks) > 0:
        peak_powers = power_pos[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1]
        
        first_peak_idx = peaks[sorted_indices[0]]
        first_freq = freq_pos[first_peak_idx]
        first_power = float(power_pos[first_peak_idx])
        
        if len(sorted_indices) > 1:
            second_peak_idx = peaks[sorted_indices[1]]
            second_freq = freq_pos[second_peak_idx]
            second_power = float(power_pos[second_peak_idx])
        else:
            second_freq = 0.0
            second_power = 0.0
    else:
        first_freq, first_power = 0.0, 0.0
        second_freq, second_power = 0.0, 0.0
    
    calc_time = (time.time() - start_time) * 1000
    
    # 全体解析結果を記録
    results['full_signal_analysis'] = {
        'samples': len(signal),
        'first_peak': {
            'frequency': float(first_freq),
            'power': first_power
        },
        'second_peak': {
            'frequency': float(second_freq),
            'power': second_power
        },
        'calculation_time_ms': calc_time
    }
    
    print(f"Full signal ({len(signal)} samples): 1st={first_freq:5.2f}Hz (power={first_power:.2e}), "
          f"2nd={second_freq:5.2f}Hz (power={second_power:.2e}), "
          f"time={calc_time:.2f}ms")
    
    return results

def process_all_columns(csv_file, sampling_rate=100):
    """全ての指定カラムについて解析を実行"""
    
    fft_columns = [
        'local_accel_x', 'local_accel_y', 'local_accel_z',
        'global_accel_x', 'global_accel_y', 'global_accel_z',
        'local_gyro_x', 'local_gyro_y', 'local_gyro_z',
        'euler_rate_roll', 'euler_rate_pitch', 'euler_rate_yaw'
    ]
    
    # 全体の結果を保存する辞書
    all_results = {
        'csv_file': csv_file,
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sampling_rate': sampling_rate,
        'columns_analyzed': fft_columns,
        'results': {}
    }
    
    print(f"Processing file: {csv_file}")
    print(f"Analyzing {len(fft_columns)} columns")
    print("=" * 60)
    
    for i, column in enumerate(fft_columns, 1):
        print(f"\n[{i}/{len(fft_columns)}] Processing column: {column}")
        try:
            column_result = adaptive_frequency_detection(csv_file, column, sampling_rate)
            all_results['results'][column] = column_result
            print(f"✓ Completed: {column}")
        except Exception as e:
            print(f"✗ Error processing {column}: {e}")
            all_results['results'][column] = {
                'error': str(e),
                'status': 'failed'
            }
    
    # JSONファイルに保存
    output_file = os.path.splitext(csv_file)[0] + '_fft_analysis.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Analysis completed!")
    print(f"Results saved to: {output_file}")
    
    return all_results

# 使用例
if __name__ == "__main__":
    csv_file = "data\\20250621_164208\\1.csv"  # ファイルパスを変更
    results = process_all_columns(csv_file)