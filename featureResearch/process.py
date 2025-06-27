import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_sensor_data(csv_file):
    """CSVファイルのセンサーデータをグラフ表示"""
    # データ読み込み
    df = pd.read_csv(csv_file)
    
    # 出力フォルダを作成
    output_folder = create_output_folder(csv_file)
    
    # 時間軸を作成（最初のタイムスタンプを0とする相対時間）
    time_relative = df['timestamp'] - df['timestamp'].iloc[0]
    
    # 4つのサブプロット作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Sensor Data: {os.path.basename(csv_file)}', fontsize=16)
    
    # 1. グローバル加速度
    axes[0, 0].plot(time_relative, df['global_accel_x'], label='X', color='red')
    axes[0, 0].plot(time_relative, df['global_accel_y'], label='Y', color='green')
    axes[0, 0].plot(time_relative, df['global_accel_z'], label='Z', color='blue')
    axes[0, 0].set_title('Global Acceleration')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Acceleration (m/s²)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. 速度
    axes[0, 1].plot(time_relative, df['velocity_x'], label='X', color='red')
    axes[0, 1].plot(time_relative, df['velocity_y'], label='Y', color='green')
    axes[0, 1].plot(time_relative, df['velocity_z'], label='Z', color='blue')
    axes[0, 1].set_title('Velocity')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. オイラー角レート
    axes[1, 0].plot(time_relative, df['euler_rate_roll'], label='Roll', color='red')
    axes[1, 0].plot(time_relative, df['euler_rate_pitch'], label='Pitch', color='green')
    axes[1, 0].plot(time_relative, df['euler_rate_yaw'], label='Yaw', color='blue')
    axes[1, 0].set_title('Euler Rates (Angular Velocity)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. ローカル角速度
    axes[1, 1].plot(time_relative, df['local_gyro_x'], label='X', color='red')
    axes[1, 1].plot(time_relative, df['local_gyro_y'], label='Y', color='green')
    axes[1, 1].plot(time_relative, df['local_gyro_z'], label='Z', color='blue')
    axes[1, 1].set_title('Local Gyroscope')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 画像を保存
    sensor_plot_path = os.path.join(output_folder, 'sensor_data.png')
    plt.savefig(sensor_plot_path, dpi=300, bbox_inches='tight')
    print(f"Sensor data plot saved to: {sensor_plot_path}")
    
    plt.close()  # メモリリーク防止のためプロットを閉じる

def simple_fft_analysis(csv_file, column_name='global_accel_x', sampling_rate=100):
    """指定した列に対して簡単なFFT解析を実行"""
    # データ読み込み
    df = pd.read_csv(csv_file)
    signal = df[column_name].values
    
    # 出力フォルダを取得
    output_folder = create_output_folder(csv_file)
    
    # FFT実行
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
    
    # パワースペクトルを計算
    power = np.abs(fft_result)**2
    
    # 正の周波数のみ取得
    positive_freq_indices = frequencies >= 0
    frequencies_positive = frequencies[positive_freq_indices]
    power_positive = power[positive_freq_indices]
    
    # 主要な周波数を検出
    max_power_index = np.argmax(power_positive[1:]) + 1  # DC成分を除く
    dominant_frequency = frequencies_positive[max_power_index]
    
    # 結果表示
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 元信号
    time_axis = np.arange(len(signal)) / sampling_rate
    ax1.plot(time_axis, signal)
    ax1.set_title(f'Original Signal: {column_name}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # パワースペクトル
    ax2.plot(frequencies_positive, power_positive)
    ax2.set_title(f'Power Spectrum (Dominant Frequency: {dominant_frequency:.2f} Hz)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power')
    ax2.set_xlim(0, 50)  # 0-50Hzの範囲を表示
    ax2.grid(True)
    
    # 主要周波数にマーカー
    ax2.axvline(x=dominant_frequency, color='red', linestyle='--', 
                label=f'Peak: {dominant_frequency:.2f} Hz')
    ax2.legend()
    
    plt.tight_layout()
    
    # 画像を保存
    fft_plot_path = os.path.join(output_folder, f'fft_analysis_{column_name}.png')
    plt.savefig(fft_plot_path, dpi=300, bbox_inches='tight')
    print(f"FFT analysis plot saved to: {fft_plot_path}")
    
    plt.close()  # メモリリーク防止のためプロットを閉じる
    
    print(f"Dominant frequency: {dominant_frequency:.2f} Hz")
    print(f"Signal length: {len(signal)} samples")
    print(f"Duration: {len(signal)/sampling_rate:.2f} seconds")
    
    return dominant_frequency, frequencies_positive, power_positive

def create_output_folder(csv_file):
    """CSVファイルと同じ階層に同じ名前のフォルダを作成"""
    # CSVファイルのディレクトリとファイル名（拡張子なし）を取得
    csv_dir = os.path.dirname(csv_file)
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
    
    # 出力フォルダのパスを作成
    output_folder = os.path.join(csv_dir, csv_basename)
    
    # フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)
    
    return output_folder

def process_folder(folder_path):
    """指定フォルダ内のすべてのCSVファイルを処理"""
    # フォルダ内のCSVファイルをすべて取得
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {folder_path}")
    print("Files to process:", [os.path.basename(f) for f in csv_files])
    print("=" * 50)
    
    # FFT解析対象のカラム一覧
    fft_columns = [
        'local_accel_x', 'local_accel_y', 'local_accel_z',
        'global_accel_x', 'global_accel_y', 'global_accel_z',
        'local_gyro_x', 'local_gyro_y', 'local_gyro_z',
        'euler_rate_roll', 'euler_rate_pitch', 'euler_rate_yaw'
    ]
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {os.path.basename(csv_file)}")
        
        try:
            # センサーデータのプロット
            plot_sensor_data(csv_file)
            
            # 全ての指定カラムに対してFFT解析を実行
            for column in fft_columns:
                print(f"=== {column} ===")
                simple_fft_analysis(csv_file, column)
            
            print(f"✓ Completed processing: {os.path.basename(csv_file)}")
            
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(csv_file)}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Finished processing all files in {folder_path}")

# 使用例
if __name__ == "__main__":
    # フォルダパスを指定
    folder_path = "data\\20250621_164208"  # 適切なフォルダパスに変更
    process_folder(folder_path)