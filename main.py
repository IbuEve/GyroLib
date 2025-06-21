import time
from module.SensorReceiver import GyroStickReceiver
from module.SensorMath import SensorDataProcessor
from module.SensorVisualizer import SensorDataVisualizer

def main():
    # 各クラスのインスタンス作成
    receiver = GyroStickReceiver(host='192.168.1.164', port=12351)
    processor = SensorDataProcessor()
    visualizer = SensorDataVisualizer(max_points=500)
    
    # データ受信時のコールバック関数
    def on_data_received(parsed_data):
        # データ処理
        processed_data = processor.process_sensor_data(parsed_data)
        
        # 可視化データに追加
        visualizer.add_data(processed_data)
        
        # デバッグ出力
        euler_rates = processed_data['euler_rates']
        print(f"Euler Rates - Roll: {euler_rates[0]:.2f}, "
              f"Pitch: {euler_rates[1]:.2f}, Yaw: {euler_rates[2]:.2f}")
    
    # エラー処理のコールバック関数
    def on_error(error_message):
        print(f"Error: {error_message}")
    
    # コールバック設定
    receiver.set_data_callback(on_data_received)
    receiver.set_error_callback(on_error)
    
    # データ受信開始
    receiver.start_receiving()
    
    # 可視化開始
    visualizer.setup_plots()
    visualizer.start_animation(interval=100)
    
    try:
        # センサー開始
        time.sleep(1)  # 接続待ち
        receiver.start_sensor()
        
        # グラフ表示
        visualizer.show()
        
    except KeyboardInterrupt:
        print("Closing application...")
    finally:
        # クリーンアップ
        receiver.stop_receiving()
        print("Application closed")

if __name__ == "__main__":
    main()