import socket
import threading
import time
import re
import csv
import os
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

class GyroStickReceiver:
    """ジャイロスティックデータの受信クラス（データ保存機能付き）"""
    
    def __init__(self, host='172.20.10.4', port=12351, enable_data_save=False, save_folder="SaveData"):
        self.host = host
        self.port = port
        self.sock = None
        self.clients = None
        self.loop_flag = True
        self.data_callback = None
        self.error_callback = None
        
        # LED制御フラグを追加
        self.led_enabled = True  # LED制御の有効/無効
        
        # 日本標準時のタイムゾーンを定義
        self.jst = timezone(timedelta(hours=9))
        
        # データ保存機能
        self.enable_data_save = enable_data_save
        self.save_folder = save_folder
        self.data_saver = None
        self.file_lock = threading.Lock()
        
        if self.enable_data_save:
            self._initialize_data_save()
        
        # データ検証パターン
        self.data_pattern = re.compile(
            r'^(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)'
            r'\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)'
            r'\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)$'
        )
    
    def _initialize_data_save(self):
        """データ保存の初期化"""
        try:
            # 保存フォルダを作成
            os.makedirs(self.save_folder, exist_ok=True)
            
            # JST時刻でタイムスタンプ付きファイル名を生成
            timestamp = datetime.now(self.jst).strftime("%Y%m%d_%H%M%S")
            filename = f"gyro_stick_raw_data_{timestamp}_JST.csv"
            self.csv_filepath = os.path.join(self.save_folder, filename)
            
            # CSVヘッダーを定義（led_enabledを追加）
            self.csv_headers = [
                'timestamp_jst',
                'receive_time_jst',
                'raw_message',
                'button',
                'quaternion_w', 'quaternion_x', 'quaternion_y', 'quaternion_z',
                'acceleration_x', 'acceleration_y', 'acceleration_z',
                'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
                'client_address',
                'valid_data',
                'led_enabled'
            ]
            
            # CSVファイルを初期化（ヘッダー書き込み）
            with open(self.csv_filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.csv_headers)
            
            print(f"データ保存開始: {self.csv_filepath}")
            
        except Exception as e:
            print(f"データ保存初期化エラー: {e}")
            self.enable_data_save = False
    
    def _save_data(self, raw_message, parsed_data, client_addr, receive_time):
        """データを保存"""
        if not self.enable_data_save:
            return
        
        try:
            # JST時刻のタイムスタンプとして保存
            current_timestamp = datetime.now(self.jst).isoformat()
            receive_timestamp = datetime.fromtimestamp(receive_time, self.jst).isoformat()
            
            # データを抽出
            if parsed_data['valid']:
                button = parsed_data['button']
                quat = parsed_data['quaternion']
                accel = parsed_data['acceleration']
                gyro = parsed_data['gyroscope']
                valid_data = True
            else:
                # 無効なデータの場合は空値で埋める
                button = None
                quat = {'w': None, 'x': None, 'y': None, 'z': None}
                accel = {'x': None, 'y': None, 'z': None}
                gyro = {'x': None, 'y': None, 'z': None}
                valid_data = False
            
            # CSVの行データを作成（JST時刻で保存）
            row_data = [
                current_timestamp,      # JST時刻（ISO形式）
                receive_timestamp,      # 受信時刻もJST（ISO形式）
                raw_message,
                button,
                quat['w'], quat['x'], quat['y'], quat['z'],
                accel['x'], accel['y'], accel['z'],
                gyro['x'], gyro['y'], gyro['z'],
                str(client_addr),
                valid_data,
                self.led_enabled
            ]
            
            # ファイルに書き込み（スレッドセーフ）
            with self.file_lock:
                with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_data)
                    
        except Exception as e:
            print(f"データ保存エラー: {e}")
    
    def set_data_callback(self, callback: Callable):
        """データ受信時のコールバック関数を設定"""
        self.data_callback = callback
    
    def set_error_callback(self, callback: Callable):
        """エラー発生時のコールバック関数を設定"""
        self.error_callback = callback
    
    def validate_and_parse_data(self, message):
        """受信データの検証と解析"""
        message = message.strip()
        match = self.data_pattern.match(message)
        
        if match:
            try:
                return {
                    'valid': True,
                    'button': int(match.group(1)),
                    'quaternion': {
                        'w': float(match.group(2)),
                        'x': float(match.group(3)),
                        'y': float(match.group(4)),
                        'z': float(match.group(5))
                    },
                    'acceleration': {
                        'x': float(match.group(6)),
                        'y': float(match.group(7)),
                        'z': float(match.group(8))
                    },
                    'gyroscope': {
                        'x': float(match.group(9)),
                        'y': float(match.group(10)),
                        'z': float(match.group(11))
                    }
                }
            except ValueError:
                return {'valid': False, 'error': 'Failed to convert data to numbers'}
        else:
            return {'valid': False, 'error': 'Invalid data format'}
    
    def start_receiving(self):
        """データ受信を開始"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            print(f'Socket created and bound to {self.host}:{self.port}')
            
            # 受信スレッド開始
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            print("Data reception thread started")
            
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Failed to start receiving: {e}")
            else:
                print(f"Error: {e}")
    
    def _receive_loop(self):
        """データ受信ループ（内部メソッド）"""
        M_SIZE = 1024
        
        while self.loop_flag:
            try:
                message, cli_addr = self.sock.recvfrom(M_SIZE)
                receive_time = time.time()  # 受信時刻を記録
                
                if self.clients is None:
                    self.clients = cli_addr
                    print(f"Client connected: {cli_addr}")
                
                raw_message = message.decode(encoding='utf-8')
                parsed_data = self.validate_and_parse_data(raw_message)
                
                # データ保存（有効/無効データ問わず保存）
                if self.enable_data_save:
                    self._save_data(raw_message, parsed_data, cli_addr, receive_time)
                
                # コールバック処理
                if parsed_data['valid']:
                    if self.data_callback:
                        self.data_callback(parsed_data)
                else:
                    if self.error_callback:
                        self.error_callback(parsed_data['error'])
                        
            except KeyboardInterrupt:
                print('\nShutting down receiver...')
                self.stop_receiving()
            except Exception as e:
                if self.error_callback:
                    self.error_callback(f"Receive error: {e}")
    
    def send_command(self, command: str):
        """センサーにコマンドを送信"""
        if self.clients is not None and self.sock is not None:
            try:
                self.sock.sendto(command.encode(encoding='utf-8'), self.clients)
                # print(f"Command sent: {command}")
            except Exception as e:
                if self.error_callback:
                    self.error_callback(f"Send error: {e}")
        else:
            print("No client connected")
    
    def stop_receiving(self):
        """データ受信を停止"""
        self.loop_flag = False
        if self.sock:
            self.sock.close()
        
        if self.enable_data_save:
            print(f"データ保存終了: {self.csv_filepath}")
        
        print("Receiver stopped")
    
    def get_save_filepath(self):
        """保存ファイルパスを取得"""
        return self.csv_filepath if self.enable_data_save else None
    
    # 便利メソッド
    def start_sensor(self):
        """センサー開始コマンド"""
        self.send_command('SENSOR S')
    
    def stop_sensor(self):
        """センサー停止コマンド"""
        self.send_command('SENSOR E')
    
    def led_on(self, r=255, g=0, b=0):
        """LED点灯（フラグチェック付き）"""
        if self.led_enabled:
            self.send_command(f'LED 0,{r},{g},{b}')
    
    def led_off(self):
        """LED消灯（フラグチェック付き）"""
        if self.led_enabled:
            self.send_command('LED 0,0,0,0')
    
    def toggle_led_control(self):
        """LED制御のオン/オフを切り替え"""
        self.led_enabled = not self.led_enabled
        if not self.led_enabled:
            # LED制御をオフにした時は消灯
            self.send_command('LED 0,0,0,0')
        
        status = "有効" if self.led_enabled else "無効"
        print(f"LED制御: {status}")
        return self.led_enabled
    
    def set_led_control(self, enabled):
        """LED制御の有効/無効を設定"""
        self.led_enabled = enabled
        if not self.led_enabled:
            self.send_command('LED 0,0,0,0')
        
        status = "有効" if self.led_enabled else "無効"
        print(f"LED制御: {status}")