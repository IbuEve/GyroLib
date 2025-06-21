import socket
import threading
import time
import re
from typing import Callable, Optional

class GyroStickReceiver:
    """ジャイロスティックデータの受信クラス"""
    
    def __init__(self, host='192.168.1.164', port=12351):
        self.host = host
        self.port = port
        self.sock = None
        self.clients = None
        self.loop_flag = True
        self.data_callback = None
        self.error_callback = None
        
        # データ検証パターン
        self.data_pattern = re.compile(
            r'^(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)'
            r'\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)'
            r'\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)$'
        )
    
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
                if self.clients is None:
                    self.clients = cli_addr
                    print(f"Client connected: {cli_addr}")
                
                message = message.decode(encoding='utf-8')
                parsed_data = self.validate_and_parse_data(message)
                
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
        print("Receiver stopped")
    
    # 便利メソッド
    def start_sensor(self):
        """センサー開始コマンド"""
        self.send_command('SENSOR S')
    
    def stop_sensor(self):
        """センサー停止コマンド"""
        self.send_command('SENSOR E')
    
    def led_on(self, r=255, g=0, b=0):
        """LED点灯"""
        self.send_command(f'LED 0,{r},{g},{b}')
    
    def led_off(self):
        """LED消灯"""
        self.send_command('LED 0,0,0,0')