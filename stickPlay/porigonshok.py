import socket
import threading
import time
import numpy as np
from scipy import signal
from collections import deque
import re
from control import tf,c2d
from control.matlab import lsim, bode, tfdata

LoopFlag = True
clients = None

M_SIZE = 1024
# host = '192.168.1.164'
host = '172.20.10.4'
port = 12351
localaddr = (host, port)
sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
print('create socket')
sock.bind(localaddr)
print('Waiting message')
dot = 0

t = np.linspace(0, 4*np.pi, 10)
square_wave = np.sign(np.sin(t))

s = np.linspace(0, 4*np.pi, 400)
sin_wave = np.sin(s)  # -1～1のsin波
sin_wave_unipolar = (sin_wave + 1) / 2


# http://qiita.com/Y-Yoshimura1997/items/af81e1fa4aa2cbc17a01


def generate_pink_noise(Ts):
    # フィルタ作成
    P = [tf([1],[1])]
    for k in range(-4, 2):
        tmp = tf([4**k, 1], [2*4**k, 1])
        P.append(tmp) 
        P[0] = P[0]*tmp
    t = np.arange(1, 20, Ts)
    size = len(t)
    u = np.random.rand(len(t))
    u = u - np.ones(size)*0.5
    y, t_o, x_0 = lsim(P[0], u, t)
    return y


Ts = 0.005
pink_noise = generate_pink_noise(Ts)
pink_noise_normalized = (pink_noise - np.min(pink_noise)) / (np.max(pink_noise) - np.min(pink_noise))

# Data validation regex pattern - 8 elements for quaternion
data_pattern = re.compile(r'^(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)$')

def validate_and_parse_data(message):
   """
   Validate received data format and parse if valid
   """
   message = message.strip()
   match = data_pattern.match(message)
   
   if match:
       try:
           push_button = int(match.group(1))
           quat_w = float(match.group(2))
           quat_x = float(match.group(3))
           quat_y = float(match.group(4))
           quat_z = float(match.group(5))
           acceleration_x = float(match.group(6))
           acceleration_y = float(match.group(7))
           acceleration_z = float(match.group(8))
           
           return {
               'valid': True,
               'push_button': push_button,
               'quaternion': {'w': quat_w, 'x': quat_x, 'y': quat_y, 'z': quat_z},
               'acceleration': {'x': acceleration_x, 'y': acceleration_y, 'z': acceleration_z}
           }
       except ValueError:
           return {'valid': False, 'error': 'Failed to convert data to numbers'}
   else:
       return {'valid': False, 'error': 'Invalid data format (expected: pushButton quat.w quat.x quat.y quat.z acceleration.x acceleration.y acceleration.z)'}

def quaternion_to_rotation_matrix(w, x, y, z):
    """Quaternionから回転行列に変換"""
    # Quaternionを正規化
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 正しい回転行列の公式
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def mainloop():
    global LoopFlag
    global clients
    global sock
    global dot

    while LoopFlag:
        try:
            message, cli_addr = sock.recvfrom(M_SIZE)
            if clients is None:
                clients = cli_addr
            message = message.decode(encoding='utf-8')
           
            # Validate and parse data
            parsed_data = validate_and_parse_data(message)
            rotation_matrix = quaternion_to_rotation_matrix(
                parsed_data['quaternion']['w'], 
                parsed_data['quaternion']['x'], 
                parsed_data['quaternion']['y'], 
                parsed_data['quaternion']['z']
            )

            rotated_x = rotation_matrix[:, 0] 
            axis_z = np.array([0,0,1])
            dot = np.dot(rotated_x, axis_z)
        except KeyboardInterrupt:
            print('\n ... \n')
            sock.close()
            LoopFlag = False

def keyloop():
    global LoopFlag
    global clients
    global sock
    global dot
    
    i=0
    j=0
    while LoopFlag:
        if clients is not None:  # クライアントが接続されている場合のみ送信
            try:
                if dot < 0.3: 
                    if i < len(square_wave) - 1:
                        i += 1
                    else: 
                        i = 0                   
                    value = int(255 * square_wave[i])
                    if square_wave[i] > 0:
                        message = f'LED 0,{value},0,0'
                    else:
                        message = f'LED 0,0,0,{value*-1}'
                else:
                    # # 1/fゆらぎ（ピンクノイズ）
                    # if j < len(pink_noise_normalized) - 1:
                    #     j += 1
                    # else: 
                    #     j = 0        
                    # value = int(255 * pink_noise_normalized[j])
                    # message = f'LED 0,0,{value},0'  # 緑色で表示

                    if j < len(sin_wave_unipolar) - 1:
                        j += 1
                    else: 
                        j = 0        
                    value = int(255 * sin_wave_unipolar[j])
                    message = f'LED 0,0,{value},0'  # 緑色で表示

                sock.sendto(message.encode('utf-8'), clients)
                time.sleep(0.01)  # 送信間隔
                
            except Exception as e:
                print(f"Error sending data: {e}")
                clients = None  # エラー時は接続をリセット
        else:
            time.sleep(0.1)  # クライアント接続待ち
        
thread1 = threading.Thread(target=mainloop)
thread2 = threading.Thread(target=keyloop)

thread1.start()
print("main thread started")
thread2.start()
print("key thread started")

thread1.join()
thread2.join()

print("program terminated")
