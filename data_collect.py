"""
センサーは100Hzが限界、100Hzだと結構値がガタついているから処理が必要そう
"""

import socket
import threading
import time
import numpy as np
import csv
import json
import os
from collections import deque
import re
from datetime import datetime
from sensor_math import SensorMath

LoopFlag = True
clients = None

M_SIZE = 1024
host = '192.168.1.164'
port = 12351
localaddr = (host, port)
sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
sock.bind(localaddr)

# Recording state
is_recording = False
current_session_data = []
session_counter = 0
current_label = None
motion_labels = {}  # セッション番号と動作ラベルの対応

# Thread synchronization
data_lock = threading.Lock()
label_lock = threading.Lock()

# Velocity tracking for integration
velocity = [0.0, 0.0, 0.0]
last_timestamp = None

# Create timestamp-based directory for this run
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_dir = f"data/{run_timestamp}"
os.makedirs(data_dir, exist_ok=True)

# Updated data validation regex pattern for 11 fields
data_pattern = re.compile(r'^(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)$')

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
            accel_x = float(match.group(6))
            accel_y = float(match.group(7))
            accel_z = float(match.group(8))
            gyro_x = float(match.group(9))
            gyro_y = float(match.group(10))
            gyro_z = float(match.group(11))
            
            return {
                'valid': True,
                'push_button': push_button,
                'quaternion': {'w': quat_w, 'x': quat_x, 'y': quat_y, 'z': quat_z},
                'acceleration': {'x': accel_x, 'y': accel_y, 'z': accel_z},
                'gyroscope': {'x': gyro_x, 'y': gyro_y, 'z': gyro_z}
            }
        except ValueError:
            return {'valid': False, 'error': 'Failed to convert data to numbers'}
    else:
        return {'valid': False, 'error': 'Invalid data format'}

def transform_sensor_data(parsed_data, current_time):
    """
    センサーデータを座標変換
    """
    global velocity, last_timestamp
    
    # ローカル座標系のデータ
    local_accel = [
        parsed_data['acceleration']['x'],
        parsed_data['acceleration']['y'],
        parsed_data['acceleration']['z']
    ]
    local_gyro = [
        parsed_data['gyroscope']['x'],
        parsed_data['gyroscope']['y'],
        parsed_data['gyroscope']['z']
    ]
    quaternion = parsed_data['quaternion']
    
    # 座標変換
    global_accel = SensorMath.transform_to_global(local_accel, quaternion)
    euler_rates = SensorMath.body_angular_velocity_to_euler_rates(local_gyro, quaternion)
    
    # 速度計算
    if last_timestamp is not None:
        dt = current_time - last_timestamp
        if dt > 0:
            velocity[0] = SensorMath.lowcut_integration(velocity[0], global_accel[0], dt, q=0.95)
            velocity[1] = SensorMath.lowcut_integration(velocity[1], global_accel[1], dt, q=0.95)
            velocity[2] = SensorMath.lowcut_integration(velocity[2], global_accel[2], dt, q=0.95)
    
    last_timestamp = current_time
    
    return {
        'local_accel': local_accel,
        'global_accel': global_accel,
        'local_gyro': local_gyro,
        'euler_rates': euler_rates,
        'velocity': velocity.copy(),
        'quaternion': quaternion
    }

def save_session_to_csv(session_data, session_id):
    """
    Save recorded session data to CSV file
    """
    if not session_data:
        return
    
    filename = f"{data_dir}/{session_id}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'button',
            'quat_w', 'quat_x', 'quat_y', 'quat_z',
            'local_accel_x', 'local_accel_y', 'local_accel_z',
            'global_accel_x', 'global_accel_y', 'global_accel_z',
            'local_gyro_x', 'local_gyro_y', 'local_gyro_z',
            'euler_rate_roll', 'euler_rate_pitch', 'euler_rate_yaw',
            'velocity_x', 'velocity_y', 'velocity_z'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in session_data:
            writer.writerow(data)

def save_labels_to_json():
    """
    Save motion labels to JSON file
    """
    labels_file = f"{data_dir}/labels.json"
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(motion_labels, f, ensure_ascii=False, indent=2)

def input_thread():
    """
    Handle user input for motion labels
    """
    global current_label, motion_labels, LoopFlag
    
    print("=== Motion Data Collector ===")
    print(f"Data will be saved to: {data_dir}")
    print("Usage:")
    print("1. Enter motion label")
    print("2. Press device button to start/stop recording")
    print("3. Type 'quit' to exit")
    print()
    
    while LoopFlag:
        try:
            user_input = input("Enter motion label (or 'quit' to exit): ").strip()
            
            if user_input.lower() == 'quit':
                LoopFlag = False
                break
            
            if user_input:
                with label_lock:
                    current_label = user_input
                    print(f"Label set: '{current_label}' - Ready to record (press device button)")
            else:
                print("Please enter a valid label")
                
        except EOFError:
            LoopFlag = False
            break
        except KeyboardInterrupt:
            LoopFlag = False
            break

def mainloop():
    global LoopFlag, clients, sock
    global is_recording, current_session_data, session_counter
    global velocity, last_timestamp, current_label, motion_labels

    while LoopFlag:
        try:
            message, cli_addr = sock.recvfrom(M_SIZE)
            if clients is None:
                clients = cli_addr
            message = message.decode(encoding='utf-8')
            
            current_time = time.time()
            
            # Validate and parse data
            parsed_data = validate_and_parse_data(message)
            
            if parsed_data['valid']:
                button_pressed = parsed_data['push_button'] == 1
                
                with data_lock:
                    # Start recording when button is pressed and label is set
                    if button_pressed and not is_recording and current_label is not None:
                        is_recording = True
                        current_session_data = []
                        session_counter += 1
                        # Reset velocity and timestamp for new session
                        velocity = [0.0, 0.0, 0.0]
                        last_timestamp = None
                        
                        with label_lock:
                            motion_labels[str(session_counter)] = current_label
                            print(f"Recording started - Session {session_counter}: '{current_label}'")
                    
                    # Stop recording when button is released
                    elif not button_pressed and is_recording:
                        is_recording = False
                        save_session_to_csv(current_session_data, session_counter)
                        save_labels_to_json()
                        print(f"Recording stopped - Session {session_counter} saved ({len(current_session_data)} data points)")
                        
                        with label_lock:
                            current_label = None
                    
                    # Save data if recording (only when button is pressed)
                    if is_recording and button_pressed:
                        # Transform sensor data
                        transformed_data = transform_sensor_data(parsed_data, current_time)
                        
                        data_row = {
                            'timestamp': current_time,
                            'button': parsed_data['push_button'],
                            'quat_w': parsed_data['quaternion']['w'],
                            'quat_x': parsed_data['quaternion']['x'],
                            'quat_y': parsed_data['quaternion']['y'],
                            'quat_z': parsed_data['quaternion']['z'],
                            'local_accel_x': transformed_data['local_accel'][0],
                            'local_accel_y': transformed_data['local_accel'][1],
                            'local_accel_z': transformed_data['local_accel'][2],
                            'global_accel_x': transformed_data['global_accel'][0],
                            'global_accel_y': transformed_data['global_accel'][1],
                            'global_accel_z': transformed_data['global_accel'][2],
                            'local_gyro_x': transformed_data['local_gyro'][0],
                            'local_gyro_y': transformed_data['local_gyro'][1],
                            'local_gyro_z': transformed_data['local_gyro'][2],
                            'euler_rate_roll': transformed_data['euler_rates'][0],
                            'euler_rate_pitch': transformed_data['euler_rates'][1],
                            'euler_rate_yaw': transformed_data['euler_rates'][2],
                            'velocity_x': transformed_data['velocity'][0],
                            'velocity_y': transformed_data['velocity'][1],
                            'velocity_z': transformed_data['velocity'][2]
                        }
                        current_session_data.append(data_row)
                
        except Exception as e:
            if LoopFlag:  # Only print errors if we're still running
                print(f"Error: {e}")

# Start input thread
input_thread_obj = threading.Thread(target=input_thread, daemon=True)
input_thread_obj.start()

# Start main loop
try:
    mainloop()
except KeyboardInterrupt:
    print("\nProgram terminated by user")
finally:
    LoopFlag = False
    
    # Save any ongoing recording
    if is_recording and current_session_data:
        save_session_to_csv(current_session_data, session_counter)
        save_labels_to_json()
        print("Saved ongoing recording before shutdown")
    
    if 'sock' in locals():
        sock.close()
    print("Cleanup completed")