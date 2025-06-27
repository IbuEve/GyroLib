import csv
import pandas as pd
from datetime import datetime

def unix_to_jst(unix_timestamp):
    """UnixタイムスタンプをJST日時文字列に変換"""
    try:
        # Unixタイムスタンプをdatetimeオブジェクトに変換（UTC）
        dt_utc = datetime.utcfromtimestamp(float(unix_timestamp))
        # JSTに変換（UTC+9時間）
        dt_jst = dt_utc.replace(tzinfo=None)
        # 9時間を手動で追加
        import datetime as dt_module
        dt_jst = dt_jst + dt_module.timedelta(hours=9)
        return dt_jst.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # ミリ秒まで表示
    except:
        return str(unix_timestamp)

def extract_high_gyro_data(input_file, output_file, threshold=100, context_size=100):
    """
    ジャイロスコープの値が閾値以上のデータとその前後のデータを抽出
    
    Args:
        input_file (str): 入力CSVファイルのパス
        output_file (str): 出力CSVファイルのパス
        threshold (float): 閾値（デフォルト：100）
        context_size (int): 前後に取得するデータ数（デフォルト：100）
    """
    
    # CSVファイルを読み込み
    df = pd.read_csv(input_file)
    
    # ジャイロスコープの列を確認
    gyro_cols = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    # 存在する列のみを使用
    existing_gyro_cols = [col for col in gyro_cols if col in df.columns]
    
    if not existing_gyro_cols:
        print("ジャイロスコープの列が見つかりません")
        return
    
    print(f"対象列: {existing_gyro_cols}")
    
    # 閾値以上の値を持つ行のインデックスを取得
    condition = (df[existing_gyro_cols].abs() >= threshold).any(axis=1)
    trigger_indices = df[condition].index.tolist()
    
    if not trigger_indices:
        print(f"閾値{threshold}以上の値が見つかりませんでした")
        return
    
    print(f"閾値以上の値が見つかった行数: {len(trigger_indices)}")
    
    # 抽出範囲を計算（重複を考慮して結合）
    ranges = []
    
    for idx in trigger_indices:
        start = max(0, idx - context_size)
        end = min(len(df), idx + context_size + 1)
        ranges.append((start, end))
    
    # 重複する範囲を結合
    merged_ranges = []
    if ranges:
        ranges.sort()
        current_start, current_end = ranges[0]
        
        for start, end in ranges[1:]:
            if start <= current_end:  # 重複または隣接
                current_end = max(current_end, end)
            else:
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_ranges.append((current_start, current_end))
    
    print(f"結合後の範囲数: {len(merged_ranges)}")
    
    # 出力用のデータを準備
    output_data = []
    
    for i, (start, end) in enumerate(merged_ranges):
        print(f"範囲 {i+1}: {start} - {end-1} 行目 ({end-start}行)")
        
        # セクションヘッダーを追加
        header_row = ['=== セクション {} ==='.format(i+1)] + [''] * (len(df.columns) - 1)
        output_data.append(header_row)
        
        # 範囲内の行を追加
        section_df = df.iloc[start:end].copy()
        
        # タイムスタンプ列があれば変換
        if 'timestamp' in section_df.columns:
            section_df['timestamp'] = section_df['timestamp'].apply(unix_to_jst)
        
        # データを追加
        for _, row in section_df.iterrows():
            output_data.append(row.tolist())
        
        # セクション間に空行を追加
        if i < len(merged_ranges) - 1:
            empty_row = [''] * len(df.columns)
            output_data.append(empty_row)
    
    # CSVファイルに書き出し
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # ヘッダーを書き込み
        writer.writerow(df.columns.tolist())
        
        # データを書き込み
        writer.writerows(output_data)
    
    print(f"抽出完了: {output_file}")
    print(f"総データ行数: {len(output_data)}")

# 使用例
if __name__ == "__main__":
    input_csv = "SaveData\gyro_stick_raw_data_20250625_132345_JST-hatannosan.csv"
    output_csv = "SaveData\\high_gyro_eztracted_hatannosan.csv"
    
    # 抽出実行
    extract_high_gyro_data(input_csv, output_csv, threshold=100, context_size=100)