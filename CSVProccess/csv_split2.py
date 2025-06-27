import pandas as pd
import os
import re

def split_extracted_csv_by_sections(input_file, output_dir="sections-hatano"):
    """
    抽出済みCSVファイルをセクションごとに分割
    
    Args:
        input_file (str): 抽出済みCSVファイルのパス
        output_dir (str): 分割ファイル保存ディレクトリ
    """
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # CSVファイルを読み込み
    df = pd.read_csv(input_file)
    
    # セクションごとにデータを分割
    sections = []
    current_section = []
    section_num = 0
    section_start_time = None
    section_end_time = None
    
    print(f"Processing {input_file}...")
    
    for idx, row in df.iterrows():
        # セクションヘッダーかチェック
        timestamp_str = str(row.iloc[0])  # 最初の列（通常timestamp）
        
        if pd.isna(row.iloc[0]) or timestamp_str.startswith('=== セクション') or timestamp_str.startswith('=== Section'):
            # 前のセクションを保存
            if current_section:
                sections.append({
                    'number': section_num,
                    'data': current_section.copy(),
                    'start_time': section_start_time,
                    'end_time': section_end_time
                })
                current_section = []
                section_num += 1
                section_start_time = None
                section_end_time = None
        elif timestamp_str.strip() == '':
            # 空行をスキップ
            continue
        else:
            # 有効なデータ行を追加
            current_section.append(row)
            
            # 時刻情報を取得（timestampカラムがある場合）
            if 'timestamp_jst' in df.columns and not pd.isna(row['timestamp_jst']):
                try:
                    timestamp_value = float(row['timestamp_jst'])
                    if section_start_time is None:
                        section_start_time = timestamp_value
                    section_end_time = timestamp_value
                except (ValueError, TypeError):
                    # タイムスタンプが数値でない場合は無視
                    pass
    
    # 最後のセクションを追加
    if current_section:
        sections.append({
            'number': section_num,
            'data': current_section,
            'start_time': section_start_time,
            'end_time': section_end_time
        })
    
    print(f"Found {len(sections)} sections")
    
    # 各セクションをCSVファイルとして保存
    saved_files = []
    
    for section in sections:
        if not section['data']:
            continue
        
        # セクションデータをDataFrameに変換
        section_df = pd.DataFrame(section['data'])
        section_df = section_df.reset_index(drop=True)
        
        # ファイル名を生成
        if section['start_time'] is not None and section['end_time'] is not None:
            # 時刻情報がある場合
            start_str = unix_to_jst_time_string(section['start_time'])
            end_str = unix_to_jst_time_string(section['end_time'])
            filename = f"section_{section['number']+1:02d}_{start_str}-{end_str}.csv"
            
            # 継続時間を計算
            try:
                duration = float(section['end_time']) - float(section['start_time'])
            except (ValueError, TypeError):
                duration = 0.0
        else:
            # 時刻情報がない場合
            filename = f"section_{section['number']+1:02d}_rows_{len(section_df)}.csv"
            duration = 0.0
        
        # 不正な文字を除去
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filepath = os.path.join(output_dir, filename)
        
        # CSVファイルに保存
        section_df.to_csv(filepath, index=False)
        saved_files.append(filepath)
        
        print(f"Saved: {filename}")
        print(f"  Rows: {len(section_df)}")
        if section['start_time'] is not None and section['end_time'] is not None:
            start_jst = unix_to_jst_display_string(section['start_time'])
            end_jst = unix_to_jst_display_string(section['end_time'])
            print(f"  Time: {start_jst} ~ {end_jst} ({duration:.1f}s)")
        else:
            print("  Time: No timestamp data")
        print()
    
    print(f"分割完了: {len(saved_files)} files saved to {output_dir}")
    return saved_files

def unix_to_jst_time_string(unix_timestamp):
    """UnixタイムスタンプをJST時刻文字列に変換（ファイル名用）"""
    try:
        from datetime import datetime
        import datetime as dt_module
        
        dt_utc = datetime.utcfromtimestamp(float(unix_timestamp))
        dt_jst = dt_utc + dt_module.timedelta(hours=9)
        return dt_jst.strftime('%H%M%S')  # コロンを除去してファイル名に適した形式
    except:
        return "000000"

def unix_to_jst_display_string(unix_timestamp):
    """UnixタイムスタンプをJST時刻文字列に変換（表示用）"""
    try:
        from datetime import datetime
        import datetime as dt_module
        
        dt_utc = datetime.utcfromtimestamp(float(unix_timestamp))
        dt_jst = dt_utc + dt_module.timedelta(hours=9)
        return dt_jst.strftime('%H:%M:%S')  # 表示用はコロンあり
    except:
        return "00:00:00"

def batch_split_extracted_csvs(input_pattern="*extracted*.csv", output_base_dir="sections"):
    """
    抽出済みCSVファイルを一括で分割
    
    Args:
        input_pattern (str): 入力ファイルのパターン
        output_base_dir (str): 出力ベースディレクトリ
    """
    import glob
    
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    for csv_file in csv_files:
        print(f"\n=== Processing {csv_file} ===")
        
        # ファイル名からディレクトリ名を生成
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir = os.path.join(output_base_dir, base_name)
        
        try:
            split_extracted_csv_by_sections(csv_file, output_dir)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

# 使用例
if __name__ == "__main__":
    # 単一ファイルを分割
    input_csv = "high_gyro_eztracted_hatannosan.csv"
    
    if os.path.exists(input_csv):
        split_extracted_csv_by_sections(input_csv)
    else:
        print(f"File not found: {input_csv}")
        
        # 代替案：SaveDataフォルダ内の抽出済みCSVファイルを一括分割
        print("Searching for extracted CSV files...")
        batch_split_extracted_csvs()