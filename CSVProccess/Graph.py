import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import math

def parse_timestamp_jst(timestamp_jst_str):
    """Parse timestamp_jst string to datetime object"""
    try:
        # ISO 8601 format with timezone: 2025-06-25T13:23:45.532148+09:00
        return datetime.fromisoformat(timestamp_jst_str)
    except:
        try:
            # ISO 8601 format without timezone: 2025-06-25T13:23:45.532148
            return datetime.fromisoformat(timestamp_jst_str.replace('+09:00', ''))
        except:
            try:
                # Fallback to old format: 2025-06-25 13:23:45.532148
                return datetime.strptime(timestamp_jst_str, '%Y-%m-%d %H:%M:%S.%f')
            except:
                try:
                    # Fallback to old format without microseconds: 2025-06-25 13:23:45
                    return datetime.strptime(timestamp_jst_str, '%Y-%m-%d %H:%M:%S')
                except:
                    return None

def create_multi_column_graphs(input_file, columns_to_plot, output_dir="graphs"):
    """
    Create graphs for specified columns from extracted data
    
    Args:
        input_file (str): Path to extracted CSV file
        columns_to_plot (list): List of column names to plot
        output_dir (str): Directory to save graphs
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Check if specified columns exist
    existing_cols = [col for col in columns_to_plot if col in df.columns]
    
    if not existing_cols:
        print("Specified columns not found")
        return
    
    if 'timestamp_jst' not in df.columns:
        print("timestamp_jst column not found")
        return
    
    print(f"Plotting columns: {existing_cols}")
    
    # Split data by sections
    sections = []
    current_section = []
    section_num = 0
    
    for idx, row in df.iterrows():
        # Check if it's a section header
        if pd.isna(row['timestamp_jst']) or str(row['timestamp_jst']).startswith('=== '):
            if current_section:
                sections.append((section_num, current_section.copy()))
                current_section = []
                section_num += 1
        elif str(row['timestamp_jst']).strip() == '':
            # Skip empty rows
            continue
        else:
            # Add valid data row
            current_section.append(row)
    
    # Add last section
    if current_section:
        sections.append((section_num, current_section))
    
    print(f"Detected sections: {len(sections)}")
    
    # Create graphs for each section
    for section_idx, section_data in sections:
        if not section_data:
            continue
            
        # Convert section data to DataFrame
        section_df = pd.DataFrame(section_data)
        
        # Parse timestamp_jsts
        valid_data = []
        for _, row in section_df.iterrows():
            ts = parse_timestamp_jst(str(row['timestamp_jst']))
            if ts is not None:
                row_data = {'timestamp_jst': ts}
                for col in existing_cols:
                    try:
                        row_data[col] = float(row[col])
                    except:
                        row_data[col] = 0.0
                valid_data.append(row_data)
        
        if not valid_data:
            print(f"Section {section_idx + 1}: No valid data")
            continue
        
        # Convert valid data to DataFrame
        plot_df = pd.DataFrame(valid_data)
        plot_df = plot_df.sort_values('timestamp_jst')
        
        # Calculate time range
        start_time = plot_df['timestamp_jst'].iloc[0]
        end_time = plot_df['timestamp_jst'].iloc[-1]
        
        # Calculate elapsed seconds from start time
        plot_df['elapsed_seconds'] = (plot_df['timestamp_jst'] - start_time).dt.total_seconds()
        
        # Calculate subplot layout
        n_plots = len(existing_cols)
        n_cols = min(3, n_plots)  # Max 3 columns
        n_rows = math.ceil(n_plots / n_cols)
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each column
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, col in enumerate(existing_cols):
            ax = axes[i]
            
            # Plot data
            ax.plot(plot_df['elapsed_seconds'], plot_df[col], 
                   color=colors[i % len(colors)], linewidth=1.5)
            
            # Add threshold lines if it's gyroscope data
            if 'gyro' in col.lower():
                ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Threshold +100')
                ax.axhline(y=-100, color='orange', linestyle='--', alpha=0.7, label='Threshold -100')
                ax.legend()
            
            # Set labels and title
            ax.set_xlabel('Elapsed Time (seconds)')
            ax.set_ylabel(col)
            ax.set_title(f'{col}')
            ax.grid(True, alpha=0.3)
            
            # Adjust Y-axis range
            y_data = plot_df[col]
            y_min = min(y_data.min(), -150 if 'gyro' in col.lower() else y_data.min() * 1.1)
            y_max = max(y_data.max(), 150 if 'gyro' in col.lower() else y_data.max() * 1.1)
            ax.set_ylim(y_min, y_max)
        
        # Hide unused subplots
        for i in range(len(existing_cols), len(axes)):
            axes[i].set_visible(False)
        
        # Set overall title
        start_str = start_time.strftime('%H:%M:%S')
        end_str = end_time.strftime('%H:%M:%S')
        fig.suptitle(f'{start_str} ~ {end_str}', fontsize=16, fontweight='bold')
        
        # Save graph
        filename = f"section_{section_idx + 1:02d}_{start_str.replace(':', '')}-{end_str.replace(':', '')}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph saved: {filepath}")
        print(f"  Time range: {start_str} ~ {end_str}")
        print(f"  Data points: {len(plot_df)}")
        print(f"  Columns: {existing_cols}")
        print()

# Usage example
if __name__ == "__main__":
    input_csv = "high_gyro_eztracted_hatannosan.csv"
    
    # Specify columns to plot
    # Example 1: Only gyroscope data
    #gyro_columns = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']
    
    # Example 2: Gyroscope + Accelerometer data
    all_columns = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z', 
                   'acceleration_x', 'acceleration_y', 'acceleration_z']
    
    # Example 3: Custom selection
    # custom_columns = ['gyroscope_x', 'accelerometer_z', 'magnetometer_x']
    
    # Execute graph creation
    create_multi_column_graphs(input_csv, all_columns)