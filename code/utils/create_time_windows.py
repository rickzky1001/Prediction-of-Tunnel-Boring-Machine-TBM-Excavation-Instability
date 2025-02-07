import numpy as np
import pandas as pd
import sys
def create_time_windows(df, time_step=10, overlapping=False, overlap_ratio=0.2,n=25):
    # 确保RingNumber列存在
    if 'RingNumber' not in df.columns:
        raise ValueError("DataFrame必须包含'RingNumber'列")
    
    # 初始化一个空列表来存储每个RingNumber的结果
    result = []
    
    # 对每个RingNumber进行处理
    for ring_number, group in df.groupby('RingNumber'):
        # 获取当前RingNumber的数据
        data = group.drop(columns=['RingNumber']).values
        total_samples = data.shape[0]
        # 根据方法选择不同的窗口划分方式
        if not overlapping:
            # 方法1：窗口间不重叠，紧挨着
            # 窗口宽度
            width = data.shape[0] // time_step
            windowed_data = data[:width * time_step]
            windowed_data = windowed_data.reshape(time_step,width, data.shape[1])
            result.append(windowed_data)
            if ring_number in [534, 535, 536, 537]:
                max_width = total_samples // time_step
                for i in range(n):
                    strengthed_width = max(10, max_width - 4 * i)  # 避免 width 过小
                    if strengthed_width * time_step > total_samples:
                        continue  # 避免超出数据范围
                    strengthed_windowed_data = data[:strengthed_width * time_step]
                    strengthed_windowed_data = strengthed_windowed_data.reshape(time_step,strengthed_width, data.shape[1])
                    result.append(strengthed_windowed_data)
            
        elif overlapping: 
            # **方法2：窗口重叠**
            adjusted_samples = total_samples - (total_samples % (time_step))  # 确保能被 10 整除
            
            if adjusted_samples < time_step :
                print(f"RingNumber {ring_number}: 数据不足，跳过")
                continue  # 无法划分 10 个窗口的情况跳过

            adjusted_data = data[:adjusted_samples]

            window_size = adjusted_samples // time_step  # 确保划分为10个窗口
            overlap_size = int(window_size * overlap_ratio)
            stride = window_size - overlap_size
            windows = []
            start_index = 0
            
            for _ in range(time_step):  # **确保有 10 个窗口**
                end_index = start_index + window_size
                windows.append(adjusted_data[start_index:end_index])
                start_index += stride  # 滑动窗口
            # **数据增强**
            if ring_number in [534, 535, 536, 537]:
                for i in range(n):
                    new_window_size = max(10, window_size - 4 * i)  # 确保窗口不变得过小
                    new_stride = new_window_size - int(new_window_size * overlap_ratio)
                    
                    start_index = 0
                    enhanced_windows = []
                    for _ in range(time_step):  
                        end_index = start_index + new_window_size
                        if end_index > adjusted_samples:
                            break  # 避免超出范围
                        enhanced_windows.append(adjusted_data[start_index:end_index])
                        start_index += new_stride
                    
                    if len(enhanced_windows) == time_step:
                        result.append(np.array(enhanced_windows))
            
            result.append(np.array(windows))

    return result