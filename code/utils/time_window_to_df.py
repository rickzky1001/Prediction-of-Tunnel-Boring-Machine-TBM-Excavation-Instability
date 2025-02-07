import numpy as np
import pandas as pd
def time_window_to_df(time_windows_list,time_step=10):
    RingNumber_index_list=[window.shape[1]*time_step for window in time_windows_list]
    for i in range(len(RingNumber_index_list)):
        if i ==0:
            continue
        RingNumber_index_list[i]=RingNumber_index_list[i]+RingNumber_index_list[i-1]#- RingNumber_index_list[0]
    RingNumber_index_list.insert(0, 0)
    RingNumber_index_list.pop()
    reshaped_data = [window.reshape(-1, window.shape[2]) for window in time_windows_list]
    stacked_data = np.vstack(reshaped_data)
    df = pd.DataFrame(stacked_data)
    return df,RingNumber_index_list