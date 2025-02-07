#思路,前top个，找到突变索引tp_idx，0:tp_idx全部变成tp_idx+1的值
import numpy as np
import pandas as pd
def stroke_process(data,top=20,k=15):
    for rn in data['RingNumber'].unique():
        mask=(data['RingNumber'] == rn)
        s0=data[mask].iloc[:top]['Stroke'].values
        s1=np.concatenate((
            data[mask].iloc[:top]['Stroke'].values[1:],\
            np.array(data[mask].iloc[:top]['Stroke'].values[-1]).reshape(1)
                        ))
        # 找到第一个 True 的索引
        if (np.abs(s1-s0)>k).any():
            tp_idx=np.argmax(np.abs(s1-s0)>k)
            value=s0[tp_idx+1]
            mask = mask & (data.index <= data[mask].index.values[tp_idx])
            data.loc[mask, 'Stroke'] = value
        else:
            continue
        return data