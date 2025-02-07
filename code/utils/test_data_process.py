import numpy as np
import pandas as pd
def test_data_process(data_test):
    #缺失值上下均值替代
    df=data_test.copy()
    df_filled = df.ffill().bfill()
    return df_filled