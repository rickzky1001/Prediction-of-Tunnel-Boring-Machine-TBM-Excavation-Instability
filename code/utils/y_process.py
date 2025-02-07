import pandas as pd
import numpy as np
def y_process(delete_threshould=100):
    data_train=pd.read_csv(r'safetyconstruction/train.csv')
    data_train = data_train.groupby('RingNumber').filter(lambda x: len(x) >= delete_threshould)
    y_train=data_train.groupby('RingNumber')['Instability'].agg(lambda x: x.mode().iloc[0])
    return y_train.values
if __name__=='__main__':
    
    y_train=y_process()
    print(y_train)
    print(len(y_train))
    print(y_train.sum()/len(y_train))