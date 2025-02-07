import pandas as pd
import numpy as np
def preprocessed(delete_threshould=100):

    data_train=pd.read_csv(r'safetyconstruction/train.csv')
    data_test=pd.read_csv(r'safetyconstruction/test.csv')
    # 删除数据行小于delete_threshould的环号
    data_train = data_train.groupby('RingNumber').filter(lambda x: len(x) >=delete_threshould)
    data_test = data_test.groupby('RingNumber').filter(lambda x: len(x) >=delete_threshould)
    #删除无用列(FeedingSlurryDensity	DischargingSlurryDensity)
    data_train.drop(['FeedingSlurryDensity', 'DischargingSlurryDensity'], axis=1, inplace=True)
    data_test.drop(['FeedingSlurryDensity', 'DischargingSlurryDensity'], axis=1, inplace=True)

    data_train['TunnelingState'].replace('FALSE',0,inplace=True)
    data_train['TunnelingState'].replace('TRUE',1,inplace=True)
    data_test['TunnelingState'].replace('FALSE',0,inplace=True)
    data_test['TunnelingState'].replace('TRUE',1,inplace=True)
    data_train['TunnelingState'].replace(False,0,inplace=True)
    data_train['TunnelingState'].replace(True,1,inplace=True)
    data_test['TunnelingState'].replace(False,0,inplace=True)
    data_test['TunnelingState'].replace(True,1,inplace=True)
    #删除标签列
    # data_train.drop(['Instability'], axis=1, inplace=True)
    # data_test.drop(['Instability'], axis=1, inplace=True)

    
    return data_train,data_test