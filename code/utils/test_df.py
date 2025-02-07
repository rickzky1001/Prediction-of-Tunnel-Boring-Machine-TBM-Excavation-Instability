import pandas as pd
import numpy as np
def test_df(raw_data_test,model_pred,threshold):
    count_dict={}
    deleted_rn=[]
    for rn in raw_data_test['RingNumber'].unique():
        count_dict[rn]=len(raw_data_test[raw_data_test['RingNumber']==rn])
    for rn,count in count_dict.items():
        if count<threshold:
            deleted_rn.append(rn)
    test_result=pd.DataFrame()
    test_result['RingNmuber']=raw_data_test['RingNumber'].unique()
    test_result['Instability']=pd.NA
    mask=test_result['RingNmuber'].isin(deleted_rn)
    
    test_result.loc[~mask,'Instability']=model_pred
    test_result.loc[mask,'Instability']=0
    return test_result