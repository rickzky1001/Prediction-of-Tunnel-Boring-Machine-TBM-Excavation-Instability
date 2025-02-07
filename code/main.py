from utils.create_time_windows import create_time_windows
from utils.data_preprocessing import preprocessed
from utils.stroke_process import stroke_process
from utils.test_data_process import test_data_process
from utils.feature_engineering import featrue_engineering
from utils.time_window_to_df import time_window_to_df
from utils.z_score_standardization import z_score_norm
from utils.statistics import statistics_comp
from utils.delete_labels import delete_label
from utils.feature_selection import feature_importance_select
from utils.cor import add_correlation_features
from utils.y_process import y_process
import pandas as pd
import numpy as np
import sys
if __name__=='__main__':
    data_train,data_test=preprocessed(delete_threshould=100)
    data_test.drop(columns=['Instability'],inplace=True)
    data_train=stroke_process(data_train,top=20,k=15)
    data_test=stroke_process(data_test,top=20,k=15)
    data_test=test_data_process(data_test)
    #y=y_process(delete_threshould=100)
    data_train,data_test=featrue_engineering(data_train,data_test)
    data_train=z_score_norm(data_train, exclude_columns=['Instability','RingNumber','TunnelingState'])
    data_test=z_score_norm(data_test, exclude_columns=['Instability','RingNumber','TunnelingState'])
    bags_list=create_time_windows(data_train, time_step=15, overlapping=True, overlap_ratio=0.2,n=25)
    statistics_result = statistics_comp(bags_list=bags_list)
    statistics_result = delete_label(statistics_result)
    X=statistics_result[:,:,:-1]
    y=statistics_result[:,:,-1]
    #X=add_correlation_features(statistics_result,bags_list)
    selected_importance, selected_index=feature_importance_select(X,y)
    #X=statistics_result
    X=statistics_result[:,:,selected_index]
    print(selected_index)
    print(X.shape)