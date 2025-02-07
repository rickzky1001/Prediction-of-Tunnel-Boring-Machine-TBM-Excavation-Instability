import pandas as pd
import numpy as np

def featrue_engineering(data_train,data_test):
    #新特征：三种increase ,MudBalance, AirChamberPressureBalance1, AirChamberPressureBalance2
    data_train['Torque_increase'] = data_train.groupby('RingNumber')['CutterTorque'].diff().fillna(0)
    data_train['Thrust_increase'] = data_train.groupby('RingNumber')['Thrust'].diff().fillna(0)
    data_train['Speed_increase'] = data_train.groupby('RingNumber')['Speed'].diff().fillna(0)

    data_test['Torque_increase'] = data_test.groupby('RingNumber')['CutterTorque'].diff().fillna(0)
    data_test['Thrust_increase'] = data_test.groupby('RingNumber')['Thrust'].diff().fillna(0)
    data_test['Speed_increase'] = data_test.groupby('RingNumber')['Speed'].diff().fillna(0)
    # 检查列名
    for col in ['Torque_increase', 'Thrust_increase', 'Speed_increase']:
        data_train[col] = data_train[col].replace([np.inf, -np.inf], np.nan)
    # 检查列名
    for col in ['Torque_increase', 'Thrust_increase', 'Speed_increase']:
        data_test[col] = data_test[col].replace([np.inf, -np.inf], np.nan)
    
    data_train['MudBalance'] = data_train['FeedingSlurryFlowRate'] - data_train['DischargingSlurryFlowRate']
    data_test['MudBalance'] = data_test['FeedingSlurryFlowRate'] - data_test['DischargingSlurryFlowRate']
    data_train['AirChamberPressureBalance1'] = data_train['AirChamberPressureValue_1']-data_train['AirChamberPressureTarget_1']
    data_test['AirChamberPressureBalance1'] = data_test['AirChamberPressureValue_1']-data_test['AirChamberPressureTarget_1']
    data_train['AirChamberPressureBalance2'] = data_train['AirChamberPressureValue_1']-data_train['AirChamberPressureTarget_1']
    data_test['AirChamberPressureBalance2'] = data_test['AirChamberPressureValue_2']-data_test['AirChamberPressureTarget_2']
    return data_train,data_test