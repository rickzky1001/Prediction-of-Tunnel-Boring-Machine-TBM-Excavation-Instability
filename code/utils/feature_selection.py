import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

def feature_importance_select(X_train,y_train):
    # 将数据调整为一维，适应模型输入
    X_train = X_train.reshape(-1, X_train.shape[-1])
    y_train = y_train.reshape(-1)
    
    # 训练模型
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    # 获取特征重要性
    importance = model.feature_importances_
    
    # 获取非零重要性的特征索引
    selected_index = [index for index, value in enumerate(importance) if value != 0]
    return importance, selected_index
