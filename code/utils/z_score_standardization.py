import pandas as pd
def z_score_norm(df, exclude_columns=['Instability','RingNumber','TunnelingState']):
    if exclude_columns is None:
        exclude_columns = []
    
    # 选择需要标准化的列
    columns_to_normalize = [col for col in df.columns if col not in exclude_columns]
    
    # 对选定的列进行标准化
    df_standardized = df.copy()
    df_standardized[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()
    
    return df_standardized