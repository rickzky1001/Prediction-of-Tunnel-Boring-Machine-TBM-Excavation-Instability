import numpy as np
from scipy.stats import pearsonr

def calculate_correlation(bag, indices):
    """计算特征间的皮尔逊相关系数"""
    correlations = []
    for (i, j) in indices:
        # 提取两个特征的值
        feature_i = bag[:, i]
        feature_j = bag[:, j]
        # 计算皮尔逊相关系数
        corr, _ = pearsonr(feature_i, feature_j)
        correlations.append(corr)
    return correlations

def add_correlation_features(result_array, bags_list):
    # 定义我们关心的特征对
    correlation_pairs = [(3, 4), (5, 14), (6, 14)]
    
    new_features = []

    # 遍历每个包
    for bag in bags_list:
        bag = bag.astype(float)  # 确保是float类型
        
        # 对每个窗口，计算特征之间的相关性
        bag_correlation = []
        for i in range(bag.shape[0]):  # 遍历窗口
            correlations = calculate_correlation(bag[i], correlation_pairs)

            bag_correlation.append(correlations)
        
        # 将相关性系数添加到新的特征中
        
        new_features.append(np.array(bag_correlation))

    # 将新的特征堆叠到原始的 result_array 后面
    new_features_array = np.array(new_features)  # 形状: (356, 10, len(correlation_pairs))
    result_array = np.concatenate([result_array, new_features_array], axis=-1)  # 在最后一维拼接
    
    return result_array

# 假设 bags_list 是你的原始数据，result_array 是你计算的统计特征的数组
# result_array的形状应为 (356, 10, 23*6)
# bags_list的形状应为 (356, 10, 150, 23)