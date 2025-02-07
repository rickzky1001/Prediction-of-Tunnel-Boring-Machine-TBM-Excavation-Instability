import numpy as np
def statistics_comp(bags_list):
    stats_list = []

    # 遍历每个包
    for bag in bags_list:
        bag= bag.astype(float)

        # 创建一个空数组来存储统计量，形状为 (10, 23 * 4)
        stats_bag = np.zeros((bag.shape[0], bag.shape[2] * 4))

        # 遍历每个窗口
        for i in range(bag.shape[0]):
            # 遍历每个特征
            for j in range(bag.shape[2]):
                # 提取该特征的所有数据（即所有窗口的第 j 个特征）
                feature_data = bag[i, :, j]
                # 计算统计量
                mean_value = np.mean(feature_data)
                variance_value = np.var(feature_data)
                max_value = np.max(feature_data)
                min_value = np.min(feature_data)
                # skewness_value = skew(feature_data)
                # kurtosis_value = kurtosis(feature_data)

                # 将结果存入 stats_bag 中
                stats_bag[i, j*4] = mean_value
                stats_bag[i, j*4 + 1] = variance_value
                stats_bag[i, j*4 + 2] = max_value
                stats_bag[i, j*4 + 3] = min_value
                # stats_bag[i, j*6 + 4] = skewness_value
                # stats_bag[i, j*6 + 5] = kurtosis_value

        # 将这个包的统计量加入到 stats_list 中
        stats_list.append(stats_bag)
    return np.array(stats_list)