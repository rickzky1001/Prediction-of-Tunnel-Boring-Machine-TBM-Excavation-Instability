import numpy as np
def delete_label(statistics_result):
    label_column = statistics_result[:, :, 4*16:4*16+1]
    statistics_result = np.concatenate((statistics_result, label_column), axis=2)
    statistics_result = np.delete(statistics_result, np.s_[4*16:4*16+4], axis=2)
    return statistics_result