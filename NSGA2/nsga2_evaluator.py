# 評価関数による評価

import numpy as np
import copy
from prediction import Prediction


class CrowdingDistance(object):
    def __init__(self, design_data, hyper_set, alpha_vector):

        self._design_data  = design_data
        self._hyper_set    = hyper_set
        self._alpha_vector = alpha_vector

    def evaluate(self, data_set):
        '''予測値を計算
        '''
        self._data_set = copy.deepcopy(data_set)
        design_data  = self._design_data
        hyper_set    = self._hyper_set
        alpha_vector = self._alpha_vector

        # individual_setの行数（個体数）を取得
        size = self._data_set.shape[0]

        # 予測値の計算
        data = Prediction(size, hyper_set, design_data, self._data_set, alpha_vector)
        predict_value = data.predict()

        # rank用,混雑度用の列を追加
        rank  = np.zeros((size, 1))
        crowd = np.zeros((size, 1))
        mat0 = np.hstack([predict_value, rank, crowd, self._data_set])
        return mat0
