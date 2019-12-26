# 評価関数による評価

import numpy as np
import copy
from .generator import Generator
from .standard_data import StandardData
from .leave_one_out import LeaveOneOut
from prediction import Prediction


class CrossValidation(object):
    def __init__(self, data, design_variables, objective_variables):
        # CrossValidation を使うときは引数が１つ増えるので注意
        self._data = data       # テストデータ
        self._design_variables = design_variables
        self._objective_variables = objective_variables

    def evaluate(self, individual_set):
        """constractor
        Args :
            individual_set (np.array) : 個体の2次元配列
            data : 訓練データの2次元配列
        Returns :
            np.array（1次元配列）：評価値配列
        """
        self._individual_set = copy.deepcopy(individual_set)


        # 設計変数の個数指定
        design_variables = self._design_variables

        # 目的関数の個数指定
        objective_variables = self._objective_variables

        #テストデータのN数を取得
        data_size = self._data.shape[0]

        individual_set = self._individual_set

        # individual_setの行数（個体数）を取得
        size = individual_set.shape[0]

        # テストデータの設計変数と目的関数を取得
        design = np.array(self._data[0:, 0:design_variables])
        object = np.array(self._data[0:, design_variables-1:-1])

        evaluate_set = np.array([], dtype = np.float64)
        for i in range(size):
            x = LeaveOneOut(individual_set[i, 0], individual_set[i, 1], individual_set[i, 2], individual_set[i, 3], design, object)
            result = x.cross_validation()
            evaluate_set = np.append(evaluate_set, result)

        return evaluate_set

class CrowdingDistance(object):
    def __init__(self, design_data, ndesign, nobject, hyper_set, alpha_vector):

        self._design_data  = design_data
        self._ndesign      = ndesign
        self._nobject      = nobject
        self._hyper_set    = hyper_set
        self._alpha_vector = alpha_vector

    def evaluate(self, individual_set):
        '''予測値を計算してランク付けと混雑度ソート
        '''
        design_data  = self._design_data
        hyper_set    = self._hyper_set
        alpha_vector = self._alpha_vector

        # individual_setの行数（個体数）を取得
        size = individual_set.shape[0]

        # 予測値の計算
        data = Prediction(size, hyper_set, design_data, individual_set, alpha_vector)
        predict_value = data.predict()

        # 非優越ソートによるランキング
        # rank用,混雑度用の列を追加
        rank  = np.ones((size, 1))
        crowd = np.zeros((size, 1))
        mat = np.append(predict_value, rank, axis=1)
        mat = np.append(mat, crowd, axis=1)
        # rankの計算
        for i in range(size):
            for j in range(size):
                if mat[i, 0]>mat[j, 0] and mat[i, 1]>mat[j, 1]:
                    mat[i, 2] = mat[i, 2] + 1

        # rankでソート
        mat = mat[np.argsort(mat[:, 2])]

        # rankの最大値を取得
        max_rank = int(np.max(mat, axis=0)[2])

        # 各目的関数の最大・最小値を取得
        max1 = np.max(mat, axis=0)[0]
        max2 = np.max(mat, axis=0)[1]
        min1 = np.min(mat, axis=0)[0]
        min2 = np.min(mat, axis=0)[1]

        nrank = np.array([], dtype = np.int64)
        sort_mat = np.array([], dtype = np.float64)

        # rank毎に混雑度を計算
        for i in range(1, max_rank+1):
            # 各rankの個体数を取得
            count = np.count_nonzero(mat == i, axis=0)[2]

            # rank毎に混雑度を計算
            if count == 0:
                continue
            else:
                # 同一rankの行を抽出
                mat_temp = mat[np.any(mat==i, axis=1)]
                # 同一rank内で1個めの目的関数でソート
                mat1 = mat_temp[np.argsort(mat_temp[:, 0])]

                # 混雑度距離を計算
                if count >= 3:
                    for j in range(count):
                        # 境界個体に対して最大距離を与える
                        if j == 0 or j == count - 1:
                            mat1[j, 3] = 10**10
                        # 境界個体以外に対して混雑度距離を計算する
                        else:
                            mat1[j, 3] = (mat1[j+1, 0] - mat1[j-1, 0])/(max1 - min1)
                else:
                    for j in range(count):
                        mat1[j, 3] = 10**10

                # 同一rank内で2個めの目的関数でソート
                mat2 = mat1[np.argsort(mat1[:, 1])]

                # 混雑度距離を計算
                if count >= 3:
                    for j in range(count):
                        if j == 0 or j == count - 1:
                            mat2[j, 3] = mat2[j, 3] + 10**10
                        else:
                            mat2[j, 3] = mat2[j, 3] + (mat2[j+1, 0] - mat2[j-1, 0])/(max2 - min2)
                else:
                    for j in range(count):
                        mat2[j, 3] = mat2[j, 3] + 10**10

                sort_mat = np.append(sort_mat, mat2)
        # 1次元配列を2次元配列に変換
        sort_mat = sort_mat.reshape([size, -1])

        evaluate_set = np.array([], dtype = np.float64)
        evaluate_set = sort_mat[:, 2:4]

        return evaluate_set
