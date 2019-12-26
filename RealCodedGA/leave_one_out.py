# ガウスパラメータ，正則化パラメータの決定のためにCV値を求める

import numpy as np
from .call_fortran import *

class LeaveOneOut(object):
    def __init__(self, beta1, lamda1, beta2, lamda2, design_data, object_data):
        """ガウスカーネルを使った予測値ベクトルを返す
        Args :
            beta (float) : ガウスパラメータ　β
            penalty (float) : 正則化パラメータ　λ
            design_data (np.array) : 標準化済みの設計変数配列
            object_data (np.array) : 訓練データの結果配列
        Returns :
        """

        self._beta1 = beta1
        self._lamda1 = lamda1
        self._beta2 = beta2
        self._lamda2 = lamda2
        self._design_data = design_data
        self._object_data = object_data

    def cross_validation(self):
        # テストデータの行数（個体数）を取得
        size = self._design_data.shape[0]
        # 設計変数配列を取得
        design_data = self._design_data
        # 訓練データの結果配列を取得
        object_data = self._object_data
        # パラメータ数を取得
        n_param = design_data.shape[1]

        beta1 = self._beta1
        beta2 = self._beta2
        lamda1 = self._lamda1
        lamda2 = self._lamda2

        # グラム行列の計算
        #gram_matrix = np.identity(data_size)
        #for i in range(data_size):
        #    for k in range(i + 1, data_size):
        #        gram_matrix[i][k] = np.exp(-1 * self._beta * np.inner(design_data[i,] - design_data[k,], design_data[i,] - design_data[k,]))
        #        gram_matrix[k][i] = gram_matrix[i][k]

        # Fortranのグラム行列計算用サブルーチンを呼び出す
        # 渡す行列データを転置（Fortranは列majorのため）
        design_data = design_data.T
        gram_matrix1 = np.identity(size)
        gram_matrix2 = np.identity(size)
        call1 = CallFortran(size, n_param, beta1, design_data, gram_matrix1)
        call2 = CallFortran(size, n_param, beta2, design_data, gram_matrix2)
        call1.call_fortran()
        call2.call_fortran()

        # 重みベクトルの計算
        i_mat1 = np.identity(size)
        i_mat2 = np.identity(size)
        alpha_vector1 = np.dot(np.linalg.inv(gram_matrix1 + lamda1 * i_mat1), object_data)
        alpha_vector2 = np.dot(np.linalg.inv(gram_matrix2 + lamda2 * i_mat2), object_data)

        # 予測値の計算
        predict_vector1 = np.dot(gram_matrix1, alpha_vector1)
        predict_vector2 = np.dot(gram_matrix2, alpha_vector2)

        # H行列の計算
        i_mat1 = np.identity(size)
        i_mat2 = np.identity(size)
        h_matrix1 = np.dot(np.linalg.inv(gram_matrix1 + lamda1 * i_mat1), gram_matrix1)
        h_matrix2 = np.dot(np.linalg.inv(gram_matrix2 + lamda2 * i_mat2), gram_matrix2)

        # CV値の計算
        cv_value1 = 0.
        cv_value2 = 0.
        for i in range(size):
            for j in range(2):
                cv_value1 += 1/size * ((object_data[i, j] - predict_vector1[i, j])/(1 - h_matrix1[i][i])) ** 2
                cv_value2 += 1/size * ((object_data[i, j] - predict_vector2[i, j])/(1 - h_matrix2[i][i])) ** 2
        return cv_value1 + cv_value2
