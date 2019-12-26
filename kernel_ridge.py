# ガウスカーネルによるカーネルリッジ回帰を行う

import numpy as np
from RealCodedGA import CallFortran

class KernelRidge(object):
    def __init__(self, hyper_set, ndesign, design_data, object_data1, object_data2):
        """ガウスカーネルを使った回帰を行い重み係数ベクトル、予測値リストを返す
        Args :
            hyper_set (np.array) : ガウスパラメータ　β, 正則化パラメータ　λ
            ndesign (int) : 設計変数の数
            design_data (np.array) : 標準化済みの設計変数配列(訓練データ)
            object_data (np.array) : 結果のデータ（訓練データ）
        Returns :
            alpha_vector(np.array) : 重み係数ベクトル
            predict_value(list)    : 予測値
        """

        self._beta1        = hyper_set[0, 0]
        self._beta2        = hyper_set[1, 0]
        self._lamda1       = hyper_set[0, 1]
        self._lamda2       = hyper_set[1, 1]
        self._ndesign      = ndesign
        self._design_data  = design_data
        self._object_data1 = object_data1
        self._object_data2 = object_data2


    def kernel_ridge(self):

        # 訓練データの行数（個体数）を取得
        data_size = self._design_data.shape[0]

        # 設計変数配列を取得
        design_data = self._design_data

        # 訓練データの結果配列を取得
        object_data1 = self._object_data1
        object_data2 = self._object_data2

        beta1 = self._beta1
        beta2 = self._beta2
        lamda1 = self._lamda1
        lamda2 = self._lamda2
        ndesign = self._ndesign

        # グラム行列の計算
        # 渡す行列データを転置（Fortranは列majorのため）
        design_data = design_data.T
        gram_matrix1 = np.identity(data_size)
        gram_matrix2 = np.identity(data_size)
        call1 = CallFortran(data_size, ndesign, beta1, design_data, gram_matrix1)
        call2 = CallFortran(data_size, ndesign, beta2, design_data, gram_matrix2)
        call1.call_fortran()
        call2.call_fortran()
        # 転置したデータを元に戻しておく
        design_data = design_data.T

        # 重みベクトルの計算
        i_mat = np.identity(data_size)

        alpha_vector1 = np.dot(np.linalg.inv(gram_matrix1 + lamda1 * i_mat), object_data1)
        alpha_vector2 = np.dot(np.linalg.inv(gram_matrix2 + lamda2 * i_mat), object_data2)

        alpha = np.array([], dtype = np.float64)
        alpha = np.hstack([alpha_vector1, alpha_vector2])
        return alpha
