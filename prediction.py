import numpy as np

class Prediction(object):
    def __init__(self, size, hyper_set, design_data, test_data, alpha_vector):
        '''カーネル関数（ベクトル）を計算して予測値のリストを返す
        Args :
            size (int): 個体数
            hyper_set (array): ハイパーパラメータ
            design_data (array): 結果の2次元配列
            test_data (array): 個体の2次元配列
            alpha_vector (float): 目的関数の重み係数ベクトル
        '''
        self._size          = size
        self._beta1         = hyper_set[0, 0]
        self._beta2         = hyper_set[1, 0]
        self._test_data     = test_data
        self._alpha_vector1 = np.array(alpha_vector[0:, 0])
        self._alpha_vector2 = np.array(alpha_vector[0:, 1])
        self._design_data   = design_data


    def predict(self):
        # 訓練データの行数（個体数）を取得
        data_size = self._design_data.shape[0]

        # テストデータを用いてカーネル関数を求める
        value_vector1 = np.array([], dtype = np.float64)
        value_vector2 = np.array([], dtype = np.float64)

        for j in range(self._size):
            kernel_vector1 = np.array([], dtype = np.float64)
            kernel_vector2 = np.array([], dtype = np.float64)

            for i in range(data_size):
                element1 = np.exp(-1 * self._beta1 * np.inner(self._design_data[i,] - self._test_data[j,], self._design_data[i,] - self._test_data[j,]))
                element2 = np.exp(-1 * self._beta2 * np.inner(self._design_data[i,] - self._test_data[j,], self._design_data[i,] - self._test_data[j,]))
                kernel_vector1 = np.append(kernel_vector1, element1)
                kernel_vector2 = np.append(kernel_vector2, element2)
            # 予測値
            value1 = np.inner(kernel_vector1, self._alpha_vector1)
            value2 = np.inner(kernel_vector2, self._alpha_vector2)
            # 予測値の配列
            value_vector1 = np.append(value_vector1, value1)
            value_vector2 = np.append(value_vector2, value2)

        predict_value = np.array([value_vector1, value_vector2])

        # 予測値の2次元配列を返す
        return predict_value.T
