# グラム行列の計算

import numpy as np
import pandas as pd


class StandardData(object):
    """訓練データを取得して標準化した訓練データ配列を返す
    """

    def __init__(self, design_variables, objective_variables):
        # 設計変数の指定
        self._design_variables = design_variables

        # 目的関数の指定
        self._objective_variables = objective_variables


    def standard(self, filepath):
        """constractor
        Args :
            filepath : 保存したcsvデータのファイルパス＋ファイル名
            num : テストデータのN数
        """

        # 訓練データの取得（クラス変数）
        training_data = pd.read_csv(filepath, header = 0)

        # 訓練データの2次元配列化
        design_arr = np.array(training_data.iloc[0:, 0:self._design_variables])
        object_arr = np.array(training_data.iloc[0:, self._design_variables:])

        # 訓練データの標準化(axis=0：行ごと，ddof=0で分散，ddof=1で不偏分散)
        arr_mean = design_arr.mean(axis=0, keepdims=True)
        arr_std = design_arr.std(axis=0, keepdims=True, ddof=0)
        design_arr = (design_arr - arr_mean) / arr_std

        # 設計変数データと結果データを結合
        arr = np.concatenate([design_arr, object_arr], 1)
        return arr


if __name__ == "__main__":
    test = StandardData(5,2)
    arr = test.standard("result.csv")
    #std_arr = test.standard(arr, axis=0, ddof=0) # axis=0で行ごと，ddof=0で標本分散（n-1）にしない
    print(arr)
