# 最適化計算のメインファイル

import numpy as np
import pandas as pd
from kernel_ridge import KernelRidge
from prediction import Prediction
from areaplot import Areaplot
from RealCodedGA import RealGA, StandardData
from NSGA2 import NSGA2, CrowdingDistance

class Optimize(object):
    """result.csvを読み込んで最適化計算を行う
    Args:
        filepath : csvデータのファイルパス
        ndesign  : 設計変数の数(int)
        nobject  : 目的関数の数(int)
    Returns:
        alpha_vector : 重み係数ベクトル(float)
    """
    def __init__(self, workdir, filepath, ndesign, nobject):
        self._workdir   = workdir
        self._filepath  = filepath
        self._ndesign   = ndesign
        self._nobject   = nobject

        # 結果データを標準化して読み込み(1行目はヘッダー行、1列目はインデックス列)
        test = StandardData(self._ndesign, self._nobject)
        data = test.standard(self._filepath)
        self._design_data  = np.array(data[:,                   : self._ndesign    ])
        self._object_data1 = np.array(data[:, self._ndesign     : self._ndesign + 1])
        self._object_data2 = np.array(data[:, self._ndesign + 1 : self._ndesign + 2])
        # 平均値
        self._mean = self._design_data.mean(axis=0, keepdims=True)
        # 標準偏差
        self._std = self._design_data.std(axis=0, keepdims=True, ddof=0)


    def hyperparameter(self):
        # カーネルリッジ回帰のハイパーパラメータを取得
        #hyperparam = RealGA(self._filepath, self._ndesign, self._nobject).realga()
        #hyperparam = np.array([[0.06710571, 0.00077507], [0.03601937, 0.00086607]]) # 熱交換器
        hyperparam = np.array([[0.04474047, 0.00027773], [0.04613555, 0.00117033]]) # ブラケット
        hyper_set = hyperparam.reshape([2, 2])
        return hyper_set


    def optimize(self, hyper_set):
        self.hyper_set = hyper_set
        hyper_set = self.hyper_set

        # 重み係数ベクトルαの計算
        vector = KernelRidge(hyper_set, self._ndesign, self._design_data, self._object_data1, self._object_data2)
        alpha_vector = vector.kernel_ridge()

        # パレート最適解集合を求める
        pareto_set = NSGA2(self._workdir, self._design_data, alpha_vector, hyper_set, self._ndesign, self._nobject).nsga2()
        # 標準化されたデータを元に戻す
        data1 = pareto_set[:,                  :self._nobject + 2]
        data2 = pareto_set[:, self._nobject + 2:             ]
        data2 = (data2 * self._std ) + self._mean
        pareto_set = np.concatenate([data1, data2], 1)
        np.savetxt(self._workdir + '/best_pop.csv', pareto_set, delimiter=',')


    def areaplot(self, hyper_set):
        self.hyper_set = hyper_set
        hyper_set = self.hyper_set

        # 重み係数ベクトルαの計算
        vector = KernelRidge(hyper_set, self._ndesign, self._design_data, self._object_data1, self._object_data2)
        alpha_vector = vector.kernel_ridge()

        # ランダムな設計変数の組み合わせに対する予測値の計算
        area = Areaplot(self._design_data, hyper_set, alpha_vector).calc()

        # 標準化されたデータを元に戻す
        data1 = area[:, :self._nobject]
        data2 = area[:, self._nobject:]
        data2 = (data2 * self._std ) + self._mean
        area = np.concatenate([data1, data2], 1)
        np.savetxt(self._workdir + '/areaplot.csv', area, delimiter=',')


if __name__ == "__main__":
    workdir = '/home/kuramoto/Work/5_pyQt'
    filepath = '/home/kuramoto/Work/5_pyQt/result.csv'
    opt = Optimize(workdir, filepath, 5, 2)
    monte = opt.optimize()
