import numpy as np
from prediction import Prediction
from NSGA2 import Generator

class Areaplot(object):
    """docstring for Montecarlo."""
    def __init__(self, design_data, hyper_set, alpha_vector):
        self.design_data  = design_data
        self.hyper_set    = hyper_set
        self.alpha_vector = alpha_vector

    def calc(self):
        maximum = self.design_data.max(axis=0)   # 遺伝子の値の最大値を列ごとに求める
        minimum = self.design_data.min(axis=0)   # 遺伝子の値の最小値を列ごとに求める
        dimension = self.design_data.shape[1]    # パラメータ数
        size = 1000                              # 個体数

        generator = Generator(maximum, minimum, dimension, size)
        test_set  = generator.generate()

        # 予測値の計算
        data = Prediction(size, self.hyper_set, self.design_data, test_set, self.alpha_vector)
        predict = data.predict()

        return np.hstack([predict, test_set])
