# 個体の生成
import random
import numpy as np

class Generator(object):
    def __init__(self, maximum, minimum, dimension, size):
        """constractor
        Args :
            maximum (float) : 遺伝子の値の最大値
            minimum (float) : 遺伝子の値の最小値
            dimension (int) : パラメータ数
            size (int) : 個体数
        """
        self._maximum = maximum
        self._minimum = minimum
        self._dimension = dimension
        self._size = size

    def generate(self):
        """初期個体集団を生成する
        Returns :
            np.array（配列）：1行が1個体
        """
        value_range = self._maximum - self._minimum
        # （ランダム値 for range(次元数)）for range(個体数)　で個体の2次元配列（np.array()）を作る
        return np.array([[value_range * np.random.rand() + self._minimum for _ in range(self._dimension)] for _ in range(self._size)])


if __name__ == "__main__":
    generator = Generator(10, 0, 4, 3)
    individual_set = generator.generate()
    print(individual_set)
