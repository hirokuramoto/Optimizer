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

        initial_set = np.empty((0, self._dimension))
        for j in range(self._size):
            gene = np.array([], dtype = np.float64)
            for i in range(self._dimension):
                # 遺伝子の存在範囲を決定
                value_range = self._maximum[i] - self._minimum[i]
                temp = value_range * np.random.rand() + self._minimum[i]
                gene = np.append(gene, temp)
            initial_set = np.vstack((initial_set, gene))
        return initial_set


if __name__ == "__main__":

    maximum = np.array([10, 8])
    minimum = np.array([0, 2])
    generator = Generator(maximum, minimum, 2, 3)
    individual_set = generator.generate()
    print(individual_set)
