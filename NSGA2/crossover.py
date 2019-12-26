# 世代交代モデルの定義
import random
import math
import numpy as np
import copy


class Simplex(object):
    def __init__(self, generate_size):
        """constractor
        Args :
            generate_size (int) : 交叉によって生成する個体数
        """
        self._generate_size = generate_size

    def crossover(self, search_set, maximum, minimum):
        """次元数+1個体から子個体を生成する
        """
        self._search_set = copy.deepcopy(search_set)
        self._maximum = maximum
        self._minimum = minimum

        # 親個体の設計変数部分を抽出
        matrix_p = self._search_set[:, 4:]
        # 列ごと(axis=0)の平均値を求める（1×次元数）
        center = matrix_p.mean(axis = 0)
        # 設計変数の数を抽出
        dimension = len(center)
        # 拡張率の計算（次元数+2の平方根）
        alpha = math.sqrt(dimension + 2)
        # 子ベクトルの存在範囲
        matrix = center + alpha * (matrix_p - center)
        #for i in range(dimension):
        #    np.where(matrix[:, i] > self._minimum[i], matrix[:, i], self._minimum[i])
        #    np.where(matrix[:, i] < self._maximum[i], matrix[:, i], self._maximum[i])
        for j in range(matrix_p.shape[0]):
            for i in range(dimension):
                if self._minimum[i] > matrix[j, i]:
                    matrix[j, i] = self._minimum[i]
                elif self._maximum[i] < matrix[j, i]:
                    matrix[j, i] = self._maximum[i]
                else:
                    pass

        # 空配列を用意
        children_set = np.array([], dtype = np.float64)

        for _ in range(self._generate_size):

            # 子1個体の遺伝子を格納する空配列を用意
            gene = np.zeros(dimension)
            for k, (vector1, vector2) in enumerate(zip(matrix, matrix[1:])):
                #while np.any(gene > self._maximum) or np.any(gene < self._minimum):
                r_k = random.uniform(0., 1.) ** (1./(k+1.0*dimension))
                gene = r_k * (vector1 - vector2 + gene)
            gene += matrix[-1]
            children_set = np.append(children_set, gene)

        # 子個体数×次元数の2次元配列に整理
        children_set = children_set.reshape(self._generate_size, -1)

        return children_set

class BLXalpha(object):
    def __init__(self, generate_size, alpha = 0.0):
        self._generate_size = generate_size
        self._alpha = alpha

    def crossover(self, search_set, maximum, minimum):
        """2個体から子個体を生成する
        """
        self._search_set = copy.deepcopy(search_set)
        self._maximum = maximum
        self._minimum = minimum

        # 親個体の設計変数部分を抽出
        matrix = self._search_set[:, 4:]

        # 次元数を取得
        dimension = matrix.shape[1]

        # 列ごと(axis=0)の最大・最小を求める
        gene_max = matrix.max(axis=0)
        gene_min = matrix.min(axis=0)
        gene_abs = np.abs(gene_max - gene_min)

        # 探索範囲を拡大　np.clip(arr, 最小, 最大)で範囲内に収める
        for i in range(dimension):
            gene_max[i] = np.clip((gene_max[i] + self._alpha * gene_abs[i]), self._minimum[i], self._maximum[i])
            gene_min[i] = np.clip((gene_min[i] - self._alpha * gene_abs[i]), self._minimum[i], self._maximum[i])

        # 空配列を用意
        children = np.array([], dtype = np.float64)
        # 探索範囲の中からランダムに子を生成
        for _ in range(self._generate_size):
            for i in range(dimension):
                gene = random.uniform(gene_max[i], gene_min[i])
                children = np.append(children, gene)

        # 子個体数×次元数の2次元配列に整理
        children_set = children.reshape(self._generate_size, -1)

        return children_set
