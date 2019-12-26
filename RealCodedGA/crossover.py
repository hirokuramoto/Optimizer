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


    def crossover(self, individual_set, parents_index):
        """次元数+1個体から子個体を生成する
        """

        self._individual_set = copy.deepcopy(individual_set)
        self._parents_index = copy.deepcopy(parents_index)

        individual_set = self._individual_set
        parents_index = self._parents_index

        # 親個体を抽出して2次元配列化（親個体数×次元数）
        matrix_parents = np.array([individual_set[i] for i in parents_index])
        # 列ごと(axis=0)の平均値を求める（1×次元数）
        center = matrix_parents.mean(axis = 0)
        # 設計変数の数を抽出
        dimension = len(center)

        alpha = math.sqrt(dimension + 2)

        # 列ごと(axis=0)の最大・最小を求める（1×次元数）
        gene_max = matrix_parents.max(axis=0)
        gene_min = matrix_parents.min(axis=0)
        gene_abs = np.abs(gene_max - gene_min)

        # matrix行列の最終行がp^nに相当
        matrix = center + alpha * (matrix_parents - center)
        for j in range(matrix_parents.shape[0]):
            for i in range(dimension):
                if gene_min[i] > matrix[j, i]:
                    matrix[j, i] = gene_min[i]
                elif gene_max[i] < matrix[j, i]:
                    matrix[j, i] = gene_max[i]
                else:
                    pass
        # 空配列を用意
        children_set = np.array([], dtype = np.float64)

        for _ in range(self._generate_size):

            # 子1個体の遺伝子を格納する空配列を用意
            gene = np.zeros(dimension)
            for k, (vector1, vector2) in enumerate(zip(matrix, matrix[1:])):
                r_k = random.uniform(0., 1.) ** (1./(k+0.5*dimension))
                gene = r_k * (vector1 - vector2 + gene)
            gene += matrix[-1]
            children_set = np.append(children_set, gene)

        # 子個体数×次元数の2次元配列に整理
        children_set = children_set.reshape(self._generate_size, -1)

        children_set = np.clip(children_set, 0, None)
        return children_set

class BLXalpha(object):
    def __init__(self, generate_size, alpha = 0.05):
        self._generate_size = generate_size
        self._alpha = alpha

    def crossover(self, individual_set, parents_index):
        """2個体から子個体を生成する
        """
        self._individual_set = copy.deepcopy(individual_set)
        self._parents_index = copy.deepcopy(parents_index)

        individual_set = self._individual_set
        parents_index = self._parents_index

        # 親個体を抽出して2次元配列化（親個体数×次元数）
        matrix_parents = np.array([individual_set[i] for i in parents_index])

        # 列ごと(axis=0)の最大・最小を求める（1×次元数）
        gene_max = matrix_parents.max(axis=0)
        gene_min = matrix_parents.min(axis=0)
        gene_abs = np.abs(gene_max - gene_min)

        # 探索範囲を拡大　np.clip(arr, 最小, 最大)で範囲内に収める
        gene_max = np.clip((gene_max + self._alpha * gene_abs), 0, None)
        gene_min = np.clip((gene_min - self._alpha * gene_abs), 0, None)

        # 空配列を用意
        children_set = np.array([], dtype = np.float64)

        for _ in range(self._generate_size):
            gene = [random.uniform(g_min, g_max) for g_max, g_min in zip(gene_max, gene_min)]

            children_set = np.append(children_set, gene)

        # 子個体数×次元数の2次元配列に整理
        children_set = children_set.reshape(self._generate_size, -1)
        return children_set
