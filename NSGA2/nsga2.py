# メインファイル
from RealCodedGA import StandardData
from .generator import Generator
from .nsga2_evaluator import CrowdingDistance
from .individual_selector import Tournament
from .crossover import *
from .generation_selector import JGG
from .sort import NonDominatedSort
import numpy as np
import time
from matplotlib import pyplot as plt

class NSGA2(object):
    """NSGA2によりパレート解集合を求める
    Args:
        filepath (str) : 結果ファイルのパス
        alpha_vector (array) : 重み係数ベクトル
        ndesign (int) : 設計変数の数
        nobject (int) : 目的関数の数
    Returns :
        pareto_set (array) : パレート最適解集合
    """

    def __init__(self, workdir, design_data, alpha_vector, hyper_set, ndesign, nobject):
        self._workdir      = workdir
        self._design_data  = design_data
        self._alpha_vector = alpha_vector
        self._hyper_set    = hyper_set
        self._ndesign      = ndesign
        self._nobject      = nobject

    def nsga2(self):
        # 訓練用データ
        workdir      = self._workdir        # 作業ディレクトリのパス
        ndesign      = self._ndesign        # 設計変数の数
        nobject      = self._nobject        # 目的関数の数
        design_data  = self._design_data    # 訓練データの設計変数の2次元配列
        hyper_set    = self._hyper_set      # ハイパーパラメータ
        alpha_vector = self._alpha_vector   # 重み係数ベクトル

        # GAパラメータ
        maximum = design_data.max(axis=0)   # 遺伝子の値の最大値を列ごとに求める
        minimum = design_data.min(axis=0)   # 遺伝子の値の最小値を列ごとに求める
        dimension = ndesign                 # パラメータ数
        size = 100                          # 個体数(4の倍数)
        generation_loop = 300               # 繰り返し数

        # 初期個体の生成
        generator = Generator(maximum, minimum, dimension, 2*size)
        initial_set = generator.generate()

        # 評価関数
        evaluator = CrowdingDistance(design_data, hyper_set, alpha_vector)

        # 初期個体の評価
        individual_set0 = evaluator.evaluate(initial_set)

        # メイン処理
        for i in range(generation_loop):

            # ランクと混雑度による非優越ソート
            if i == 0:
                archive_set = NonDominatedSort(individual_set0).ndsort()[:size]
                hoge = archive_set[:]
            else:
                archive_set = NonDominatedSort(individual_set).ndsort()[:size]

            # 混雑度トーナメント選択により新たな探索母集団を生成
            search_set = Tournament(archive_set).tournament()

            # 交叉
            #children = BLXalpha(dimension * 20).crossover(search_set, maximum, minimum)
            children = Simplex(dimension * 40).crossover(search_set, maximum, minimum)

            # 子集団の評価
            children_value = evaluator.evaluate(children)

            # 非優越ソート
            children_set = NonDominatedSort(children_value).ndsort()

            # 親集団と子集団の入れ替え
            individual_set = JGG(size, archive_set, children_set).select()
            if i == 0:
                huga = individual_set[:]

            if i == generation_loop - 1:
                archive_set2 = NonDominatedSort(individual_set).ndsort()[:size]

            pro_size = 10
            pro_bar = ('=' * int(10*(i+1)/generation_loop)) + (' ' * (pro_size - int(10*(i+1)/generation_loop)))
            print('\r(2/2)[{0}] {1}%'.format(pro_bar, int((i+1) / generation_loop * 100)), end='')

        # パレート解集合を返す
        return archive_set2
