# 世代交代モデルの定義

import random
import numpy as np
import copy
from .individual_selector import *
from .evaluator import *

class JGG(object):
    """ Just Generation Gap による世代交代
        子個体からエリート個体を選択し，親個体と入れ替える
    Args :
        individual_set : 個体の2次元配列
        parents_index : 親個体のindex配列
        children_set : 生成した子個体
    """
    def __init__(self):
        pass

    def select(self, individual_set, parents_index, children_set, children_value):
        self._individual_set = copy.deepcopy(individual_set)
        self._parents_index = parents_index
        self._children_set = children_set
        self._children_value = children_value

        individual_set = self._individual_set

        # 子個体からエリート個体のindexを取得
        elite_index = EliteSelector(self._parents_index.size).select(self._children_set, self._children_value)

        # 個体入れ替え
        for p, e in zip(self._parents_index, elite_index):
            individual_set[p] = self._children_set[e]
        return individual_set
