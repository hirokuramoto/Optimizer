# 世代交代モデルの定義

import random
import numpy as np
import copy

class JGG(object):
    """Just Generation Gap による世代交代
    """
    def __init__(self, size, archive_set, children_set):
        self._size = size
        self._archive_set = archive_set
        self._children_set = children_set

    def select(self):
        """子個体からエリート個体を選択し，親個体と入れ替える
        """
        # ソート済みchildren_setからsize分を選択する
        size = int(self._size)
        return np.vstack((self._archive_set, self._children_set[:size]))
