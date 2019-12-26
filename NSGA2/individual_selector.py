# 混雑度トーナメント選択により新たな探索母集団Ｑt+1を生成

import numpy as np
import random
import copy

class Tournament(object):
    """混雑度トーナメント選択
    """
    def __init__(self, archive_set):
        self._archive_set = copy.deepcopy(archive_set)

    def tournament(self):
        # アーカイブ母集団の個体数分の探索母集団を生成
        size = int(self._archive_set.shape[0])

        search_set = np.array([], dtype = np.float64)
        for i in range(size):
            rnd1 = random.randrange(size)
            rnd2 = random.randrange(size)

            # まずランクで比較
            if self._archive_set[rnd1, 2] < self._archive_set[rnd2, 2]:
                search_set = np.append(search_set, self._archive_set[rnd1, :])

            elif self._archive_set[rnd1, 2] > self._archive_set[rnd2, 2]:
                search_set = np.append(search_set, self._archive_set[rnd2, :])

            # 次に混雑度距離で比較
            elif self._archive_set[rnd1, 3] > self._archive_set[rnd2, 3]:
                search_set = np.append(search_set, self._archive_set[rnd1, :])

            else:
                search_set = np.append(search_set, self._archive_set[rnd2, :])

        search_set = search_set.reshape(size, -1)

        return search_set
