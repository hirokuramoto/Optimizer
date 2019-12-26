import numpy as np
import copy

class NonDominatedSort(object):
    '''非優越ソートを実行
    '''
    def __init__(self, data_set):
        self._data_set = data_set

    def ndsort(self):
        mat = copy.deepcopy(self._data_set)
        mat[:, 2:4] = 0

        # individual_setの行数（個体数）を取得
        row = mat.shape[0]
        col = mat.shape[1]

        rmat = np.empty((0, col))

        # 非優越ソートに基づくランキング
        rank1 = 0
        rank2 = 0

        for _ in range(mat.shape[0]):
            if rmat.shape[0] != row:
                rank1 = rank1 + 1
                mat[:, 2] = rank1
                rank2 = row
                # 個体群の中から非劣個体を選択
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[0]):
                        if mat[i, 0] < mat[j, 0] and mat[i, 1] < mat[j, 1]:
                            mat[j, 2] = rank2
                        else:
                            pass
                # 非劣個体を新しい配列rmatに記録
                rmat = np.append(rmat, mat[np.any(mat==rank1, axis=1)], axis=0)
                # 非劣個体以外を残す
                mat = mat[~np.any(mat==rank1, axis=1)]

        #rmat = np.reshape(rmat, [-1, col])
        mat = rmat[np.argsort(rmat[:, 2])]

        # rankの最大値を取得
        max_rank = int(np.max(mat[:, 2]))

        # 各目的関数の最大・最小値を取得
        max1 = np.max(mat[:, 0])
        max2 = np.max(mat[:, 1])
        min1 = np.min(mat[:, 0])
        min2 = np.min(mat[:, 1])

        nrank = np.array([], dtype = np.int64)
        sort_mat = np.array([], dtype = np.float64)

        # rank毎に混雑度を計算
        for i in range(1, max_rank+1):
            # 各rankの個体数を取得
            count = np.count_nonzero(mat == i, axis=0)[2]

            # rank毎に混雑度を計算
            if count != 0:
                # 同一rankの行を抽出
                mat_temp = mat[np.any(mat==i, axis=1)]
                # 同一rank内で1個めの目的関数でソート
                mat1 = mat_temp[np.argsort(mat_temp[:, 0])]

                # 混雑度距離を計算
                if count >= 3:
                    for j in range(count):
                        # 境界個体に対して最大距離を与える
                        if j == 0 or j == count - 1:
                            mat1[j, 3] = 10**10
                        # 境界個体以外に対して混雑度距離を計算する
                        else:
                            mat1[j, 3] = (mat1[j+1, 0] - mat1[j-1, 0])/(max1 - min1)
                else:
                    for j in range(count):
                        mat1[j, 3] = 10**10

                # 同一rank内で2個めの目的関数でソート
                mat2 = mat1[np.argsort(mat1[:, 1])]

                # 混雑度距離を計算
                if count >= 3:
                    for j in range(count):
                        if j == 0 or j == count - 1:
                            mat2[j, 3] = mat2[j, 3] + 10**10
                        else:
                            mat2[j, 3] = mat2[j, 3] + (mat2[j+1, 1] - mat2[j-1, 1])/(max2 - min2)
                else:
                    for j in range(count):
                        mat2[j, 3] = mat2[j, 3] + 10**10

                # 同一rank内で混雑度距離で降順にソート
                mat2 = mat2[np.argsort(mat2[:, 3])[::-1]]

                sort_mat = np.append(sort_mat, mat2)

        # 1次元配列を2次元配列に変換
        return sort_mat.reshape([row, -1])
