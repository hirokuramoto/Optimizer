import ctypes
import numpy as np

class CallFortran(object):
    """Fortranのサブルーチンを呼び出す
    """
    def __init__(self, data_size, n_param, beta, matrix, gram_matrix):
        self._data_size = data_size     # テストデータ数
        self._n_param = n_param         # パラメータ数
        self._beta = beta               # β
        self._matrix = matrix           # テストデータの設計変数配列
        self._gram_matrix = gram_matrix # グラム行列

    def call_fortran(self):
        #f = np.ctypeslib.load_library("libfort.so", ".")   # Linux, MacOSの場合
        f = np.ctypeslib.load_library("libfort.dll", "./RealCodedGA")   # Windowsの場合

        # f.{関数名}_.argtypes で、引数の型指定
        f.test_.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            np.ctypeslib.ndpointer(dtype = np.float64),
            np.ctypeslib.ndpointer(dtype = np.float64)
            ]

        # f.{関数名}_.restype で、関数の戻り値の型指定
        f.test_.restype = ctypes.c_void_p

        # 定数をポインタとして渡す
        fn = ctypes.byref(ctypes.c_int32(self._data_size))
        fp = ctypes.byref(ctypes.c_int32(self._n_param))
        fb = ctypes.byref(ctypes.c_double(self._beta))
        f.test_(fn, fp, fb, self._matrix, self._gram_matrix)

if __name__ == '__main__':
    # python側から与えられた２次元配列は転置されてしまうので注意が必要
    # pythonは行-major、fortranは列-major

    # テストデータを取得
    from standard_data import StandardData
    test = StandardData(5,2)
    data = test.standard("result.csv")

    design = np.array(data[0:, 0:5])
    data_size = design.shape[0]
    n_param = design.shape[1]
    design = design.T

    beta = 0.1
    gram_matrix = np.identity(data_size)


    test = CallFortran(data_size, n_param, beta, design, gram_matrix)
    hoge = test.call_fortran()
    print(gram_matrix)
