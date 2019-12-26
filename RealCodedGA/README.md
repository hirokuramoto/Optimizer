# 実数値GAを用いたハイパーパラメータ値予測

## 1. 環境
- python3
  - numpy
  - matplotlib
  - pandas

- 対象OS
  - Windows 10
  - Mac OS
  - Linux

## 2. 準備するもの
### 2-1．共通
  - CAE結果のcsvファイルを`result.csv`という名称で同じフォルダに配置しておく．（csvファイルは，1行目：ヘッダー行，2行目以降：設計変数と結果）

### 2-2．Windowsの場合
  - `call_fortran.py`内で使用する共有ライブラリを`libfort.dll`に指定する．

### 2-3．Linux，MacOSの場合
  - `call_fortran.py`内で使用する共有ライブラリを`libfort.so`に指定する．**(# を消して有効化)**


```python
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
      f = np.ctypeslib.load_library("libfort.dll", ".")   # Windowsの場合
```

## 3．使用方法
### 3-1．ファイル設定
  - `main.py`内の **GAパラメータ** ，および **訓練用データ** の値を入力する．
  - GAパラメータでは，`size`と`generation_loop`のみ変更すれば良い．
  - 訓練用データでは，訓練データのcsvファイルに合わせて，**設計変数，目的関数の数を変更** する．
  - 訓練用データの，何個目の目的関数についてハイパーパラメータを求めるかを，`num`パラメータで指定する．**（0-originになるので注意）**

```python
def main():
    # GAパラメータ
    maximum = 1             # 遺伝子の値の最大値
    minimum = 0             # 遺伝子の値の最小値
    dimension = 2           # パラメータ数
    size = 40               # 個体数
    generation_loop = 500   # 繰り返し数

    # 訓練用データ
    design = 5              # 設計変数の数
    object = 2              # 目的関数の数
    num = 0                 # 何番目の目的関数について調べるか
    filename = "result.csv" # 訓練データのcsvファイル
```

### 3-2．計算実行
 - コンソールで`$ python3 main.py` で計算開始．

```
499 0.20388500699305018 [9.76647480e-03 2.50217329e-07]     # 計算回数　CV値　[ β  λ ]の並
35.36523389816284                                           # 計算時間
```
