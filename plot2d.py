# モジュールのインポート
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D


class Plot2d(object):
    def __init__(self, workdir, filepath, ndesign, nobject, name):
        self.workdir  = workdir
        self.filepath = filepath
        self.ndesign = ndesign
        self.nobject = nobject
        self.name = name

        # フォントの設定
        plt.rcParams['font.family'] = 'serif' # 使用するフォント
        plt.rcParams['font.size'] = 8   # フォントの大きさ
        # 軸の設定
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0 # x軸主目盛り線の幅
        plt.rcParams['ytick.major.width'] = 1.0 # y軸主目盛り線の幅
        plt.rcParams['axes.linewidth'] = 1.0    # 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['grid.linestyle']='--' # グリッド線を破線に
        # 凡例の設定
        plt.rcParams["legend.markerscale"] = 1
        plt.rcParams["legend.fancybox"] = False
        plt.rcParams["legend.framealpha"] = 1
        plt.rcParams["legend.edgecolor"] = 'black'

    def plot(self, header):
        self.header = header
        # CSVからデータ読み込み.1行目は列名として指定
        data = pd.read_csv(self.filepath, header = self.header) # index（＝行名）はheader行の次の行を0として付与してくれる
        obj_data = data.iloc[:, self.ndesign:self.ndesign + self.nobject]

        # 2Dグラフの作成
        fig = plt.figure(figsize=(3.4, 3.4)) # プロットエリアが正方形になるように

        ax = fig.add_subplot(1, 1, 1)

        # 2D散布図の作成
        ax.scatter(obj_data.iloc[:, 0], obj_data.iloc[:, 1], s=10, c='blue', edgecolors='black', linewidths='1', marker='o', alpha = '0.5')

        # ラベルの指定
        ax.set_xlabel(r'Object Function 1')
        ax.set_ylabel(r'Object Function 2')

        # グラフタイトルの設定
        ax.set_title(self.name)

        # 軸目盛りの指数表示指定
        ax.xaxis.set_major_formatter(FixedOrderFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(FixedOrderFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis="both")

        #ax.set_xlim(self.xmin, self.xmax)
        #ax.set_ylim(self.ymin, self.ymax)

        # グリッド
        ax.grid(zorder=0)

        # 凡例の表示
        #ax.legend(loc='upper right') # locで場所の固定

        # グラフの保存
        plt.savefig(self.workdir + '/' + self.name + '.png', format='png', dpi=600, bbox_inches="tight", pad_inches=0.05)

        # グラフの表示
        plt.show()

#クラス設定  ※ScalarFormatterを継承
class FixedOrderFormatter(ScalarFormatter):
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        self.orderOfMagnitude = self._order_of_mag
