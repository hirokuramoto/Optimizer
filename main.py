# !/usr/bin/env python3

import sys
import os
import subprocess
import numpy as np
from distutils import dir_util
from optimize import Optimize
from plot2d import Plot2d
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt


class Dialog(QDialog):
    def __init__(self):
        super(Dialog, self).__init__()

        # ウインドウタイトル
        self.setWindowTitle("Optimizer")
        # ウインドウサイズ
        #self.resize(500, 400)
        # ウインドウを画面中央に表示
        self.centerOnScreen()
        self.createWorkDirBox()
        self.createParamfileBox()
        self.createOptimizeBox()
        self.createHyperBox()
        self.createVisualizeBox()

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.WorkDirBox, 0, 0, 1, 2)
        mainLayout.addWidget(self.ParamfileBox, 1, 0, 1, 2)
        mainLayout.addWidget(self.OptimizeBox, 2, 0)
        mainLayout.addWidget(self.HyperBox, 2, 1)
        mainLayout.addWidget(self.VisualizeBox, 3, 0, 1, 2)

        self.setLayout(mainLayout)
        self.show()

    def centerOnScreen(self):
        '''GUIを画面中央に表示させる
        '''
        res = QDesktopWidget().screenGeometry()
        self.move((res.width()/2) - (self.frameSize().width()/2),
                  (res.height()/2) - (self.frameSize().height()/2))


    def createWorkDirBox(self):
        '''作業用ディレクトリの設定グループ
        '''
        self.WorkDirBox = QGroupBox("作業用ディレクトリの設定")
        self.edit1 = QLineEdit()
        layout = QHBoxLayout()

        # ボタンの設定
        button1 = QPushButton("参照")
        # ボタンを押したときに実行する関数をconnectでつなぐ
        button1.clicked.connect(self.open1)
        # Box内のレイアウト
        layout.addWidget(self.edit1)
        layout.addWidget(button1)
        self.WorkDirBox.setLayout(layout)


    def open1(self):
        '''参照ボタンを押したときの動作
        '''
        # ディレクトリの選択ダイアログ
        dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.edit1.setText(dir)

        if os.path.exists(os.path.join(dir, "hoge")): # os.path.joinでパス名をつなげる
            pass
        else:
            dir_util.copy_tree("/home/kuramoto/Work/5_pyQt/hoge", os.path.join(path, "hoge"))
            print("必要なファイルをコピーしました")


    def createParamfileBox(self):
        '''結果ファイルの読み込みグループ
        '''
        self.ParamfileBox = QGroupBox("結果ファイルの読み込み")
        self.edit2 = QLineEdit()
        layout = QGridLayout()

        # ボタンの設定
        button2 = QPushButton("参照")

        # ボタンを押したときに実行する関数をconnectでつなぐ
        button2.clicked.connect(self.open2)

        # Box内のレイアウト
        layout.addWidget(self.edit2, 0, 0)
        layout.addWidget(button2, 0, 1)
        self.ParamfileBox.setLayout(layout)


    def open2(self):
        '''参照ボタンを押したときの動作
        '''
        filename = QFileDialog.getOpenFileName(self, "Open File", None, "csv Files (*.csv)")
        text = filename[0]
        self.edit2.setText(text)


    def createOptimizeBox(self):
        '''計算パラメータの設定グループ
        '''
        self.OptimizeBox = QGroupBox("計算条件")
        self.edit3 = QLineEdit()
        self.edit4 = QLineEdit()
        self.edit3.setValidator(QIntValidator())
        self.edit4.setValidator(QIntValidator())
        self.edit3.setMaxLength(2) # 最大2文字
        self.edit3.setFixedWidth(50) # 50ピクセル幅
        self.edit4.setMaxLength(1) # 最大1文字
        self.edit4.setFixedWidth(50) # 50ピクセル幅
        self.edit4.setText("2")
        label3 = QLabel("設計変数の数")
        label4 = QLabel("目的関数の数")
        button3 = QPushButton("ハイパーパラメータの計算")
        layout = QGridLayout()

        # ボタンを押したときに実行する関数をconnectでつなぐ
        button3.clicked.connect(self.button3)

        # Box内のレイアウト
        layout.addWidget(label3, 0, 0)
        layout.addWidget(self.edit3, 0, 1)
        layout.addWidget(label4, 1, 0)
        layout.addWidget(self.edit4, 1, 1)
        layout.addWidget(button3, 2, 0, 1, 2)
        self.OptimizeBox.setLayout(layout)

    def button3(self):
        '''実行ボタンを押したときの動作
        '''
        ndesign = int(self.edit3.text())
        nobject = int(self.edit4.text())

        # 計算実行ボタン カーネルリッジ回帰のハイパーパラメータを計算
        workdir = self.edit1.text()    #　作業ディレクトリのパス
        resultfile = self.edit2.text() #　結果ファイルのパス
        opt = Optimize(workdir, resultfile, ndesign, nobject)
        hyper_set = opt.hyperparameter()
        self.edit5.setText(str(hyper_set[0,0])) # β1
        self.edit6.setText(str(hyper_set[1,0])) # β2
        self.edit7.setText(str(hyper_set[0,1])) # λ1
        self.edit8.setText(str(hyper_set[1,1])) # λ2
        print("計算完了")


    def createHyperBox(self):
        '''計算したハイパーパラメータを格納
        '''
        self.HyperBox = QGroupBox("ハイパーパラメータ")
        self.edit5 = QLineEdit() # β1
        self.edit6 = QLineEdit() # β2
        self.edit7 = QLineEdit() # λ1
        self.edit8 = QLineEdit() # λ2
        self.edit5.setValidator(QDoubleValidator())
        self.edit6.setValidator(QDoubleValidator())
        self.edit7.setValidator(QDoubleValidator())
        self.edit8.setValidator(QDoubleValidator())
        self.edit5.setFixedWidth(100)
        self.edit6.setFixedWidth(100)
        self.edit7.setFixedWidth(100)
        self.edit8.setFixedWidth(100)
        label5 = QLabel("β")
        label6 = QLabel("λ")
        label7 = QLabel("目的1")
        label8 = QLabel("目的2")
        # 文字列の位置をラベル中央に固定
        label5.setAlignment(Qt.AlignCenter)
        label6.setAlignment(Qt.AlignCenter)
        label7.setAlignment(Qt.AlignCenter)
        label8.setAlignment(Qt.AlignCenter)
        layout = QGridLayout()

        layout.addWidget(self.edit5, 1, 1)
        layout.addWidget(self.edit6, 1, 2)
        layout.addWidget(self.edit7, 2, 1)
        layout.addWidget(self.edit8, 2, 2)
        layout.addWidget(label5, 1, 0)
        layout.addWidget(label6, 2, 0)
        layout.addWidget(label7, 0, 1)
        layout.addWidget(label8, 0, 2)
        self.HyperBox.setLayout(layout)


    def createVisualizeBox(self):
        '''計算実行と可視化のグループ
        '''
        self.VisualizeBox = QGroupBox("計算")
        button4 = QPushButton("計算実行")
        button5 = QPushButton("解範囲プロット")
        button6 = QPushButton("結果ファイル")
        button7 = QPushButton("パレート解")
        layout = QGridLayout()

        # ボタンを押したときに実行する関数をconnectでつなぐ
        button4.clicked.connect(self.button4)
        button5.clicked.connect(self.button5)
        button6.clicked.connect(self.button6)
        button7.clicked.connect(self.button7)

        layout.addWidget(button4, 0, 0)
        layout.addWidget(button5, 0, 1)
        layout.addWidget(button6, 0, 2)
        layout.addWidget(button7, 0, 3)

        self.VisualizeBox.setLayout(layout)


    def button4(self):
        '''結果ファイルボタンを押したときの動作
        '''
        ndesign = int(self.edit3.text())
        nobject = int(self.edit4.text())
        # 計算実行ボタン
        workdir = self.edit1.text()    #　作業ディレクトリのパス
        resultfile = self.edit2.text() #　結果ファイルのパス
        opt = Optimize(workdir, resultfile, ndesign, nobject)
        hyper_set = np.zeros((2, 2))
        hyper_set[0, 0] = self.edit5.text() # β1
        hyper_set[1, 0] = self.edit6.text() # β2
        hyper_set[0, 1] = self.edit7.text() # λ1
        hyper_set[1, 1] = self.edit8.text() # λ2
        predict_value = opt.optimize(hyper_set)
        print("計算完了")


    def button5(self):
        '''解範囲プロットボタンを押したときの動作
        '''
        ndesign = int(self.edit3.text())
        nobject = int(self.edit4.text())
        # 計算実行ボタン
        workdir = self.edit1.text()    #　作業ディレクトリのパス
        resultfile = self.edit2.text() #　結果ファイルのパス
        opt = Optimize(workdir, resultfile, ndesign, nobject)
        hyper_set = np.zeros((2, 2))
        hyper_set[0, 0] = self.edit5.text() # β1
        hyper_set[1, 0] = self.edit6.text() # β2
        hyper_set[0, 1] = self.edit7.text() # λ1
        hyper_set[1, 1] = self.edit8.text() # λ2
        opt.areaplot(hyper_set)
        filepath = workdir + "/areaplot.csv"
        name = "areaplot"
        header = None
        Plot2d(workdir, filepath, 0, nobject, name).plot(header)


    def button6(self):
        '''結果ファイルボタンを押したときの動作
        '''
        workdir  = self.edit1.text()
        filepath = self.edit2.text() #　取得済みのファイルパス
        name = "result"
        ndesign = int(self.edit3.text())
        nobject = int(self.edit4.text())
        header = 0
        Plot2d(workdir, filepath, ndesign, nobject, name).plot(header)


    def button7(self):
        '''グラフ化ボタンを押したときの動作
        '''
        workdir = self.edit1.text()
        filepath = self.edit1.text() + "/best_pop.csv"
        name = "pareto_set"
        ndesign = 0
        nobject = int(self.edit4.text())
        header = None
        Plot2d(workdir, filepath, ndesign, nobject, name).plot(header)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    win = Dialog()
    sys.exit(app.exec_())
