"""
モデルベース特徴量選択
教師あり学習モデルを用いて個々の特徴量の重要度を判断して、重要なものだけを残す方法。
特徴選択に使うモデルは最終的に使うモデルと同じでなくてもよい。

SelectFromModelは、フィッティング後に coef_ またはfeature_importances_属性を持つ推定器と一緒に使用できるメタ変換器。
coef_またはfeature_importances_の対応する値が、
指定されたしきい値パラメータを下回る場合、特徴は重要ではないとみなされ、削除されます。
しきい値を数値で指定する以外にも、文字列引数を使用してしきい値を見つけるためのヒューリスティックスが組み込まれています。
利用可能なヒューリスティックスは、「平均」、「中央値」、およびこれらの浮動小数点数の「0.1 *平均」。

反復特徴量選択
すべての特徴量を利用してモデルを作り、そのモデルで最も重要度が低い特徴量を削除する。
そしてまたモデルを作り、事前に定めた数の特徴量まで繰り返す。
このために、計算量ははるかに多い。

Randomforest, Gradientboosting, SVM
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')
import scipy as sp

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def randomforestreg(X_train, y_train, X_test, y_test, X_name_list,threshold="median"):
    '''
    randomforestで特徴量選択を行う。
    その後Linierregressionで識別機を作成
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:feature name
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))

    # 特徴量選択
    sfm = SelectFromModel(RandomForestRegressor(), threshold=threshold)

    sfm.fit(X_train, y_train)
    X_train_sfm = sfm.transform(X_train)
    X_test_sfm = sfm.transform(X_test)
    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_sfm.shape: {}".format(X_train_sfm.shape))
    reg2 = RandomForestRegressor()
    reg2.fit(X_train_sfm, y_train)
    print("Select From Model")
    print("Training Best score : {:.3f}".format(reg2.score(X_train_sfm, y_train)))
    print("Test Best score : {:.3f}".format(reg2.score(X_test_sfm, y_test)))

    mask_sfm = sfm.get_support()
    print("RandomForestRegression")
    # visualize the mask_sfm. black is True, white is False
    plt.matshow(mask_sfm.reshape(1, -1), cmap='gray_r')

    plt.yticks(())
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # RFE
    rfe = RFE(RandomForestRegressor())

    rfe.fit(X_train, y_train)
    # visualize the selected features:
    mask_rfe = rfe.get_support()
    print("RandomForest")
    plt.matshow(mask_rfe.reshape(1, -1), cmap='gray_r')
    plt.yticks(())
    plt.xticks(Xl, X_name_list, rotation=90)

    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    print("LinearRegression")
    reg3 = LinearRegression()
    reg3.fit(X_train_rfe, y_train)
    print("Training Best score : {:.3f}".format(reg3.score(X_train_rfe, y_train)))
    print("Test Best score : {:.3f}".format(reg3.score(X_test_rfe, y_test)))

def gradientboostingreg(X_train, y_train, X_test, y_test, X_name_list,threshold="median"):
    '''

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param Xl: feature number
    :param X_name_list:feature name
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))

    # 特徴量選択
    sfm = SelectFromModel(GradientBoostingRegressor(), threshold=threshold)

    sfm.fit(X_train, y_train)
    X_train_sfm = sfm.transform(X_train)
    X_test_sfm = sfm.transform(X_test)
    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_sfm.shape: {}".format(X_train_sfm.shape))

    reg2 = GradientBoostingRegressor()
    reg2.fit(X_train_sfm, y_train)
    print("SFM")
    print ("Training Best score : {:.3f}".format(reg2.score(X_train_sfm, y_train)))
    print ("Test Best score : {:.3f}".format(reg2.score(X_test_sfm, y_test)))

    mask_sfm = sfm.get_support()
    print("GradientBoosting")
    # visualize the mask_sfm. black is True, white is False
    plt.matshow(mask_sfm.reshape(1, -1), cmap='gray_r')

    plt.yticks(())
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # RFE
    rfe = RFE(GradientBoostingRegressor())

    rfe.fit(X_train, y_train)
    # visualize the selected features:
    mask_rfe = rfe.get_support()
    print("GradientBoostingRegressor")
    plt.matshow(mask_rfe.reshape(1, -1), cmap='gray_r')

    plt.yticks(())
    plt.xticks(Xl, X_name_list, rotation=90)

    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    print("LinearRegression","RFE")
    reg3 = LinearRegression()
    reg3.fit(X_train_rfe, y_train)
    print ("Training Best score : {:.3f}".format(reg3.score(X_train_rfe, y_train)))
    print ("Test Best score : {:.3f}".format(reg3.score(X_test_rfe, y_test)))


def svmreg(X_train, y_train, X_test, y_test, X_name_list,threshold="median"):
    '''

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param Xl: feature number
    :param X_name_list:feature name
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))

    # 特徴量選択
    sfm = SelectFromModel(SVR(), threshold=threshold)

    sfm.fit(X_train, y_train)
    X_train_sfm = sfm.transform(X_train)
    X_test_sfm = sfm.transform(X_test)
    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_sfm.shape: {}".format(X_train_sfm.shape))

    reg2 = SVR()
    reg2.fit(X_train_sfm, y_train)
    print("SFM")
    print("Training Best score : {:.3f}".format(reg2.score(X_train_sfm, y_train)))
    print("Test Best score : {:.3f}".format(reg2.score(X_test_sfm, y_test)))

    mask_sfm = sfm.get_support()
    # visualize the mask_sfm. black is True, white is False
    plt.matshow(mask_sfm.reshape(1, -1), cmap='gray_r')
    plt.xlabel("SVR")
    plt.yticks(())
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # RFE
    rfe = RFE(SVR())

    rfe.fit(X_train, y_train)
    # visualize the selected features:
    mask_rfe = rfe.get_support()
    plt.matshow(mask_rfe.reshape(1, -1), cmap='gray_r')
    plt.yticks(())
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.xlabel("SVR")
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    print("LinearRegression")
    reg3 = LinearRegression()
    reg3.fit(X_train_rfe, y_train)
    print("Training Best score : {:.3f}".format(reg3.score(X_train_rfe, y_train)))
    print("Test Best score : {:.3f}".format(reg3.score(X_test_rfe, y_test)))