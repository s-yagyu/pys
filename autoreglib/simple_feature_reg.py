"""
単変量特徴選択
個々の特徴量とターゲットとの間に統計的に顕著な関係がるかどうかを計算する。
最も高い確信度で関連している特徴量が選択される。
この方法は、計算が高速でモデルを構築する必要がないが、
個々の特徴量を個別に考慮するために他の特徴量と組み合わさって意味のある特徴量は捨てられる。
特徴選択後に使われるモデルとは完全に独立である。

selectPercentile：全部の特徴量に対する上位の割合を指定
selectKBest：使用される上位の特徴量
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import numpy as np
import scipy as sp

from sklearn.feature_selection import SelectPercentile, SelectKBest, f_regression

def spt(X_train, y_train, X_test, y_test, X_name_list, percentile=20):
    """
    単変量特徴選択
    個々の特徴量とターゲットとの間に統計的に顕著な関係がるかどうかを計算する。
    selectPercentile：全部の特徴量に対する上位の割合を指定
    selectPercentile  20%
    
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list: feature list
            ex) X_name_list=list(dfR2.drop('Corrosion',axis=1).columns)
    :param percentile:
    :return: X_train_selected, X_test_selected
    """
    Xl = list(range(0, len(X_name_list)))
    select=SelectPercentile(score_func=f_regression,percentile=percentile)
    select.fit(X_train, y_train)
    X_train_selected=select.transform(X_train)
    X_test_selected=select.transform((X_test))
    print("SelectPercentile {} %".format(percentile))
    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_selected.shape:{}".format(X_train_selected.shape))

    mask =select.get_support()
    #plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.matshow(mask.reshape(1, -1), cmap="YlGnBu")
    plt.xlabel("SelectPercentile {} %".format(percentile))
    plt.yticks(())
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()

    # Array listで受け取る
    select_name_index, = np.where(mask == True)

    X_name_selected = []
    for li in  select_name_index:
        new_X = X_name_list[li]
        X_name_selected.append(new_X)

    print(X_name_selected)
    return  X_train_selected, X_test_selected, X_name_selected


def skb(X_train, y_train, X_test, y_test, X_name_list, k=5):
    """
    単変量特徴選択
    個々の特徴量とターゲットとの間に統計的に顕著な関係がるかどうかを計算する。
    selectKBest
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :param k:
    :return: X_train_selected, X_test_selected
    """
    Xl = list(range(0, len(X_name_list)))
    select = SelectKBest(score_func=f_regression,k=k)
    select.fit(X_train, y_train)
    X_train_selected = select.transform(X_train)
    X_test_selected = select.transform((X_test))
    print("SelectKBest {}".format(k))
    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_selected.shape:{}".format(X_train_selected.shape))

    mask = select.get_support()
    plt.matshow(mask.reshape(1, -1), cmap="YlGnBu")
    plt.xlabel("SelectKBest {}".format(k))
    plt.yticks(())
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.show()
    # Array listで受け取る
    select_name_index, = np.where(mask == True)

    X_name_selected = []
    for li in select_name_index:
        new_X = X_name_list[li]
        X_name_selected.append(new_X)

    print(X_name_selected)

    return X_train_selected, X_test_selected, X_name_selected
