"""
主成分分析のPlot
PairPlot

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')
import scipy as sp
from scipy import stats

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from. import eval_plot

def pca_plot(Xs, X_name_list, n_components=2, explanation=True ):
    """
    主成分分析
    データの分散が大きな次元ほど、より多くの情報を含んでいると仮定。
    分散は、データのバラつきの大きさを表す統計量。
    分散が小さいと、どの値も似たり寄ったりで差異を見出すのが難しい。
    分散が大きければ値ごとの違いも見つけやすくなる。

    評価項目
    寄与率
    各主成分の重要性を表す
    全ての主成分の寄与率を足し合わせると1.0になる

    累積寄与率
    第１主成分から第m主成分までの寄与率の和
    第１主成分から第m主成分での圧縮がデータの散らばり具合をどの程度カバーしているかの説明する割合

    因子負荷量
    各変数の各主成分への影響力を見つけて、各主成分の意味を推定
    因子負荷量が１か-1に近い因子ほど、主成分に強く寄与している

    Standerd scalerでノーマライズしている
    :param Xs: Xdata
    :param X_name_list:
    :param n_components:
    :return:
    """

    scaler = StandardScaler()
    scaler.fit(Xs)
    X_scaled = scaler.transform(Xs)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    print("Original shape:{}".format(str(X_scaled.shape)))
    print("Reduced shape:{}".format(str(X_pca.shape)))
    # 主成分の寄与率を出力する
    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
    print("固有ベクトル:{0}".format(pca.components_))
    if explanation == True:
        print("寄与率:各主成分の重要性を表す.全ての主成分の寄与率を足し合わせると1.0になる")
        print("累積寄与率:第１主成分から第m主成分までの寄与率の和.")
        print("因子負荷量:因子負荷量が１か-1に近い因子ほど、主成分に強く寄与している")

    print("Principal components(因子負荷量)")
    plt.matshow(pca.components_, cmap='viridis')
    #plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(X_name_list)), X_name_list, rotation=90, ha='left')
    plt.xlabel("Feature")
    plt.show()
    #plt.ylabel("Principal components")
    plt.matshow(pca.components_, cmap='viridis')
    plt.xticks(range(len(X_name_list)), X_name_list, rotation=90)
    plt.show()

    if n_components == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.show()



def corrfunc(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    p_stars = ''
    if abs(p) <= 0.05:
        p_stars = '*'
    if abs(p) <= 0.01:
        p_stars = '**'
    if abs(p) <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    # ax.annotate('r = {:.2f} '.format(r) + p_stars,xy=(0.05, 0.9), xycoords=ax.transAxes)
    ax.set_title('r = {:.2f} '.format(r) + p_stars)


def annotate_colname(x, **kws):
    ax = plt.gca()
    # ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,fontweight='bold')
    ax.set_title(x.name)


def cor_matrix(df):
    """
    pair plot
	右上段に相関積率Plotが表示される
	
    ex)
    cor_matrix(dfRpf)
    cor_matrix(dfRpf.drop([ 'Rain', 'AW', 'MW', 'PMW', 'Hum', 'Lhum', 'Place'],axis=1))
    
    cf)#普通のPairPlot(seabornを利用）
    sns.pairplot(dfRpf.drop([ 'Rain', 'AW', 'MW', 'PMW', 'Hum', 'Lhum'],axis=1),hue='Place')

    :param df:
    :return:
    """
    # g = sns.PairGrid(df, palette=['red'])
    g = sns.PairGrid(df, palette=['red'])
    # Use normal regplot as `lowess=True` doesn't provide CIs.
    g.map_diag(plt.hist)
    g.map_diag(annotate_colname)
    # g.map_upper(sns.kdeplot, cmap='Blues_d')
    g.map_upper(sns.kdeplot)
    g.map_upper(corrfunc)
    g.map_lower(plt.scatter)
    # g.map_lower(plt.scatter, cmap='Blues_d')
    # Remove axis labels, as they're in the diagonals.
    """
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
    """
    return g