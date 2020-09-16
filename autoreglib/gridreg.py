"""
grid and hyper parameter auto search program
including the residual plot, feaure importance plot
Linerregression Lasso Decisiontree, Randomforest, Gradientboosting


"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')
import scipy as sp

from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from. import eval_plot

def grid_liner(X_train, y_train, X_test, y_test, X_name_list):
    """
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list: namelist
    :return:　reg
    """
    Xl = list(range(0, len(X_name_list)))
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    
    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_mat_plot(X_train, y_train, X_test, y_test, reg, X_name_list)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_plot(X_train, y_train, X_test, y_test, reg, X_name_list)
    
    return reg

def grid_lasso(X_train, y_train, X_test, y_test, X_name_list,params=None,cv=5):
    '''
    GridSearch lasso
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :param params: default =None ={'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    :return: reg
    '''
    Xl = list(range(0, len(X_name_list)))
    # 正則化パラメータ
    if params == None:
        params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    # Crossvaridationを5回、すべてのコアを使う
    grid = GridSearchCV(Lasso(max_iter=100000), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # score
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    print("Best paramator: {}".format(grid.best_params_))
    print()
    
    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_mat_plot(X_train, y_train, X_test, y_test, reg, X_name_list)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_plot(X_train, y_train, X_test, y_test, reg, X_name_list)

    return reg

def grid_decisiontree(X_train, y_train, X_test, y_test, X_name_list,params=None,cv=5):
    '''

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :param params: default =None = {'max_depth': [2, 3, 4, 5]}
    :return: reg
    '''
    Xl = list(range(0, len(X_name_list)))

    # 正則化パラメータ
    if params == None:
        params = {'max_depth': [2, 3, 4, 5]}

    # Cross varidationを5回、すべてのコアを使う
    grid = GridSearchCV(DecisionTreeRegressor(), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    print("Best paramator: {}".format(grid.best_params_))
    print()
    
    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_mat_plot(X_train, y_train, X_test, y_test, reg, X_name_list)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_plot(X_train, y_train, X_test, y_test, reg, X_name_list)

    return reg


def grid_randomforest(X_train, y_train, X_test, y_test, X_name_list, params=None, cv=5):
    '''
    GridSearch RandomForest
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :param params: default =None = {'n_estimators': [5, 10, 50, 100, 200]}
    CVを行わない場合　default＝10
    estimatorsは大きければ大きい程よいが、より多くの決定木の平均値をとり過剰適合が低減される。
    訓練時間が多くかかることが問題。
    主なハイパーパラメータは、 n_estimators と max_features 
    前者は木の数で大きい方が良いが、計算に時間がかかる。重要な数のツリーを超えても結果が大幅に改善されないことがある。
    後者は、ノードを分割するときに考慮すべき特徴量のランダムなサブセットのサイズ。
    低いほど、分散の減少は大きくなるが、バイアスの増加も大きくなる。
    経験的に良いデフォルト値は、回帰問題の場合は max_features = n_features （ n_features はデータ内の特徴量の数）
    max_depth = None を min_samples_split = 1（すなわち、ツリーを完全に展開するとき）と組み合わせて設定すると、良い結果が得られることが多い
    しかし、これらの値は通常最適ではなく、多くのRAMを消費するモデルになる可能性があることに注意。
    
    :return:
    '''
    Xl = list(range(0, len(X_name_list)))

    if params == None:
        params = {'n_estimators': [5, 10, 50, 100, 200]}

    grid = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    print("Best paramator: {}".format(grid.best_params_))
    print()
    
    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_mat_plot(X_train, y_train, X_test, y_test, reg, X_name_list)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_plot(X_train, y_train, X_test, y_test, reg, X_name_list)

    return reg

def grid_gradientboosting(X_train, y_train, X_test, y_test, X_name_list, params=None, cv=5):
    '''
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :param params: default =None
    ={'max_depth': [1, 2, 3, 4, 5], 'n_estimators': [5, 10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

    :return:
    '''
    Xl = list(range(0, len(X_name_list)))
    if params == None:
        params = {'max_depth': [1, 2, 3, 4, 5], 'n_estimators': [5, 10, 50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}

    grid = GridSearchCV(GradientBoostingRegressor(), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    print("Best paramator: {}".format(grid.best_params_))
    print()
    
    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_mat_plot(X_train, y_train, X_test, y_test, reg, X_name_list)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_plot(X_train, y_train, X_test, y_test, reg, X_name_list)


    return reg


def grid_svm(X_train, y_train, X_test, y_test, X_name_list, params=None, cv=5):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param Xl:
    :param X_name_list:
    :return:
    """
    Xl = list(range(0, len(X_name_list)))
    if params == None:
        params = {'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]}

    grid = GridSearchCV(SVR(), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    print("Best paramator: {}".format(grid.best_params_))
    print()
    
    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)

    return reg,


def grid_kneighbors(X_train, y_train, X_test, y_test, params=None, cv=5):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    if params == None:
        params = {'n_neighbors': [1, 2, 3, 4, 5]}

    grid = GridSearchCV(KNeighborsRegressor(), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    
    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    
    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    print("Best paramator: {}".format(grid.best_params_))
    print()
    
    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)

    return reg


def simple_randomforest(X_train, y_train, X_test, y_test, X_name_list, params=None, cv=5):
    '''
    GridSearchを行わないRandomForest関数
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :param params: default = None  = {'n_estimators': [10]}
  
    CVを行わない場合　default＝10
    estimatorsは大きければ大きい程よいが、より多くの決定木の平均値をとり過剰適合が低減される。
    訓練時間が多くかかることが問題。
    主なハイパーパラメータは、 n_estimators と max_features
    前者は木の数で大きい方が良いが、計算に時間がかかる。重要な数のツリーを超えても結果が大幅に改善されないことがある。

    後者は、ノードを分割するときに考慮すべき特徴量のランダムなサブセットのサイズ。
    低いほど、分散の減少は大きくなるが、バイアスの増加も大きくなる。
    経験的に良いデフォルト値は、回帰問題の場合は max_features = n_features （ n_features はデータ内の特徴量の数）
    max_depth = None を min_samples_split = 1（すなわち、ツリーを完全に展開するとき）と組み合わせて設定すると、良い結果が得られることが多い
    しかし、 これらの値は通常最適ではなく、多くのRAMを消費するモデルになる可能性があることに注意。

    :return:
    '''
    Xl = list(range(0, len(X_name_list)))

    if params == None:
        params = {'n_estimators': [10]}

    grid = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    reg_name = reg.__class__.__name__
    trainr2 = reg.score(X_train, y_train)
    testr2 = reg.score(X_test, y_test)

    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    print("Best paramator: {}".format(grid.best_params_))
    print()

    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_mat_plot(X_train, y_train, X_test, y_test, reg, X_name_list)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_plot(X_train, y_train, X_test, y_test, reg, X_name_list)

    return reg


def simple_gradientboosting(X_train, y_train, X_test, y_test, X_name_list, params=None, cv=5):
    '''
    GridSearchを行わないGradientboosting関数
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param X_name_list:
    :param params: default = None
    =params = {'max_depth': [3], 'n_estimators': [100], 'learning_rate': [0.1]}

    :return:
    '''
    Xl = list(range(0, len(X_name_list)))
    if params == None:
        params = {'max_depth': [3], 'n_estimators': [100], 'learning_rate': [0.1]}

    grid = GridSearchCV(GradientBoostingRegressor(), param_grid=params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    reg = grid.best_estimator_
    reg.fit(X_train, y_train)
    reg_name = reg.__class__.__name__
    trainr2 = reg.score(X_train, y_train)
    testr2 = reg.score(X_test, y_test)

    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    print("Best paramator: {}".format(grid.best_params_))
    print()

    eval_plot.yyplot_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_mat_plot(X_train, y_train, X_test, y_test, reg, X_name_list)
    eval_plot.residual_plot(X_train, y_train, X_test, y_test, reg)
    eval_plot.feature_plot(X_train, y_train, X_test, y_test, reg, X_name_list)

    return reg
