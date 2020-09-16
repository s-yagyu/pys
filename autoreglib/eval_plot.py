"""
回帰結果を評価プロットするための関数

"""
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns
sns.set_style('whitegrid')
import scipy as sp
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


def residual_plot(X_train, y_train, X_test, y_test, reg):
    """
    残差プロット
    回帰モデルを診断して非線形や外れ値を検出し、
    誤差がランダムに分布しているかどうかをチェックするのに用いる

    予測が完璧である場合、残差はちょうどゼロになる。
    現実のアプリケーションでは残差がゼロになることはない。
    よい回帰モデルは誤差がランダムに分布し、残差が中央の0の直線の周りにランダムに散らばる。
    残差プロットにパターンがみられる場合は、モデルが何かの情報を補足できていないことを意味する。
    さらに残差プロットでは外れ値を検出できる。中央の直線から大きく外れている点が外れ値である。
    横軸に予測値、縦軸に実際の値との差をプロットしたもの
    多くのデータがy=0の直線に近いところに集まれば、よいモデル
    均一に点がプロットされている場合、線形回帰が適切。
    そうでは無い場合は、非線形なモデルを使うことを検討。


    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param reg: regretion instance
    :return:
    """
    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # R2スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {:.3f}".format(trainr2))
    print("R Test Best score : {:.3f}".format(testr2))


    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)

    train = plt.scatter(pred_train, (pred_train - y_train), c='b', alpha=0.5)
    test = plt.scatter(pred_test, (pred_test - y_test), c='r', alpha=0.5)
    plt.legend((train, test), ('Training', 'Test'), loc='upper left')
    plt.title('{}  Residual Plots'.format(reg_name))
    plt.xlabel("Observed data")
    plt.ylabel("Predict-Observed")
    plt.figtext(0.65, 0.2, "{}\nR2 train:{:.2f}\nR2 test:{:.2f}".format(reg_name, trainr2, testr2))
    #plt.figtext(0.65,0.2,"{}\nR2 train:{:.2f}\nR2 test:{:.2f}".format(reg_name,trainr2,testr2),size=15)
    plt.show()

def yyplot_plot(X_train, y_train, X_test, y_test, reg, comment=False):
    """
    Observed-Predicted Plot (yyplot)
    yyplot は、横軸に実測値(yobs)、縦軸に予測値(ypred)をプロットしたもの.
    プロットが対角線付近に多く存在すれば良い予測

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param reg:
    :param comment:
    :return:
    """

    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    rr = rmse / mae

    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {:.3f}".format(trainr2))
    print("R2 Test Best score : {:.3f}".format(testr2))
    print()
    print("Test Root Mean Squared Error (RMSE):", rmse)
    print("Test Mean Absolute Error (MAE):", mae)
    print("Test RMSE/MAE:", rr)
    print()
    if comment == True:
        if rr < 1.2532:
            print("RMSE/MAEが1.2533（最適モデル）より小さい")
            print("各サンプルについて同じような大きさの誤差が生じている可能性あり。")
            print("予測にバイアスを加えて、ハイパーパラメータの変更をしてみる")
        elif rr >= 1.2532 and  rr <= 1.2534:
            print("誤差が正規分布に従う場合、適切なモデル構築ができている可能性が高い。")
            print(":誤差の絶対値も必ずチェックすること！")
        else: #rr>1.2534
            print("RMSE/MAEが1.2533（最適モデル）より大きい")
            print("予測を大きく外しているデータが存在する可能性がある。=1.414ならラプラス分布誤差の可能性あり。")
            print("外れ値だと思われるデータを消去する。ハイパーパラメータを変更してみる。")

    obs = np.concatenate((y_train, y_test), axis=0)
    pre = np.concatenate((pred_train, pred_test), axis=0)
    #om = np.max(obs)
    yyplot_train = plt.scatter(y_train, pred_train, c='b', alpha=0.5)
    yyplot_test = plt.scatter(y_test, pred_test, c='r', alpha=0.5)
    plt.plot(obs, obs, c='g', alpha=0.5)
    plt.legend((yyplot_train, yyplot_test), ('Training', 'Test'), loc='upper left')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('{}  Observed-Predicted Plot'.format(reg_name))
    plt.figtext(0.65,0.2,"{}\nR2 train:{:.2f}\nR2 test:{:.2f}".format(reg_name,trainr2,testr2))

    plt.show()

def yyplot_plot_data(X_train, y_train, X_test, y_test, xdata, rdata, reg):
    """
    yy-plotに測定データをプロットする関数。
    測定点は一つのみ表示。

    # Observed-Predicted Plot (yyplot)
    # yyplot は、横軸に実測値(yobs)、縦軸に予測値(ypred)をプロットしたもの.
    # プロットが対角線付近に多く存在すれば良い予測が行えています。
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param xdata: 測定点の説明変数リストデータ　この値を用いて予測します。
    :param rdata: 実測データ
    :param reg:
    :return:
    """

    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    pred_train = reg.predict(X_train)
    pred_test = reg.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)
    rr = rmse / mae

    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {:.3f}".format(trainr2))
    print("R2 Test Best score : {:.3f}".format(testr2))
    print("R2 ratio Test/Train:{}".format(testr2/trainr2))
    print()
    print("Test Root Mean Squared Error (RMSE):{:.3f}".format(rmse))
    print("Test Mean Absolute Error (MAE):{:.3f}".format(mae))
    print("Test RMSE/MAE(cf.1.253):{:.3f}".format(rr))
    
    #Listdataからarrayにする
    #測定点の説明変数リスト
    x_array=np.array(xdata).reshape(1,-1)
    pred_obs = reg.predict(x_array)
    obs = np.concatenate((y_train, y_test), axis=0)
    pre = np.concatenate((pred_train, pred_test), axis=0)

    print()
    print("Observation Value:",rdata)
    print("Prediction:{:.2f}".format(pred_obs[0]))

    yyplot_train = plt.scatter(y_train, pred_train, c='b', alpha=0.5,label='Training')
    yyplot_test = plt.scatter(y_test, pred_test, c='r', alpha=0.5,label='Test')
    plt.plot(obs, obs, c='y', alpha=0.5)
    plt.plot(rdata,pred_obs,'*',c='g',markersize=15,label='Observation')
    plt.legend()
    #plt.legend((yyplot_train, yyplot_test,data_plot), ('Training', 'Test','Observation'), loc='upper left')
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('{}  Observed-Predicted Plot'.format(reg_name))
    plt.figtext(0.65,0.2,"{}\nR2 train:{:.2f}\nR2 test:{:.2f}\nPrediction:{:.2f}".format(reg_name,trainr2,testr2,pred_obs[0]))
    plt.show()
    
    
def feature_mat_plot(X_train, y_train, X_test, y_test, reg, X_name_list,colorbar="False"):
    """
    特徴量の寄与度をmat_plot形式で表示します。
    LinerRegression, Lasso,Decision tree, Random forest, Gradient boosting
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param reg:
    :param X_name_list:
    :param colorbar:
    :return:
    """

    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {:.3f}".format(trainr2))
    print("R2 Test Best score : {:.3f}".format(testr2))
    
    Xl = list(range(0, len(X_name_list)))
    
    if reg_name == "LinearRegression" or reg_name =='Lasso':
        feat=np.abs(reg.coef_)
        title_reg_name = "Linear Regression"
    elif reg_name =='Lasso':
        feat = np.abs(reg.coef_)
        title_reg_name = "Lasso Regression"

    elif reg_name == 'DecisionTreeRegressor':
        feat=reg.feature_importances_
        title_reg_name = 'Decision Tree Regressor'

    elif reg_name =='RandomForestRegressor':
        feat=reg.feature_importances_
        title_reg_name = 'Random Forest Regressor'

    elif reg_name == 'GradientBoostingRegressor':
        feat=reg.feature_importances_
        title_reg_name = 'Gradient Boosting Regressor'

    else:
        pass
    plt.matshow(feat.reshape(1, -1), cmap="Blues")
    plt.xticks(Xl, X_name_list, rotation=90,verticalalignment='bottom' )
    plt.title("{}  $R^2$ train:{:.2f}  $R^2$ test:{:.2f}".format(title_reg_name,trainr2,testr2),size=15,y=-1)
    
    plt.yticks(())

    if colorbar == "True" or colorbar =="T":
        plt.colorbar()
        plt.figtext(0.05,0.3,"{}\n$R^2$ train:{:.2f}\n$R^2$ test:{:.2f}".format(reg_name,trainr2,testr2),size=15)
    
    plt.show()

    
    plt.matshow(feat.reshape(1, -1), cmap="YlGnBu")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.yticks(())
    plt.show()

def feature_plot(X_train, y_train, X_test, y_test, reg, X_name_list):
    """
    特徴量の寄与度をplot形式で表示します。
    LinerRegression, Lasso,Decision tree, Random forest, Gradient boosting

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param reg:
    :param X_name_list:
    :return:
    """

    reg_name=reg.__class__.__name__
    trainr2=reg.score(X_train, y_train)
    testr2=reg.score(X_test, y_test)
    
    # スコアー
    print("{}".format(reg_name))
    print("R2 Training Best score : {}".format(trainr2))
    print("R2 Test Best score : {}".format(testr2))
    
    Xl = list(range(0, len(X_name_list)))
    feat_sum=len(X_name_list)
    #Decision tree, Random forest, Gradient boostingの(Feature importance)を持つもの場合分け
    if reg_name == "LinearRegression" or reg_name =='Lasso':
        feat=np.abs(reg.coef_)
        feat_count=np.sum(feat != 0)
    elif reg_name == 'DecisionTreeRegressor' or reg_name =='RandomForestRegressor' or reg_name == 'GradientBoostingRegressor':
        feat=reg.feature_importances_
        feat_count=np.sum(feat != 0)
    else:
        pass

    
    # 使用された特徴量
    print("Number of features: {}".format(feat_sum))
    print("Number of features used: {}".format(feat_count))
    plt.plot(feat, 's')
    plt.title('{}  Features Plot'.format(reg_name))
    plt.xlabel("feature index")
    plt.ylabel("importance")
    plt.xticks(Xl, X_name_list, rotation=90)
    plt.figtext(0.7,0.2,"{}\nR2 train:{:.2f}\nR2 test:{:.2f}".format(reg_name,trainr2,testr2))
        
    plt.show()
        