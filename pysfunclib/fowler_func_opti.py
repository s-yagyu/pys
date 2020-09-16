"""
Optimaization functions

"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as optimize
import scipy as sp

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from pysfunclib import fowler_func as ff


####　Loss function
def pys_residual_func(parameter, x, y):
    '''
    pys residual function
    '''
    residual = y - ff.pys(x, parameter[0], parameter[1], parameter[2], parameter[3])
    return residual


def spys_residual_func(parameter, x, y):
    '''
    Sqrt PYS residual function
    use for  sp.leastsq

    note:
    result = sp.leastsq(spys_residual_func, ini_para,args=(xdata_,data_),maxfev=100000)

    '''
    residual = y - ff.spys(x, parameter[0], parameter[1], parameter[2], parameter[3])
    return residual


### error function
def rmse_res(y_obs, y_pred):
    residual = y_obs - y_pred
    resid_=np.sqrt(((residual*residual).sum())/len(y_obs))
    return resid_

def mae_res(y_obs, y_pred):
    residual = np.abs(y_obs - y_pred)
    resid_=(residual.sum())/len(y_obs)
    return resid_

# residual
def spys_mae_res(param, x, y):
    fit = ff.spys(x, param[0], param[1], param[2], param[3])
    
    # mae = mean_absolute_error(y, fit)
    mae = mae_res(y,fit)

    return mae


def spys_rmse_res(param, x, y):
    fit = ff.spys(x, param[0], param[1], param[2], param[3])
    
    # rmse = np.sqrt(mean_squared_error(y, fit))
    rmse = rmse_res(y,fit)
    return rmse


def spys_abs_residual_sum(parameter, x, y):
    '''
    spysの絶対値の残差（平均を取っていないので、残差平均ではない）
    absolute error function

    平均を取れば
    spys_abs_residual_sum(parameter,x,y)/len(x)

    '''

    abs_residual = np.abs(spys_residual_func(parameter, x, y)).sum()

    return abs_residual


def spys_sq_residual_sum(parameter, x, y):
    '''
    SPYSの絶対値の2乗の残差
    # squre error function

    平均を取れば
    np.sqrt(spys_sq_residual_sum(para,xx,y)/len(xx))

    '''

    sq_residual = np.square(np.abs(spys_residual_func(parameter, x, y))).sum()

    return sq_residual


#### Evaluation function


def evaluation(y_obs, y_pred):
    """
    y_obs: 測定データ
    y_pred:回帰予測（Fitting）データ

    """
    
    # r2 = r2_score(y_obs, y_pred)
    # rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    # mae = mean_absolute_error(y_obs, y_pred)
    y_obs_ = np.array(y_obs)
    y_pred_ = np.array(y_pred)
    r2, _ = evaluation_determination(y_obs_, y_pred_)
    rmse = rmse_res(y_obs_, y_pred_)
    mae = mae_res(y_obs_, y_pred_)
    ratio = rmse / mae

    return r2, rmse, mae, ratio


def evaluation_determination(y_obs, y_pred):
    """
    決定係数を分解して計算する
    Coefficient of determination

    Parameters
    ----------
    y_obs
    y_pred

    Returns
    -------
    r2c: 決定係数
    evs：

    """
    std = np.std(y_obs)
    var = np.var(y_obs)
    mean = np.mean(y_obs)

    # arrayの個数
    n_ = int(y_obs.shape[0])

    # 全変動
    all_var = np.square(y_obs - mean).sum()

    # 回帰変動
    reg_var = np.square(y_pred - mean).sum()

    # 残差変動
    res_var = np.square(y_pred - y_obs).sum()

    # 決定係数=1-残差変動/全変動
    r2c = -(res_var / all_var) + 1

    evs = explained_variance_score(y_obs, y_pred)

    return r2c, evs


def static_evaluation(y_obs):
    std = np.std(y_obs)
    var = np.var(y_obs)
    mean = np.mean(y_obs)
    sq_res = np.square(y_obs - mean)
    abs_res = np.abs(y_obs - mean)

    return mean, std, var, sq_res, abs_res


#### Optimaisation function

def abs_spys_fit(xdata, ydata, para):
    """
    Loss function: MAE
    sipy optimaize.minimize 
    """
    
    fit_parax = optimize.minimize(spys_mae_res, para, args=(xdata, ydata), method='Nelder-Mead')
    fit_para = fit_parax.x
    fit_spys = ff.spys(xdata, fit_para[0], fit_para[1], fit_para[2], fit_para[3])
    
    r2, rmse, mae, ratio = evaluation(ydata, fit_spys)

    return fit_spys, fit_para, r2, ratio


def lsq_spys_fit(xdata, ydata, para):
    """
    Loss function: RMSE
    optimiaize -> curve_fit 
    """
    try:
        fit_parax = optimize.curve_fit(f=ff.spys, xdata=xdata, ydata=ydata, p0=para)
        fit_para = fit_parax[0]
    except RuntimeError:
        print('RuntimeError')
        fit_para = [7.0,100,1,0]

    fit_spys = ff.spys(xdata, fit_para[0], fit_para[1], fit_para[2], fit_para[3])

    r2, rmse, mae, ratio = evaluation(ydata, fit_spys)

    return fit_spys, fit_para, r2, ratio

def lsq_op_spys_fit(xdata, ydata, para):
    """
    Loss function: RMSE
    sipy optimaize.minimize 
    """
    
    fit_parax = optimize.minimize(spys_rmse_res, para, args=(xdata, ydata), method='Nelder-Mead')
    fit_para = fit_parax.x
    fit_spys = ff.spys(xdata, fit_para[0], fit_para[1], fit_para[2], fit_para[3])

    r2, rmse, mae, ratio = evaluation(ydata, fit_spys)

    return fit_spys, fit_para, r2, ratio
