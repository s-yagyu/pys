"""
分析範囲を絞って解析する関数

"""

import os
import time
from datetime import datetime

import pandas as pd
from pandas import Series, DataFrame

import scipy as sp
from scipy.optimize import leastsq
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt

from pysfunclib import fowler_func as ff
from pysfunclib import fowler_func_opti as ffo
from pysfunclib import fit_prediction_lib as fpl


class SPYSRangFit(object):
    """
    For JSA Paper
    Fitting range restricted methods
    
    """
    def __init__(self, xdata, ydata, para=None):

        if para == None:
            para = [4.8,300,1,1]
            
        self.para = para
        
        self.xdata = xdata
        self.ydata = ydata    
    
    def fitting_del_range(self,start_remove=0, iteration_step=1):
        """
        Loss function: mae
        ratio=rmse/mae<1.42
        
        
        """
        
        _, n_abs_fit_para, n_abs_r2, n_abs_ratio = ffo.abs_spys_fit(self.xdata, 
                                                                    self.ydata,
                                                                    self.para)

        _, n_lsq_fit_para, n_lsq_r2, n_lsq_ratio = ffo.lsq_spys_fit(self.xdata, 
                                                                    self.ydata,
                                                                    self.para)
        
        self.start_remove = start_remove
        self.iteration_step = iteration_step

        # fittingで評価
        remove = self.start_remove

        while True:

            if remove == 0:
                self.xdata_ = self.xdata[:]
                self.ydata_ = self.ydata[:]

            else:
                self.xdata_ = self.xdata[:(-1 * remove)]
                self.ydata_ = self.ydata[:(-1 * remove)]

            self.abs_fit_spys, self.abs_fit_para, abs_r2, abs_ratio = ffo.abs_spys_fit(self.xdata_, self.ydata_,
                                                                                       self.para)

            self.lsq_fit_spys, self.lsq_fit_para, lsq_r2, lsq_ratio = ffo.lsq_spys_fit(self.xdata_, self.ydata_,
                                                                                       self.para)

            ratio = abs_ratio
            print('----')
            print('Remove point:', remove)
            print("Ratio:",ratio)
            print('spys_lsq[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.lsq_fit_spys))
            print('lsq_fit para', self.lsq_fit_para)
            print()
            print('spys_abs[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.abs_fit_spys))
            print('abs_fit para', self.abs_fit_para)
            print()
            print('-----------------')

            # 1.42
            if ratio < 1.42:

                break
                        # 1.42
                        
            if abs_r2 < 0:
                print('R2 is negaive')
                break

            
            remove = remove + self.iteration_step
                
        # return self.abs_fit_para, self.lsq_fit_para
        return { "n_abs_para":n_abs_fit_para,
                "n_abs_r2":n_abs_r2, "n_abs_ratio":n_abs_ratio, 
                "n_lsq_para":n_lsq_fit_para,
                "n_lsq_r2":n_lsq_r2, "n_lsq_ratio":n_lsq_ratio,
                "abs_para":self.abs_fit_para,
                "lsq_para":self.lsq_fit_para,
                "ratio":ratio,
                "r2":abs_r2,
                "remove":remove}
        
    
    def fitting_del_range_plot(self,start_remove=0, iteration_step=1):
        """
        Limit: re=RMSE/MAE=14.2
        
        re value is calculated loss function MAE

        lsq: re = 1.253 <- Gaussian
        abs: re = 1.414 <-laplacean

        この条件になるまでデータを削減する
        do-while 

        remove で削減値を決める


        """
        self.start_remove = start_remove
        self.iteration_step = iteration_step

        # fittingで評価
        start_time = time.time()
        print("Start time: ", datetime.now().strftime('%Y%m%d %H:%M:%S'))

        remove = self.start_remove

        while True:

            if remove == 0:
                self.xdata_ = self.xdata[:]
                self.ydata_ = self.ydata[:]

            else:
                self.xdata_ = self.xdata[:(-1 * remove)]
                self.ydata_ = self.ydata[:(-1 * remove)]

            self.abs_fit_spys, self.abs_fit_para, abs_r2, abs_ratio = ffo.abs_spys_fit(self.xdata_, self.ydata_, self.para)

            self.lsq_fit_spys, self.lsq_fit_para, lsq_r2, lsq_ratio = ffo.lsq_spys_fit(self.xdata_, self.ydata_, self.para)
            
            #ratio <- MAE 
            ratio = abs_ratio
            print('----')
            print('Remove point:', remove)
            print("Ratio:",ratio)
            print('spys_lsq[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.lsq_fit_spys))
            print('lsq_fit para', self.lsq_fit_para)
            print()
            print('spys_abs[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.abs_fit_spys))
            print('abs_fit para', self.abs_fit_para)
            print()
            _,self.lsq_rmse,_,_ = ffo.evaluation(self.ydata_, self.lsq_fit_spys)
            _,_,self.abs_mae,_ = ffo.evaluation(self.ydata_, self.abs_fit_spys)
            print('lsq_rmse/abs_mae:',self.lsq_rmse/self.abs_mae)
            
            print('-----------------')
            fitplot = fpl.FittingComparePlot(self.xdata_, self.ydata_, self.lsq_fit_spys, self.lsq_fit_para, 
                                         self.abs_fit_spys, self.abs_fit_para)
            fitplot.fit_res_plot()

            # 1.42
            if ratio < 1.42:

                break

            else:
                remove = remove + self.iteration_step

        elapsed_time = time.time() - start_time
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ", datetime.now().strftime('%Y%m%d %H:%M:%S'))
        
    def fitting_del_range_rer2(self,start_remove=0, iteration_step=1):
        """
        Loss function: mae
        ratio=rmse/mae<1.42
        r2 compare
        
        
        """
        
        _, n_abs_fit_para, n_abs_r2, n_abs_ratio = ffo.abs_spys_fit(self.xdata, 
                                                                    self.ydata,
                                                                    self.para)

        _, n_lsq_fit_para, n_lsq_r2, n_lsq_ratio = ffo.lsq_spys_fit(self.xdata, 
                                                                    self.ydata,
                                                                    self.para)
        
        self.start_remove = start_remove
        self.iteration_step = iteration_step

        # fittingで評価
        remove = self.start_remove
        r2_val = 0
        abs_fit_para_val = []
        lsq_fit_para_val = []

        while True:

            if remove == 0:
                self.xdata_ = self.xdata[:]
                self.ydata_ = self.ydata[:]

            else:
                self.xdata_ = self.xdata[:(-1 * remove)]
                self.ydata_ = self.ydata[:(-1 * remove)]

            self.abs_fit_spys, self.abs_fit_para, abs_r2, abs_ratio = ffo.abs_spys_fit(self.xdata_, self.ydata_,
                                                                                       self.para)

            self.lsq_fit_spys, self.lsq_fit_para, lsq_r2, lsq_ratio = ffo.lsq_spys_fit(self.xdata_, self.ydata_,
                                                                                       self.para)

            ratio = abs_ratio
            print('----')
            print('Remove point:', remove)
            print("Ratio:",ratio)
            print('spys_lsq[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.lsq_fit_spys))
            print('lsq_fit para', self.lsq_fit_para)
            print()
            print('spys_abs[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.abs_fit_spys))
            print('abs_fit para', self.abs_fit_para)
            print()
            print('-----------------')

            # 1.42
            if abs_r2 < 0:
                print('R2 is negaive')
                r2_val = abs_r2
                abs_fit_para_val = self.abs_fit_para
                lsq_fit_para_val = self.lsq_fit_para
                break
            
            elif ratio < 1.42:
                print('stop!  under 1.41')
                r2_val = abs_r2
                abs_fit_para_val = self.abs_fit_para
                lsq_fit_para_val = self.lsq_fit_para
                break
            
            elif r2_val > abs_r2 and ratio > 1.414 :
                print('stop! over 1.41 and good R2 ')
                
                break

            remove = remove + self.iteration_step
            r2_val = abs_r2
            abs_fit_para_val = self.abs_fit_para
            lsq_fit_para_val = self.lsq_fit_para
            
        # return abs_fit_para_val, lsq_fit_para_val, remove
        return { "n_abs_para":n_abs_fit_para,
                "n_abs_r2":n_abs_r2, "n_abs_ratio":n_abs_ratio, 
                "n_lsq_para":n_lsq_fit_para,
                "n_lsq_r2":n_lsq_r2, "n_lsq_ratio":n_lsq_ratio,
                "abs_para":abs_fit_para_val,
                "lsq_para":lsq_fit_para_val,
                "ratio":ratio,
                "r2":r2_val,
                "remove":remove}
