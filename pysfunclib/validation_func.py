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
from pysfunclib import ml_prediction_lib as mp
from pysfunclib import rest_fit_prediction_lib as rfpl



"""
検証用のデータ解析用コード
Excelでまとめた検証データの解析用モジュール

"""

class DfFrame():
    
    def __init__(self, df):
        self.df = df

    def df_column_set(self):
        self.df["diff_fit"] = 0.01
        self.df["diff_gb"] = 0.01
        self.df["diff_rf"] = 0.01
        self.df["diff_fit_abs"] = 0.01
        
        self.df["predict_fit"] = 0.01
        self.df["predict_gb"] = 0.01
        self.df["predict_rf"] = 0.01
        self.df["predict_fit_abs"] = 0.01
        self.df["del_abs_fit"] = 0.01
        self.df["del_lsq_fit"] = 0.01
        self.df["del_abs_diff"] = 0.01
        self.df["del_lsq_diff"] = 0.01
        self.df["remove"] = 0.01
       
        
    def df_return(self):
        
        return self.df
    
    
    def df_ml(self, path_name): 
        """
        path_name: regration parameter path  
        ex: './spys_reg_20200228_pure/'
        
        """
        
        prd=mp.MLPredict(path_name=path_name)
        prd.param_load()
        
        start_time = time.time()
        print("Start time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
        

        for i in self.df.index:
            
            xdata_ =np.array(eval(self.df["ene"][i]))
            ydata_ = np.array(eval(self.df["n_pys"][i]))
            ml_values= prd.prediction(xdata_,ydata_)

            self.df["predict_gb"][i]=ml_values['gb']
            self.df["diff_gb"][i]=self.df["predict_gb"][i]-self.df['estimate_wf'][i]
            self.df["predict_rf"][i]=ml_values['rf']
            self.df["diff_rf"][i]=self.df["predict_rf"][i]-self.df['estimate_wf'][i]


                
            print("Time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
            print('sample nmae:',self.df["Sample_name"][i])
            print('renge:',self.df["energy_range"][i])
            print('wf:',self.df["estimate_wf"][i])
            print('predict_GB:',self.df["predict_gb"][i])
            print('difference_GB:',self.df["diff_gb"][i])
            print('predict_RF:',self.df["predict_rf"][i])
            print('difference_RF:',self.df["diff_rf"][i])
            print()


        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
 
    
    
    def df_fitting(self, lossfunc='rmse'):
        """
        all range fitting
        lossfunc =
        'mae' : absolute loss function
        'rmse':least squrt loss function
        """

        start_time = time.time()
        print("Start time: ", datetime.now().strftime('%Y%m%d %H:%M:%S') )

        for i in self.df.index:
            #1行の行列に変換

            xdata_ =np.array(eval(self.df["ene"][i]))
            ydata_ = np.array(eval(self.df["n_pys"][i]))
            ini_para = np.array([self.df["estimate_wf"][i],300,1,10])

            
            if lossfunc == 'mae':
                fit_spys, fit_para, r2, ratio = ffo.abs_spys_fit(xdata_,ydata_,ini_para)
                
            elif lossfunc == 'rmse' :
                fit_spys, fit_para, r2, ratio = ffo.lsq_spys_fit(xdata_,ydata_,ini_para)

                
            else :
                pass

            
            self.df.loc[i,["predict_fit"]] = fit_para[0]
            self.df.loc[i,["diff_fit"]]=self.df["predict_fit"][i]-self.df['estimate_wf'][i]
            r2_=r2



            print("Time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
            print('sample name:',self.df["Sample_name"][i])
            print('renge:',self.df["energy_range"][i])
            print('wf:',self.df["estimate_wf"][i])
            print('predict_fit:',self.df["predict_fit"][i],type(self.df["predict_fit"][i]))
            print('difference_fit:',self.df["diff_fit"][i])
            print('r2:',r2_)
            print()

        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )

    def df_fitting2(self):
        """
        all range fitting
        lossfunc =
        'mae' : absolute loss function
        'rmse':least squrt loss function
        """

        start_time = time.time()
        print("Start time: ", datetime.now().strftime('%Y%m%d %H:%M:%S') )

        for i in self.df.index:
            #1行の行列に変換

            xdata_ =np.array(eval(self.df["ene"][i]))
            ydata_ = np.array(eval(self.df["n_pys"][i]))
            ini_para = np.array([self.df["estimate_wf"][i],300,1,10])

            
            
            fit_spys_abs, fit_para_abs, r2_abs, ratio_abs = ffo.abs_spys_fit(xdata_,ydata_,ini_para)
            fit_spys_lsq, fit_para_lsq, r2_lsq, ratio_lsq = ffo.lsq_spys_fit(xdata_,ydata_,ini_para)

            # fitplot=fpl.FittingComparePlot(xdata=xdata_, ydata=ydata_, 
            #                            lsq_fit=fit_spys_lsq, 
            #                            lsq_para=fit_para_lsq, 
            #                            abs_fit=fit_spys_abs, 
            #                            abs_para=fit_para_abs,
            #                            label=ini_para[0])
            # fitplot.fit_plot()
            
            if fit_para_lsq[0] > 7.0:
                self.df.loc[i,["predict_fit"]] = 7.0
                
            elif fit_para_lsq[0] <= 7.0:   
                self.df.loc[i,["predict_fit"]] = fit_para_lsq[0]
                
            self.df.loc[i,["diff_fit"]]=self.df["predict_fit"][i]-self.df['estimate_wf'][i]
            
            if fit_para_abs[0] > 7.0:
                self.df.loc[i,["predict_fit_abs"]] = 7.0
                
            elif   fit_para_abs[0] <= 7.0: 
                self.df.loc[i,["predict_fit_abs"]] = fit_para_abs[0]
                
            self.df.loc[i,["diff_fit_abs"]]=self.df["predict_fit_abs"][i]-self.df['estimate_wf'][i]



            print("Time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
            print('sample name:',self.df["Sample_name"][i])
            print('renge:',self.df["energy_range"][i])
            print('wf:',self.df["estimate_wf"][i])
            print('lsq')
            print('predict_fit:',self.df["predict_fit"][i],type(self.df["predict_fit"][i]))
            print('difference_fit:',self.df["diff_fit"][i])
            print('r2:',r2_lsq)
            print('abs')
            print('predict_fit:',self.df["predict_fit_abs"][i],type(self.df["predict_fit_abs"][i]))
            print('difference_fit:',self.df["diff_fit_abs"][i])
            print('r2:',r2_abs)
            print()

        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
        
    def df_plot_raw(self):
        """
        Fittingなし
        データのみ
        """

        start_time = time.time()
        print("Start time: ", datetime.now().strftime('%Y%m%d %H:%M:%S') )

        for i in self.df.index:
            #1行の行列に変換

            xdata_ =np.array(eval(self.df["ene"][i]))
            ydata_ = np.array(eval(self.df["n_pys"][i]))
            ini_para = np.array([self.df["estimate_wf"][i],300,1,10])

            
            
            fit_spys_abs, fit_para_abs, r2_abs, ratio_abs = ffo.abs_spys_fit(xdata_,ydata_,ini_para)
            fit_spys_lsq, fit_para_lsq, r2_lsq, ratio_lsq = ffo.lsq_spys_fit(xdata_,ydata_,ini_para)

            fitplot=fpl.FittingComparePlot(xdata=xdata_, ydata=ydata_, 
                                       lsq_fit=fit_spys_lsq, 
                                       lsq_para=fit_para_lsq, 
                                       abs_fit=fit_spys_abs, 
                                       abs_para=fit_para_abs,
                                       label=ini_para[0])
            fitplot.raw_plot()
            
            if fit_para_lsq[0] > 7.0:
                self.df.loc[i,["predict_fit"]] = 7.0
                
            elif fit_para_lsq[0] <= 7.0:   
                self.df.loc[i,["predict_fit"]] = fit_para_lsq[0]
                
            self.df.loc[i,["diff_fit"]]=self.df["predict_fit"][i]-self.df['estimate_wf'][i]
            
            if fit_para_abs[0] > 7.0:
                self.df.loc[i,["predict_fit_abs"]] = 7.0
                
            elif   fit_para_abs[0] <= 7.0: 
                self.df.loc[i,["predict_fit_abs"]] = fit_para_abs[0]
                
            self.df.loc[i,["diff_fit_abs"]]=self.df["predict_fit_abs"][i]-self.df['estimate_wf'][i]



            print("Time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
            print('sample name:',self.df["Sample_name"][i])
            print('renge:',self.df["energy_range"][i])
            print('wf:',self.df["estimate_wf"][i])
            print('lsq')
            print('predict_fit:',self.df["predict_fit"][i],type(self.df["predict_fit"][i]))
            print('difference_fit:',self.df["diff_fit"][i])
            print('r2:',r2_lsq)
            print('abs')
            print('predict_fit:',self.df["predict_fit_abs"][i],type(self.df["predict_fit_abs"][i]))
            print('difference_fit:',self.df["diff_fit_abs"][i])
            print('r2:',r2_abs)
            print()

        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
        
    def df_del_fitting_plot_rre(self):
        """
        Loss function:mae mimimaize ->　mae
        Loss function:rmse mimimaize ->　rmse
        relative_ratio: rr= rmse/mae 
        Stop condition: rr < 1.41

        Note: Not Good argolithm
        問題点：最小化した回帰線それぞれからmae,rmseを導出してその比を取っている
        （特許で一番最初のもの）
        データ点が少なくモデルの誤差が多いときrrは不安定になる

        """

        
        start_time = time.time()
        print("Start time: ", datetime.now().strftime('%Y%m%d %H:%M:%S') )
        
        

        for i in self.df.index:
            
            xdata = np.array(eval(self.df["ene"][i]))
            ydata = np.array(eval(self.df["n_pys"][i]))
            ini_para = np.array([self.df["estimate_wf"][i],300,1,10])



            self.start_remove = 0
            self.iteration_step = 1

            # Fitting
            remove = self.start_remove

            while True:

                if remove == 0:
                    self.xdata_ = xdata[:]
                    self.ydata_ = ydata[:]
                    
                                        
                    #All range fit_lsq
                    _, fit_para, _, _ = ffo.lsq_spys_fit(self.xdata_, self.ydata_, ini_para)
                    self.df.loc[i,["predict_fit"]] = fit_para[0]


                else:
                    self.xdata_ = xdata[:(-1 * remove)]
                    self.ydata_ = ydata[:(-1 * remove)]

                self.abs_fit_spys, self.abs_fit_para, abs_r2, abs_ratio = ffo.abs_spys_fit(self.xdata_, self.ydata_, ini_para)

                self.lsq_fit_spys, self.lsq_fit_para, lsq_r2, lsq_ratio = ffo.lsq_spys_fit(self.xdata_, self.ydata_, ini_para)

            
                
                print('---info------')
                print('Sample name:',self.df["Sample_name"][i])
                print('Label WF:',self.df["estimate_wf"][i])
                print('Remove point:', remove)
                print('spys_lsq[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.lsq_fit_spys))
                print('lsq_fit para', self.lsq_fit_para)
                print()
                print('spys_abs[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.abs_fit_spys))
                print('abs_fit para', self.abs_fit_para)
                print()

                _,lsq_rmse,_,_ = ffo.evaluation(self.ydata_, self.lsq_fit_spys)
                _,_,abs_mae,_ = ffo.evaluation(self.ydata_, self.abs_fit_spys)
                relative_ratio=lsq_rmse/abs_mae
                ratio2 = lsq_ratio/abs_ratio

                print('lsq_rmse/abs_mae:{}'.format(relative_ratio))
                print()
                
                print('-----------------')
                
                fitplot = fp.FittingComparePlot(self.xdata_, self.ydata_, self.lsq_fit_spys, self.lsq_fit_para, self.abs_fit_spys, self.abs_fit_para)
                fitplot.fit_res_plot()
                
                # 1.42
#                 if abs_ratio < 1.42 and lsq_ratio < 1.26 and relative_ratio < 1.26:  and (abs_ratio < 1.5 or lsq_ratio<1.3)
#                 if   0.83 < ratio2 < 1.0  :
#                 if   0.886 < ratio2 < 1.128  :
                # if 1.24 < relative_ratio < 1.42 :
                if  relative_ratio < 1.42 :

                    break
                
                else:
                    remove = remove + self.iteration_step

            self.df.loc[i,["del_abs_fit"]]=self.abs_fit_para[0]
            self.df.loc[i,["del_lsq_fit"]]=self.lsq_fit_para[0]
            self.df.loc[i,["del_abs_diff"]]=self.df["del_abs_fit"][i]-self.df['estimate_wf'][i]
            self.df.loc[i,["del_lsq_diff"]]=self.df["del_lsq_fit"][i]-self.df['estimate_wf'][i]
            self.df.loc[i,["remove"]] = remove

            print('############### Final results ############################')
            print("Time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
            print('Sample name:',self.df["Sample_name"][i])
            print('Energy renge:',self.df["energy_range"][i])
            print('Label WF:',self.df["estimate_wf"][i])
            
            print('---')
            print('Ip(lsq):', self.df["del_lsq_fit"][i])
            print('Ip(abs):', self.df["del_abs_fit"][i])
            print()
            print("remove:",remove)
            print('lsq_rmse/abs_mae:{}'.format(relative_ratio))
            print('---')   

            print()
            print('R2_lsq:',lsq_r2)
            print('lsq_ratio',lsq_ratio)
            print('Ip(lsq):', self.df["del_lsq_fit"][i])
            print()
            print('R2_abs:',abs_r2)
            print('abs_ratio',abs_ratio)
            print('Ip(abs):', self.df["del_abs_fit"][i])
            print()
            print('#########################################################')
            
        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )


    def df_del_fitting_plot_re(self): 
        """
        Simple model
        
        Loss function : MAE
        Stop condition: Re=RMSE/MAE<1.41 
        
        Note:
        分析範囲の削減が止まらない可能性がある。
        """

        start_time = time.time()
        print("Start time: ", datetime.now().strftime('%Y%m%d %H:%M:%S') )
        
        

        for i in self.df.index:
        

            xdata = np.array(eval(self.df["ene"][i]))
            ydata = np.array(eval(self.df["n_pys"][i]))
            ini_para = np.array([self.df["estimate_wf"][i],300,1,10])


            self.start_remove = 0
            self.iteration_step = 1

            # Fitting
            remove = self.start_remove
            r2_val = 0
            abs_fit_para_val = 0
            lsq_fit_para_val = 0
            
            while True:

                if remove == 0:
                    self.xdata_ = xdata[:]
                    self.ydata_ = ydata[:]
                    
                    #All range fit_lsq
                    _, fit_para, _, _ = ffo.lsq_spys_fit(self.xdata_, self.ydata_, ini_para)
                    
                    
                    self.df.loc[i,["predict_fit"]]= fit_para[0]
                    
                else:
                    self.xdata_ = xdata[:(-1 * remove)]
                    self.ydata_ = ydata[:(-1 * remove)]

                self.abs_fit_spys, self.abs_fit_para, abs_r2, abs_ratio = ffo.abs_spys_fit(self.xdata_, self.ydata_, ini_para)

                self.lsq_fit_spys, self.lsq_fit_para, lsq_r2, lsq_ratio = ffo.lsq_spys_fit(self.xdata_, self.ydata_, ini_para)

            
                print('---info------')
                print('sample name:',self.df["Sample_name"][i])
                print('Label WF:',self.df["estimate_wf"][i])
                print('Remove point:', remove)
                print('spys_lsq[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.lsq_fit_spys))
                print('lsq_fit para', self.lsq_fit_para)
                print()
                print('spys_abs[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.abs_fit_spys))
                print('abs_fit para', self.abs_fit_para)
                print()
                print('-----------------')


                ratio = abs_ratio
                
                fitplot = fp.FittingComparePlot(self.xdata_, self.ydata_, self.lsq_fit_spys, self.lsq_fit_para, self.abs_fit_spys, self.abs_fit_para)
                fitplot.fit_res_plot()

                if abs_r2 < 0:
                    print('R2 is negaive')
                    print('remove point: ',remove)
                    self.df.loc[i,["del_abs_fit"]]=7.0
                    self.df.loc[i,["del_lsq_fit"]]=7.0
                    self.df.loc[i,["del_abs_diff"]]=self.df["del_abs_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["del_lsq_diff"]]=self.df["del_lsq_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["remove"]] = remove
                    break

                elif ratio < 1.414:
                    print('stop!  under 1.41')
                    print('remove point: ',remove)
                    self.df.loc[i,["del_abs_fit"]]=self.abs_fit_para[0]
                    self.df.loc[i,["del_lsq_fit"]]=self.lsq_fit_para[0]
                    self.df.loc[i,["del_abs_diff"]]=self.df["del_abs_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["del_lsq_diff"]]=self.df["del_lsq_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["remove"]] = remove
                    
                    break
                    
                else:
                    pass
                    
                remove = remove + self.iteration_step

            print('############### Final results ############################')
            print("Time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
            print('Sample name:',self.df["Sample_name"][i])
            print('Energy renge:',self.df["energy_range"][i])
            print('Label WF:',self.df["estimate_wf"][i])
            
            print('---')
            print('Ip(lsq):', self.df["del_lsq_fit"][i])
            print('Ip(abs):', self.df["del_abs_fit"][i])
            print()
            print("remove:",remove)
            print('R2_abs:',abs_r2)
            print('abs_ratio',abs_ratio)
            print('---')   

            print()
            print('R2_lsq:',lsq_r2)
            print('lsq_ratio',lsq_ratio)
            print('Ip(lsq):', self.df["del_lsq_fit"][i])
            print()
            print('R2_abs:',abs_r2)
            print('abs_ratio',abs_ratio)
            print('Ip(abs):', self.df["del_abs_fit"][i])
            print()
            print('#########################################################')       
            
        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )   
        
    def df_del_fitting_plot_rer2(self):
        """
        Loss function : MAE
        Stop condition: Re=RMSE/MAE<1.41 and R2(abs) improvement
        
        Note:
        比とR2の比較で削減範囲を止める
        最新のアルゴリズム

        """

        start_time = time.time()
        print("Start time: ", datetime.now().strftime('%Y%m%d %H:%M:%S') )
        

        for i in self.df.index:

            xdata = np.array(eval(self.df["ene"][i]))
            ydata = np.array(eval(self.df["n_pys"][i]))
            ini_para = np.array([self.df["estimate_wf"][i],300,1,10])


            self.start_remove = 0
            self.iteration_step = 1

            # Fitting
            remove = self.start_remove
            r2_val = 0
            abs_fit_para_val = 0
            lsq_fit_para_val = 0
            
            while True:

                if remove == 0:
                    self.xdata_ = xdata[:]
                    self.ydata_ = ydata[:]
                    
                    #All range fit_lsq
                    _, fit_para, _, _ = ffo.lsq_spys_fit(self.xdata_, self.ydata_, ini_para)
                    
                    
                    self.df.loc[i,["predict_fit"]]= fit_para[0]
                    
                else:
                    self.xdata_ = xdata[:(-1 * remove)]
                    self.ydata_ = ydata[:(-1 * remove)]

                self.abs_fit_spys, self.abs_fit_para, abs_r2, abs_ratio = ffo.abs_spys_fit(self.xdata_, self.ydata_, ini_para)

                self.lsq_fit_spys, self.lsq_fit_para, lsq_r2, lsq_ratio = ffo.lsq_spys_fit(self.xdata_, self.ydata_, ini_para)

            
                print('---info------')
                print('sample name:',self.df["Sample_name"][i])
                print('Label WF:',self.df["estimate_wf"][i])
                print('Remove point:', remove)
                print('spys_lsq[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.lsq_fit_spys))
                print('lsq_fit para', self.lsq_fit_para)
                print()
                print('spys_abs[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.abs_fit_spys))
                print('abs_fit para', self.abs_fit_para)
                print()
                print('-----------------')


                ratio = abs_ratio

                fitplot = fp.FittingComparePlot(self.xdata_, self.ydata_, self.lsq_fit_spys, self.lsq_fit_para, self.abs_fit_spys, self.abs_fit_para)
                fitplot.fit_res_plot()

                if abs_r2 < 0:
                    print('R2 is negaive')
                    print('remove point: ',remove)
                    self.df.loc[i,["del_abs_fit"]]=7.0
                    self.df.loc[i,["del_lsq_fit"]]=7.0
                    self.df.loc[i,["del_abs_diff"]]=self.df["del_abs_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["del_lsq_diff"]]=self.df["del_lsq_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["remove"]] = remove
                    break

                elif ratio < 1.414:
                    print('stop!  under 1.41')
                    print('remove point: ',remove)
                    self.df.loc[i,["del_abs_fit"]]=self.abs_fit_para[0]
                    self.df.loc[i,["del_lsq_fit"]]=self.lsq_fit_para[0]
                    self.df.loc[i,["del_abs_diff"]]=self.df["del_abs_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["del_lsq_diff"]]=self.df["del_lsq_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["remove"]] = remove
                    
                    break
                
                elif r2_val > abs_r2 and ratio > 1.414 :
                    print('stop! over 1.41 and good R2 ')
                    print('remove point: ',remove-1)
                    self.df.loc[i,["del_abs_fit"]]=abs_fit_para_val
                    self.df.loc[i,["del_lsq_fit"]]=lsq_fit_para_val
                    self.df.loc[i,["del_abs_diff"]]=self.df["del_abs_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["del_lsq_diff"]]=self.df["del_lsq_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["remove"]] = remove-1
                    break

                else:
                    pass
                    
                
                remove = remove + self.iteration_step
                r2_val = abs_r2
                abs_fit_para_val = self.abs_fit_para[0]
                lsq_fit_para_val = self.lsq_fit_para[0]

            print('############### Final results ############################')
            print("Time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
            print('Sample name:',self.df["Sample_name"][i])
            print('Energy renge:',self.df["energy_range"][i])
            print('Label WF:',self.df["estimate_wf"][i])
            
            print('---')
            print('Ip(lsq):', self.df["del_lsq_fit"][i])
            print('Ip(abs):', self.df["del_abs_fit"][i])
            print()
            print("remove:",remove)
            print('R2_abs:',abs_r2)
            print('abs_ratio',abs_ratio)
            print('---')   

            print()
            print('R2_lsq:',lsq_r2)
            print('lsq_ratio',lsq_ratio)
            print('Ip(lsq):', self.df["del_lsq_fit"][i])
            print()
            print('R2_abs:',abs_r2)
            print('abs_ratio',abs_ratio)
            print('Ip(abs):', self.df["del_abs_fit"][i])
            print()
            print('#########################################################')       
            
        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
        
    def df_del_fitting_plot_r2(self):
        """
        Loss function MAE
        Stop condtion: R2 improvement
        
        Note:
        R2の値だけでR2の値が前の値よりも悪くなった時点でストップする。

        """

        start_time = time.time()
        print("Start time: ", datetime.now().strftime('%Y%m%d %H:%M:%S') )
        
        

        for i in self.df.index:


            xdata = np.array(eval(self.df["ene"][i]))
            ydata = np.array(eval(self.df["n_pys"][i]))
            ini_para = np.array([self.df["estimate_wf"][i],300,1,10])


            self.start_remove = 0
            self.iteration_step = 1

            # Fitting
            remove = self.start_remove
            r2_val = 0
            abs_fit_para_val = 0
            lsq_fit_para_val = 0
            
            while True:

                if remove == 0:
                    self.xdata_ = xdata[:]
                    self.ydata_ = ydata[:]
                    
                    #All range fit_lsq
                    _, fit_para, _, _ = ffo.lsq_spys_fit(self.xdata_, self.ydata_, ini_para)
                    
                    
                    self.df.loc[i,["predict_fit"]]= fit_para[0]
                    
                else:
                    self.xdata_ = xdata[:(-1 * remove)]
                    self.ydata_ = ydata[:(-1 * remove)]

                self.abs_fit_spys, self.abs_fit_para, abs_r2, abs_ratio = ffo.abs_spys_fit(self.xdata_, self.ydata_, ini_para)

                self.lsq_fit_spys, self.lsq_fit_para, lsq_r2, lsq_ratio = ffo.lsq_spys_fit(self.xdata_, self.ydata_, ini_para)

            
                
                print('---info------')
                print('sample name:',self.df["Sample_name"][i])
                print('Label WF:',self.df["estimate_wf"][i])
                print('Remove point:', remove)
                print('spys_lsq[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.lsq_fit_spys))
                print('lsq_fit para', self.lsq_fit_para)
                print()
                print('spys_abs[r2,rmse,mae,ratio]:', ffo.evaluation(self.ydata_, self.abs_fit_spys))
                print('abs_fit para', self.abs_fit_para)
                print()
                print('-----------------')

                
                fitplot = fp.FittingComparePlot(self.xdata_, self.ydata_, self.lsq_fit_spys, self.lsq_fit_para, self.abs_fit_spys, self.abs_fit_para)
                fitplot.fit_res_plot()
                

                
                if r2_val > abs_r2  :
                    print('stop! good R2 ')
                    print('remove point: ',remove-1)
                    self.df.loc[i,["del_abs_fit"]]=abs_fit_para_val
                    self.df.loc[i,["del_lsq_fit"]]=lsq_fit_para_val
                    self.df.loc[i,["del_abs_diff"]]=self.df["del_abs_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["del_lsq_diff"]]=self.df["del_lsq_fit"][i]-self.df['estimate_wf'][i]
                    self.df.loc[i,["remove"]] = remove-1
                    break
                    
                
                remove = remove + self.iteration_step
                r2_val = abs_r2
                abs_fit_para_val = self.abs_fit_para[0]
                lsq_fit_para_val = self.lsq_fit_para[0]

            print('############### Final results ############################')
            print("Time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )    
            print('Sample name:',self.df["Sample_name"][i])
            print('Energy renge:',self.df["energy_range"][i])
            print('Label WF:',self.df["estimate_wf"][i])
            
            print('---')
            print('Ip(lsq):', self.df["del_lsq_fit"][i])
            print('Ip(abs):', self.df["del_abs_fit"][i])
            print()
            print("remove:",remove-1)
            print('R2_abs:',r2_val)
            print('abs_ratio',abs_ratio)
            print('---')   

            print()
            print('R2_lsq:',lsq_r2)
            print('lsq_ratio',lsq_ratio)
            print('Ip(lsq):', self.df["del_lsq_fit"][i])
            print()
            print('R2_abs:',abs_r2)
            print('abs_ratio',abs_ratio)
            print('Ip(abs):', self.df["del_abs_fit"][i])
            print()
            print('#########################################################')    
            
        elapsed_time = time.time() - start_time
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )




def select_plot(df_name,comment='',gb=True, rf=True, fit=True, dabsfit=True, dlsqfit=True):
    """
    Graph making 
    df_name is select df
    
    """

    #default　12
    plt.rcParams["font.size"] = 14

    plt.tight_layout()
    
    start_time = time.time()
    print("Start time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
    print("comment: ",comment)
    print("number of data: ",df_name['diff_gb'].count())
    
    for ii in df_name.index:
#         print('index:',ii)
#         print()
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        ax.plot(np.array(eval(df_name["ene"][ii])),np.array(eval(df_name["n_pys"][ii])),'-o',label='Data')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]')
        max_value_y=max(np.array(eval(df_name["n_pys"][ii])))
        min_value_y=min(np.array(eval(df_name["n_pys"][ii])))
        max_value_x=max(np.array(eval(df_name["ene"][ii])))
        min_value_x=min(np.array(eval(df_name["ene"][ii])))        
        ax.set_ylim(0,max_value_y)
    
        ax.set_title('Sample: {}, Power: {} [nW]'.format(df_name["Sample_name"][ii],df_name["photon_power"][ii]))
        
        ax.annotate('Label:{:.2f}'.format(df_name["estimate_wf"][ii]), xy=(df_name["estimate_wf"][ii], 0), 
                    xytext=(df_name["estimate_wf"][ii], max_value_y*0.1),
                    arrowprops=dict(facecolor='green',lw=1,shrinkA=0,shrinkB=0))
        if gb ==  True:
            ax.annotate('GB:{:.2f}'.format(df_name["predict_gb"][ii]), xy=(df_name["predict_gb"][ii], 0), 
                        xytext=(df_name["predict_gb"][ii],  max_value_y*0.3),
                        arrowprops=dict(facecolor='red',lw=1,shrinkA=0,shrinkB=0))
        if rf == True:
            ax.annotate('RF:{:.2f}'.format(df_name["predict_rf"][ii]), xy=(df_name["predict_rf"][ii], 0), 
                        xytext=(df_name["predict_rf"][ii],  max_value_y*0.5),
                        arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0))
            
        if fit == True:
            ax.annotate('Fit:{:.2f}'.format(df_name["predict_fit"][ii]), xy=(df_name["predict_fit"][ii], 0),  
                        xytext=(df_name["predict_fit"][ii],  max_value_y*0.9),
                        arrowprops=dict(facecolor='black',lw=1,shrinkA=0,shrinkB=0))
            
        if dabsfit == True:
            ax.annotate('arFit:{:.2f}'.format(df_name["del_abs_fit"][ii]), xy=(df_name["del_abs_fit"][ii], 0),  
                        xytext=(df_name["del_abs_fit"][ii],  max_value_y*0.6),
                        arrowprops=dict(facecolor='magenta',lw=1,shrinkA=0,shrinkB=0))
        
        if dlsqfit==True:
            ax.annotate('srfit:{:.2f}'.format(df_name["del_lsq_fit"][ii]), xy=(df_name["del_lsq_fit"][ii], 0),  
                        xytext=(df_name["del_lsq_fit"][ii],  max_value_y*0.8),
                        arrowprops=dict(facecolor='cyan',lw=1,shrinkA=0,shrinkB=0))


#         plt.text(min_value_x+0.01,max_value_y*0.45,
#                  "Label: {:.2f}\nGB : {:.2f}\ndif : {:.2f}\nRF : {:.2f}\ndif : {:.2f}\nsFit : {:.2f}\ndif : {:.2f}"
#                  .format(df_name["estimate_wf"][ii],
#                          df_name["predict_gb"][ii],df_name['diff_gb'][ii],
#                          df_name["predict_rf"][ii],df_name['diff_rf'][ii],
#                          df_name["del_abs_fit"][ii],df_name["del_lsq_diff"][ii]),fontsize=14)
        
        plt.text(min_value_x+0.01,max_value_y*0.8,"Label: {:.2f}".format(df_name["estimate_wf"][ii],fontsize=14))

    plt.show()

    elapsed_time = time.time() - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
    print(df_name.count())

def select_plot_widerange(df_name,comment='',gb=True, rf=True, fit=True, dabsfit=True, dlsqfit=True):
    """
    Graph making 
    df_name is select df
    xmaxを7.0eVにする
    
    """
    #default　12
    plt.rcParams["font.size"] = 14

    plt.tight_layout()
    
    start_time = time.time()
    print("Start time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
    print("comment: ",comment)
    print("number of data: ",df_name['diff_gb'].count())
    
    for ii in df_name.index:
#         print('index:',ii)
#         print()
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        ax.plot(np.array(eval(df_name["ene"][ii])),np.array(eval(df_name["n_pys"][ii])),'-o',label='Data')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]')
        max_value_y=max(np.array(eval(df_name["n_pys"][ii])))
        min_value_y=min(np.array(eval(df_name["n_pys"][ii])))
        max_value_x=7.0
        # max_value_x=max(np.array(eval(df_name["ene"][ii])))
        min_value_x=min(np.array(eval(df_name["ene"][ii])))        
        ax.set_ylim(0,max_value_y)
        ax.set_xlim(min_value_x,max_value_x)
        ax.set_title('Sample: {}, Power: {} [nW]'.format(df_name["Sample_name"][ii],df_name["photon_power"][ii]))
        
        ax.annotate('Label:{:.2f}'.format(df_name["estimate_wf"][ii]), xy=(df_name["estimate_wf"][ii], 0), 
                    xytext=(df_name["estimate_wf"][ii], max_value_y*0.1),
                    arrowprops=dict(facecolor='green',lw=1,shrinkA=0,shrinkB=0))
        if gb ==  True:
            ax.annotate('GB:{:.2f}'.format(df_name["predict_gb"][ii]), xy=(df_name["predict_gb"][ii], 0), 
                        xytext=(df_name["predict_gb"][ii],  max_value_y*0.3),
                        arrowprops=dict(facecolor='red',lw=1,shrinkA=0,shrinkB=0))
        if rf == True:
            ax.annotate('RF:{:.2f}'.format(df_name["predict_rf"][ii]), xy=(df_name["predict_rf"][ii], 0), 
                        xytext=(df_name["predict_rf"][ii],  max_value_y*0.5),
                        arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0))
            
        if fit == True:
            ax.annotate('Fit:{:.2f}'.format(df_name["predict_fit"][ii]), xy=(df_name["predict_fit"][ii], 0),  
                        xytext=(df_name["predict_fit"][ii],  max_value_y*0.9),
                        arrowprops=dict(facecolor='black',lw=1,shrinkA=0,shrinkB=0))
            
        if dabsfit == True:
            ax.annotate('arFit:{:.2f}'.format(df_name["del_abs_fit"][ii]), xy=(df_name["del_abs_fit"][ii], 0),  
                        xytext=(df_name["del_abs_fit"][ii],  max_value_y*0.6),
                        arrowprops=dict(facecolor='magenta',lw=1,shrinkA=0,shrinkB=0))
        
        if dlsqfit==True:
            ax.annotate('srfit:{:.2f}'.format(df_name["del_lsq_fit"][ii]), xy=(df_name["del_lsq_fit"][ii], 0),  
                        xytext=(df_name["del_lsq_fit"][ii],  max_value_y*0.8),
                        arrowprops=dict(facecolor='cyan',lw=1,shrinkA=0,shrinkB=0))
            
#         plt.text(min_value_x+0.01,max_value_y*0.45,
#                  "Label: {:.2f}\nGB : {:.2f}\ndif : {:.2f}\nRF : {:.2f}\ndif : {:.2f}\nsFit : {:.2f}\ndif : {:.2f}"
#                  .format(df_name["estimate_wf"][ii],
#                          df_name["predict_gb"][ii],df_name['diff_gb'][ii],
#                          df_name["predict_rf"][ii],df_name['diff_rf'][ii],
#                          df_name["del_abs_fit"][ii],df_name["del_lsq_diff"][ii]),fontsize=14)
        
        plt.text(min_value_x+0.01,max_value_y*0.8,"Label: {:.2f}".format(df_name["estimate_wf"][ii],fontsize=14))
    
    plt.show()

    elapsed_time = time.time() - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
    print(df_name.count())


def select_plot_rangeselect(df_name, comment='', maxlimt=7, gb=True, rf=True, fit=True, dabsfit=True, dlsqfit=True):
    """
    Fittingの回帰曲線を載せる
    
    Graph making 
    df_name is select df
    xmaxを7.0eVにする
    
    """
    #default　12
    plt.rcParams["font.size"] = 10

    plt.tight_layout()
    
    start_time = time.time()
    print("Start time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
    print("comment: ",comment)
    print("number of data: ",df_name['diff_gb'].count())
    
    for ii in df_name.index:
#         print('index:',ii)
#         print()
        ini_para=np.array([df_name["estimate_wf"][ii],300,1,10])
        xdata = np.array(eval(df_name["ene"][ii]))
        ydata = np.array(eval(df_name["n_pys"][ii]))
        
        resultx = leastsq(ffo.spys_residual_func, ini_para,args=(xdata,ydata),maxfev=1000000)
        result = resultx[0]
        fit_spys = ff.spys(xdata, result[0], result[1], result[2], result[3])
        print(result)
        
        fig, ax = plt.subplots(1,1,figsize=(6,4),dpi=300)
        ax.plot(xdata, ydata,'-o',label='Data')
        ax.plot(xdata,fit_spys,linestyle='solid', label='Fit')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]')
        max_value_y=max(ydata)
        min_value_y=min(ydata)
        max_value_x=maxlimt
        # max_value_x=max(np.array(eval(df_name["ene"][ii])))
        min_value_x=min(xdata)     
        min_value_x=4.0
        ax.set_ylim(0,max_value_y)
        ax.set_xlim(min_value_x,max_value_x)

        ax.set_title('Sample: {}, Power: {} [nW]'.format(df_name["Sample_name"][ii],df_name["photon_power"][ii]))
        
        if dlsqfit==True:
            ax.annotate('srfit:{:.2f}'.format(df_name["del_lsq_fit"][ii]), xy=(df_name["del_lsq_fit"][ii], 0),  
                        xytext=(df_name["del_lsq_fit"][ii],  max_value_y*0.8),
                        arrowprops=dict(facecolor='cyan',lw=1,shrinkA=0,shrinkB=0))
            
        if dabsfit == True:
            ax.annotate('arFit:{:.2f}'.format(df_name["del_abs_fit"][ii]), xy=(df_name["del_abs_fit"][ii], 0),  
                        xytext=(df_name["del_abs_fit"][ii],  max_value_y*0.6),
                        arrowprops=dict(facecolor='magenta',lw=1,shrinkA=0,shrinkB=0))
        if fit == True:
            ax.annotate('Fit:{:.2f}'.format(df_name["predict_fit"][ii]), xy=(df_name["predict_fit"][ii], 0),  
                        xytext=(df_name["predict_fit"][ii],  max_value_y*0.9),
                        arrowprops=dict(facecolor='black',lw=1,shrinkA=0,shrinkB=0))
        if rf == True:
            ax.annotate('RF:{:.2f}'.format(df_name["predict_rf"][ii]), xy=(df_name["predict_rf"][ii], 0), 
                        xytext=(df_name["predict_rf"][ii],  max_value_y*0.5),
                        arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0))
        if gb ==  True:
            ax.annotate('GB:{:.2f}'.format(df_name["predict_gb"][ii]), xy=(df_name["predict_gb"][ii], 0), 
                        xytext=(df_name["predict_gb"][ii],  max_value_y*0.3),
                        arrowprops=dict(facecolor='red',lw=1,shrinkA=0,shrinkB=0))
                    
        ax.annotate('Label:{:.2f}'.format(df_name["estimate_wf"][ii]), xy=(df_name["estimate_wf"][ii], 0), 
                    xytext=(df_name["estimate_wf"][ii], max_value_y*0.1),
                    arrowprops=dict(facecolor='green',lw=1,shrinkA=0,shrinkB=0))

        plt.legend(loc='upper left')

#         plt.text(min_value_x+0.01,max_value_y*0.45,
#                  "Label: {:.2f}\nGB : {:.2f}\ndif : {:.2f}\nRF : {:.2f}\ndif : {:.2f}\nsFit : {:.2f}\ndif : {:.2f}"
#                  .format(df_name["estimate_wf"][ii],
#                          df_name["predict_gb"][ii],df_name['diff_gb'][ii],
#                          df_name["predict_rf"][ii],df_name['diff_rf'][ii],
#                          df_name["del_abs_fit"][ii],df_name["del_lsq_diff"][ii]))
        plt.text(min_value_x+0.1,max_value_y*0.6,
                 "Label: {:.2f}\nGB : {:.2f}\nRF : {:.2f}\nFit : {:.2f}"
                 .format(df_name["estimate_wf"][ii],
                         df_name["predict_gb"][ii],
                         df_name["predict_rf"][ii],
                         df_name["predict_fit"][ii]))
        
#         plt.text(min_value_x+0.01,max_value_y*0.8,"Label: {:.2f}".format(df_name["estimate_wf"][ii],fontsize=14))
    plt.show()

    elapsed_time = time.time() - start_time
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("Finished time: ",datetime.now().strftime('%Y%m%d %H:%M:%S') )
    print(df_name.count())


def yy_plot(df,xyrange=(4.5,6.0), gb=True, rf=True, fit=True, dabsfit=True, mark_name=True, abs_fit=False):
    
    fig = plt.figure(figsize=(15, 15))
    
    yyplot = plt.plot(df["estimate_wf"],df["estimate_wf"], label="R$^{2}$=1",c='r', alpha=1)
    yyplot = plt.plot(df["estimate_wf"],df["estimate_wf"]+0.3, label="+0.3 eV",c='y',linestyle='-', alpha=0.5)
    yyplot = plt.plot(df["estimate_wf"],df["estimate_wf"]-0.3, label="-0.3 eV",c='y',linestyle='-', alpha=0.5)
    
    if gb == True:
        yplot_gb = plt.scatter(df["estimate_wf"], df["predict_gb"], c='r',marker='o',label="GB",alpha=1)
    
    if rf == True:
        yplot_rf = plt.scatter(df["estimate_wf"], df["predict_rf"], c='m', marker='D',label="RF", alpha=1)
        
    if fit == True:
        yplot_fit = plt.scatter(df["estimate_wf"], df["predict_fit"], c='b', marker='^' ,label="Fit_lsq",alpha=1)
        
    if abs_fit == True:
        yplot_fit = plt.scatter(df["estimate_wf"], df["predict_fit_abs"], c='g', marker='s' ,label="Fit_abs",alpha=1)
            
    if dabsfit == True:
        yplot_abs = plt.scatter(df["estimate_wf"], df["del_abs_fit"], c='c', marker='*', label="Fit_rem",alpha=1)
    
    
    plt.xlim(xyrange[0],xyrange[1]) 
    plt.ylim(xyrange[0],xyrange[1]) 

    plt.legend(loc='lower right',fontsize=15)
    plt.xlabel('Observed[eV]',fontsize=20,labelpad=10)
    plt.ylabel('Predicted[eV]',fontsize=20,labelpad=10)
    plt.title('Observed-Predicted Plot',fontsize=20,pad=20)
    plt.tick_params(labelsize=16)
    
    if mark_name == True:
        
        for i, txt in enumerate(df["Sample_name"].values):
            if gb == True:
                plt.annotate(txt,(df["estimate_wf"].values[i], df["predict_gb"].values[i]),size = 15)
            if rf == True:
                plt.annotate(txt,(df["estimate_wf"].values[i], df["predict_rf"].values[i]),size = 15)
            if fit == True:   
                plt.annotate(txt,(df["estimate_wf"].values[i], df["predict_fit"].values[i]),size = 15)
            if dabsfit == True:            
                plt.annotate(txt,(df["estimate_wf"].values[i], df["del_abs_fit"].values[i]),size = 15) 
            if abs_fit == True:   
                plt.annotate(txt,(df["estimate_wf"].values[i], df["predict_fit_abs"].values[i]),size = 15)  
            
    plt.show()
    
    return fig

def class_separation(df,comment='', gb=True, rf=True, fit=True, dabsfit=False, dlsqfit=False, abs_fit=False):
    
    """
    example:
    newdf_name=df_select01
    df=df
    para="diff_gb"
    para="diff_rf"
    para="diff_fit"
    para="diff_del_abs"           
    para="diff_del_lsq"
    """

    paras = []
    
    if gb == True:
        paras.append("diff_gb")
    
    if rf == True:
        paras.append("diff_rf")
        
    if fit == True:
        paras.append("diff_fit")
        
    if abs_fit == True:
        paras.append("diff_fit_abs")
    
    if dabsfit == True:
        paras.append("del_abs_diff")

    if  dlsqfit == True:
        paras.append("del_lsq_diff")
    
    for para in paras:

        df_temp01=df[(abs(df[para])>=0) &(abs(df[para])<=0.1)]
        df_temp02=df[(abs(df[para])>0.1) &(abs(df[para])<=0.2)]
        df_temp03=df[(abs(df[para])>0.2) &(abs(df[para])<=0.3)]
        df_temp04=df[(abs(df[para])>0.3) &(abs(df[para])<=0.4)]
        df_temp05=df[(abs(df[para])>0.4) &(abs(df[para])<=0.5)]
        df_temp06=df[abs(df[para])>0.5]
        df_temp07=df[abs(df[para])<0.3]
        df_temp08=df[abs(df[para])>1.0]
        
        
        print(para)
        print('Comment:',comment)
        print('----------------------')
        print('total',df[para].count())
        print('0<=0.1:',df_temp01[para].count(), '/total:',df_temp01[para].count()/df[para].count())
        print('0.1<=0.2:',df_temp02[para].count(), '/total:',df_temp02[para].count()/df[para].count())
        print('0.2<=0.3:',df_temp03[para].count(), '/total:',df_temp03[para].count()/df[para].count())
        print('0.3<=0.4:',df_temp04[para].count(), '/total:',df_temp04[para].count()/df[para].count())
        print('0.4<=0.5:',df_temp05[para].count(), '/total:',df_temp05[para].count()/df[para].count())
        print('0.5>:',df_temp06[para].count(), '/total:',df_temp06[para].count()/df[para].count())
        print('1.0>:',df_temp08[para].count(), '/total:',df_temp08[para].count()/df[para].count())
        print()
        print('0.3<:',df_temp07[para].count(), '/total:',df_temp07[para].count()/df[para].count())
        print('----------------------')
        print()
        
def fitting_3plots(df,comment=''):
    """
    20200417 add function
    For JSA paper
    lsq, mae, remove fit
    plot graph
    
    """
    
    for i in df.index:
            
        xdata = np.array(eval(df["ene"][i]))
        ydata = np.array(eval(df["n_pys"][i]))
        ini_para = [df["estimate_wf"][i],300,1,10]
        sample_name= df["Sample_name"][i]

        fit_3=rfpl.SPYSRangFit(xdata=xdata, ydata=ydata, para=ini_para )
        fit_3p = fit_3.fitting_del_range_rer2(start_remove=0, iteration_step=1)
        # fit_3p = fit_3.fitting_del_range(start_remove=0, iteration_step=1)
        fit_lsq_p=fit_3p['n_lsq_para']
        fit_abs_p=fit_3p['n_abs_para']
        fit_rem_p=fit_3p['abs_para']
        rem_p=fit_3p['remove']
        
        fit_lsq = ff.spys(xdata, fit_lsq_p[0],fit_lsq_p[1],fit_lsq_p[2],fit_lsq_p[3])
        fit_abs = ff.spys(xdata, fit_abs_p[0],fit_abs_p[1],fit_abs_p[2],fit_abs_p[3])
        
        if rem_p == 0:
            xdata_ = xdata[:]
            ydata_ = ydata[:]
        else:
            xdata_ = xdata[:(-1 * rem_p)]
            ydata_ = ydata[:(-1 * rem_p)]
            
        fit_rem = ff.spys(xdata_, fit_rem_p[0],fit_rem_p[1],fit_rem_p[2],fit_rem_p[3])
        
        if fit_rem_p[0] == fit_abs_p[0]:
            print('abs = rem')
        print('**** remove: {} ****'.format(rem_p))
        
        
        plt.rcParams["font.size"] = 16
        
        fig =plt.figure(figsize=(8,6))
        ax=fig.add_subplot(1,1,1)
        ax.set_title(sample_name)
        ax.set_title('Sample: {}, Power: {} [nW]'.format(df["Sample_name"][i],
                                                         df["photon_power"][i]))
        ax.plot(xdata,ydata,"ro",label='Data')
        ax.plot(xdata,fit_lsq,"b^-",label='Fit_lsq')
        ax.plot(xdata,fit_abs,"gs-",label='Fit_lab')
        ax.plot(xdata_,fit_rem,"c*-",label='Fit_ar')

        ax.legend(loc='upper left')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]')
        max_value_y=max(ydata) 
        ax.set_ylim(0,max_value_y)

        if fit_lsq_p[0] <=7.0 :
            ax.annotate('Fit_lsq:{:.2f}'.format(fit_lsq_p[0]), xy=(fit_lsq_p[0], 0), 
                        xytext=(fit_lsq_p[0],  max_value_y*0.4),
                        arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0))
        
        if fit_abs_p[0] <= 7.0:
            ax.annotate('Fit_lab:{:.2f}'.format(fit_abs_p[0]), xy=(fit_abs_p[0], 0), 
                        xytext=(fit_abs_p[0],  max_value_y*0.6),
                        arrowprops=dict(facecolor='green',lw=1,shrinkA=0,shrinkB=0))
        if fit_rem_p[0] <= 7.0:
            ax.annotate('Fit_ar:{:.2f}\nRem:{}'.format(fit_rem_p[0],rem_p), xy=(fit_rem_p[0] , 0), 
                        xytext=(fit_rem_p[0],  max_value_y*0.8),
                        arrowprops=dict(facecolor='cyan',lw=1,shrinkA=0,shrinkB=0))
        
        
        ax.annotate('Label:{:.2f}'.format(ini_para[0]), xy=(ini_para[0], 0), 
                    xytext=(ini_para[0],  max_value_y*0.2),
                    arrowprops=dict(facecolor='red',lw=1,shrinkA=0,shrinkB=0))
            

        plt.show()