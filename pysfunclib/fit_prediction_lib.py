"""
fitting  and plot functions

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


class SPYSFit(object):
    """
    Two methos of optimaization: 
    loss function: RMSE, MAE
    
    """
    def __init__(self):
        pass

    def fit(self, xdata, ydata, para=None):
        """
        loss function: RMSE, MAE

        """
        
        if para == None:
            para = [4.8,300,1,1]
        
        self.para = para

        self.xdata = xdata
        self.ydata = ydata
        
        
        self.abs_fit_spys, self.abs_fit_para, abs_r2, abs_ratio = ffo.abs_spys_fit(self.xdata, self.ydata,
                                                                                   self.para)
        self.lsq_fit_spys, self.lsq_fit_para, lsq_r2, lsq_ratio = ffo.lsq_spys_fit(self.xdata, self.ydata,
                                                                                   self.para)
        

        return {"abs_fit":self.abs_fit_spys, "abs_para":self.abs_fit_para,
                "abs_r2":abs_r2, "abs_ratio":abs_ratio, 
                "lsq_fit":self.lsq_fit_spys, "lsq_para":self.lsq_fit_para,
                "lsq_r2":lsq_r2, "lsq_ratio": lsq_ratio}

    def plot(self):
        """
        
        """
        abs_plot = FittingPlot(self.xdata,self.ydata,self.abs_fit_spys,self.abs_fit_para)
        lsq_plot = FittingPlot(self.xdata,self.ydata,self.lsq_fit_spys,self.lsq_fit_para)
        print("Loss function: MAE")
        abs_plot.fit_plot()  
        
        print("---")
        print()
        
        print("Loss function: RMSE")
        lsq_plot.fit_plot()  

        print("---------END---------")
        

    def res_plot(self):
        abs_plot2 = FittingPlot(self.xdata,self.ydata,self.abs_fit_spys,self.abs_fit_para)
        lsq_plot2 = FittingPlot(self.xdata,self.ydata,self.lsq_fit_spys,self.lsq_fit_para)
        print("Loss function: MAE")
        abs_plot2.fit_res_plot() 
        
        print("---")
        print()
        
        print("Loss function: RMSE")
        lsq_plot2.fit_res_plot()  

        print("---------END---------")


class FittingPlot():
    
    def __init__(self, xdata, ydata, fit, fit_para):
        self.xdata = xdata
        self.ydata = ydata
        self.fit = fit
        self.fit_para = fit_para

    def fit_plot(self):
        plt.rcParams["font.size"] = 16
        fig =plt.figure(figsize=(8,6))
        ax=fig.add_subplot(1,1,1)
        ax.plot(self.xdata, self.ydata, "ro",label='Data')
        ax.plot(self.xdata, self.fit, "bo-",label='Fit')


        ax.legend(loc='lower right',fontsize=16)
        ax.set_xlabel('Energy [eV]',fontsize=16)
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]',fontsize=16)
        max_value_y=max(self.ydata) 
        ax.set_ylim(0,max_value_y)

        ax.annotate('Ip:{:.2f}'.format(self.fit_para[0]), xy=(self.fit_para[0], 0), 
                    xytext=(self.fit_para[0],  max_value_y*0.4),
                    arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0),fontsize=16)

        plt.show()
        
        
    def fit_res_plot(self):
        
        plt.rcParams["font.size"] = 16
        fig =plt.figure(figsize=(20,8))
        ax=fig.add_subplot(1,3,1)

        ax.plot(self.xdata,self.ydata,"ro",label='Data')
        ax.plot(self.xdata,self.fit,"bo-",label='Fit')
        ax.set_title('Fitting Plots')
        ax.legend(loc='lower right')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]')
        max_value_y=max(self.ydata) 
        ax.set_ylim(0,max_value_y)

        ax.annotate('Ip:{:.2f}'.format(self.fit_para[0]), xy=(self.fit_para[0], 0), 
                    xytext=(self.fit_para[0],  max_value_y*0.4),
                    arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0))


        #residual_plot
        ax2=fig.add_subplot(1,3,2)
        ax2.set_title('Residual Plots')
        ax2.plot(self.fit,self.fit-self.ydata,"bo-",label='Fit')
        ax2.legend(loc='lower right')
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Prediction-Observed')

        #yyplot
        ax3=fig.add_subplot(1,3,3)
        ax3.set_title('yy Plots')
    #     ydelta=ydata*0.42-ydata

        ax3.plot(self.ydata,self.ydata,"ro-",label='Data')
    #     ax3.plot(ydata,ydata-ydelta,"ro-")
    #     ax3.plot(ydata,ydata+ydelta,"ro-")
        ax3.plot(self.ydata,self.fit,"bo-",label='Fit')

        ax3.legend(loc='lower right')
        ax3.set_xlabel('Observed')
        ax3.set_ylabel('Prediction')
        
        plt.show()


class FittingComparePlot():
    
    def __init__(self, xdata, ydata, lsq_fit, lsq_para, abs_fit, abs_para,label=None):
        self.xdata = xdata
        self.ydata = ydata
        self.lsq_fit = lsq_fit
        self.lsq_para = lsq_para
        self.abs_fit = abs_fit
        self.abs_para = abs_para
        self.label = label
    
    def fit_plot(self):
        
        plt.rcParams["font.size"] = 16
        
        fig =plt.figure(figsize=(8,6))
        ax=fig.add_subplot(1,1,1)
        ax.plot(self.xdata,self.ydata,"ro",label='Data')
        ax.plot(self.xdata,self.lsq_fit,"b^-",label='Fit_lsq')
        ax.plot(self.xdata,self.abs_fit,"gs-",label='Fit_abs')

        ax.legend(loc='upper left')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]')
        max_value_y=max(self.ydata) 
        ax.set_ylim(0,max_value_y)

        if self.lsq_para[0] <=7.0 :
            ax.annotate('Fit_lsq:{:.2f}'.format(self.lsq_para[0]), xy=(self.lsq_para[0], 0), 
                        xytext=(self.lsq_para[0],  max_value_y*0.4),
                        arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0))
        
        if self.abs_para[0] <= 7.0:
            ax.annotate('Fit_abs:{:.2f}'.format(self.abs_para[0]), xy=(self.abs_para[0], 0), 
                        xytext=(self.abs_para[0],  max_value_y*0.6),
                        arrowprops=dict(facecolor='green',lw=1,shrinkA=0,shrinkB=0))
        
        if self.label != None:
            ax.annotate('Label:{:.2f}'.format(self.label), xy=(self.label, 0), 
                        xytext=(self.label,  max_value_y*0.2),
                        arrowprops=dict(facecolor='red',lw=1,shrinkA=0,shrinkB=0))
            

        plt.show()
        
    def raw_plot(self):
        
        plt.rcParams["font.size"] = 16
        
        fig =plt.figure(figsize=(8,6))
        ax=fig.add_subplot(1,1,1)
        ax.plot(self.xdata,self.ydata,"ro-",label='Data')
        
        ax.legend(loc='upper left')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]')
        max_value_y=max(self.ydata) 
        ax.set_ylim(0,max_value_y)
        
        if self.label != None:
            ax.annotate('Label:{:.2f}'.format(self.label), xy=(self.label, 0), 
                        xytext=(self.label,  max_value_y*0.2),
                        arrowprops=dict(facecolor='green',lw=1,shrinkA=0,shrinkB=0))
            

        plt.show()
        
        
    def fit_res_plot(self):
        
        plt.rcParams["font.size"] = 16
        fig =plt.figure(figsize=(20,8))
        ax=fig.add_subplot(1,3,1)

        ax.plot(self.xdata,self.ydata,"ro",label='Data')
        ax.plot(self.xdata,self.lsq_fit,"bo-",label='lsq')
        ax.plot(self.xdata,self.abs_fit,"go-",label='abs')
        ax.set_title('Fitting Plots')
        ax.legend(loc='lower right')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]')
        max_value_y=max(self.ydata) 
        ax.set_ylim(0,max_value_y)

        ax.annotate('Fit_lsq:{:.2f}'.format(self.lsq_para[0]), xy=(self.lsq_para[0], 0), 
                    xytext=(self.lsq_para[0],  max_value_y*0.4),
                    arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0))

        ax.annotate('Fit_abs:{:.2f}'.format(self.abs_para[0]), xy=(self.abs_para[0], 0), 
                    xytext=(self.abs_para[0],  max_value_y*0.6),
                    arrowprops=dict(facecolor='green',lw=1,shrinkA=0,shrinkB=0))

        #residual_plot
        ax2=fig.add_subplot(1,3,2)
        ax2.set_title('Residual Plots')
        ax2.plot(self.lsq_fit,self.lsq_fit-self.ydata,"bo-",label='lsq')
        ax2.plot(self.abs_fit,self.abs_fit-self.ydata,"go-",label='abs')
        ax2.legend(loc='lower right')
        ax2.set_xlabel('prediction')
        ax2.set_ylabel('prediction-observed')

        #yyplot
        ax3=fig.add_subplot(1,3,3)
        ax3.set_title('yy Plots')
    #     ydelta=ydata*0.42-ydata

        ax3.plot(self.ydata,self.ydata,"ro-",label='Data')
    #     ax3.plot(ydata,ydata-ydelta,"ro-")
    #     ax3.plot(ydata,ydata+ydelta,"ro-")
        ax3.plot(self.ydata,self.lsq_fit,"bo-",label='lsq')
        ax3.plot(self.ydata,self.abs_fit,"go-",label='abs')
        ax3.legend(loc='lower right')
        ax3.set_xlabel('observed')
        ax3.set_ylabel('prediction')

        plt.show()
