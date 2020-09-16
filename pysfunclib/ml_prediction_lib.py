from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from autoreglib import gridreg as gs

from pysfunclib import fowler_func as ff

class MLPredict():
    """
    example
        -------
        prd = mpl.MLPredict(path_name='./spys_reg_20200228_pure/')
        prd.param_load()
        # prd.sgb_xl4070_005
        prd.prediction(xx,yy)
    
    """
    
    def __init__(self, path_name):
        """
        path_name: str or Pathlib Path
                    set paramater dir
        """
        self.path_name = path_name
        
    def param_load(self):
    
        reg_dir_p = Path(self.path_name)
        reg_lists = list(reg_dir_p.glob("*.pkl"))
        
        for i in reg_lists:
            if 'sgb' in i.stem :
                # print("sgb: ",i.stem)
                sentencegb = 'self.{0}=joblib.load("{1}")'.format(i.stem,str(i))

                exec(sentencegb)
                
            elif 'srf' in i.stem:

                # print("srf: ", i.stem)
                sentencerf = 'self.{0}=joblib.load("{1}")'.format(i.stem,str(i))

                exec(sentencerf)
                
            else:
                print("pass: ",i.stem)
    
    def prediction(self,xdata,ydata):
        """
        xdata, ydata recommend list data
        np.array data is also possible
        
        return
        -----
        dict keys:'gb', 'rf' values: float
        pred['gb'] or pred['rf']
        """
        
        if str(type(ydata)) == "<class 'numpy.ndarray'>":
            ydata_list=ydata.flatten().tolist()
            xdata_list=xdata.flatten().tolist()
            self.xdata = xdata_list
            self.ydata = ydata_list
        else:
            self.xdata = xdata
            self.ydata = ydata
        
        if len(self.xdata) == len(self.ydata):
            stax = self.xdata[0]
            endx = self.xdata[-1]
            stepx = ff.my_round(self.xdata[1]-self.xdata[0], 2)
            self.range_index = 'xl{s:.0f}{e:.0f}_{ste:0=3}'.format(s=stax*10,
                                                                   e=endx*10,
                                                                   ste=int(stepx*100))
            print(self.range_index)
            ydata_ = np.array(self.ydata).reshape(1,-1)
            # print(ydata_)
            # ydata_=self.ydata
            try:
                sgb_reg = eval('self.sgb_'+ self.range_index)
                srf_reg = eval('self.srf_'+ self.range_index)
                # print(sgb_reg)
                predict_gb=sgb_reg.predict(ydata_)
                predict_rf=srf_reg.predict(ydata_)
                
            except:
                print("Prediction Error")
                # except ValueError:
                predict_gb = np.nan
                predict_rf = np.nan
                
        else :
            print('Prediction Error Check the data')
            predict_gb = np.nan
            predict_rf = np.nan
            
        self.predict_gb = float(predict_gb)
        self.predict_rf = float(predict_rf)
            
        return {'gb':self.predict_gb, 'rf': self.predict_rf}
    
    def plot(self):
        plt.rcParams["font.size"] = 16
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.xdata, self.ydata,"ro",label="Data")
        
        ax.legend(loc='lower right',fontsize=16)
        ax.set_xlabel('Energy [eV]',fontsize=16)
        ax.set_ylabel('PYS$^{1/2}$ [a.u.]',fontsize=16)
        max_value_y=max(self.ydata) 
        ax.set_ylim(0,max_value_y)

        ax.annotate('GB:{:.2f}'.format(self.predict_gb), xy=(self.predict_gb, 0), 
                    xytext=(self.predict_gb,  max_value_y*0.2),
                    arrowprops=dict(facecolor='red',lw=1,shrinkA=0,shrinkB=0),fontsize=16)

        ax.annotate('RF:{:.2f}'.format(self.predict_rf), xy=(self.predict_rf, 0), 
                    xytext=(self.predict_rf,  max_value_y*0.6),
                    arrowprops=dict(facecolor='blue',lw=1,shrinkA=0,shrinkB=0),fontsize=16)

        plt.show()
