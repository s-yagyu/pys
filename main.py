from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import PySimpleGUI as sg
import numpy as np


#オリジナルモジュールの読み込み
from pysfunclib import fowler_func as ff
from pysfunclib import fowler_func_opti as ffo
from pysfunclib import fit_prediction_lib as fpl
from pysfunclib import ml_prediction_lib as mpl
from pysfunclib import data_read as dr

#機械学習の自動化とプロットのモジュールの読み込み
from autoreglib import gridreg as gs


def draw_plot(x,y):
    plt.plot(x,y)
    plt.show(block=False)
    #block=Falseの指定をしないと、その間コンソールは何も入力を受け付けなくなり、GUI を閉じないと作業復帰できない。

def check_file(file_name):
    p = Path(file_name)
    # print(p.suffix)
    if p.suffix == '.csv':
        df = pd.read_csv(p) 
        x = df.iloc[:,0].values
        y = df.iloc[:,1].values

        return x, y

    else:
        print('Wrong data file, data must be CSV')  
        return None, None

def main():
    
    # 識別パラメータの入っているパスを指定してインスタンスを作成
    prd=mpl.MLPredict(path_name='./spys_reg_20200623/')
    # 識別パラメーターの読み込み
    prd.param_load()
    
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('Enter csv data file')],
            [sg.Text('File', size=(8, 1)),sg.Input(key='-file_name-'), sg.FileBrowse()],
            [sg.Radio('PYS',group_id='g1',default=True,key='-pys-'),
             sg.Radio('PYS1/2',group_id='g1',default=False,key='-spys-')],
            [sg.Submit()],
            [sg.Button('Plot'), sg.Button ('Predict')], 
            [sg.Cancel()],
            [sg.Text('Set ML paramater holder'), sg.InputText(default_text='./spys_reg_20200623/',key='-param_holder-')],
            [sg.Button('ParamReSet')]]

    window = sg.Window('Plot', layout)

    while True:
        event, values = window.read()

        if event in (None, 'Cancel'):
            break
        elif event in 'Submit':
            print('File name:{}'.format(values['-file_name-']))
            x,y = check_file(values['-file_name-'])
            if x[0] == None:
                sg.popup('Set file is not CSV')
                
            if values['-pys-'] == True:
                y = np.sqrt(y)
       
        elif event == 'Plot':
            draw_plot(x,y)
            
        elif event == 'Predict':
            # データを与えて予測を行う
            prd.prediction(x,y)
            
            # データと予測結果の図示
            prd.plot()

        elif event == 'ParamReSet':
            # 識別パラメータの入っているパスを指定してインスタンスを作成
            pathname = values['-param_holder-']
            prd=mpl.MLPredict(path_name=pathname)
            # 識別パラメーターの読み込み
            prd.param_load()
            
    window.close()
    
if __name__ =='__main__':
    main()
