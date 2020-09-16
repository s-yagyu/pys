from pathlib import Path

import pandas as pd
import numpy as np

def read_pys_data(filename, pys_or_spys='spys'):
    """
    filename: input file path 
    pys_or_spys: original data -> 'pys' or 'spys'
    
    return : list data
    energy, spys
    
    example:
    xe,ypc= read_pys_data("au481_pys.csv",pys_or_spys='pys')
    
    """
    file = Path(filename)
    print("Input file path: ", filename)
    print("Basename: ", file.name)
    
    suffix_ = file.suffix.lstrip('.')
    
    if suffix_ == 'csv' :
        
        df_data= pd.read_csv(str(file)) 
        
    elif suffix_ == 'xlsx':
        
        df_data= pd.read_excel(str(file)) 
        
    else:
        print('Can not read.')
        pass
    
    
    if pys_or_spys == 'spys':
        
        return df_data.iloc[:,0].to_list(), df_data.iloc[:,1].to_list()

    elif pys_or_spys == 'pys':

        return df_data.iloc[:,0].to_list(), np.sqrt(df_data.iloc[:,1]).to_list()
    
    else: 
        pass