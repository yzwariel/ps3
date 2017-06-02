import os
import pandas as pd

def read(file_name):
    '''
    Read/load data file that is put in the default folder "data"
    Input: 
        file_name(String)
    Return:
        pd dataframe
    '''
    file_path = 'data/'+ file_name
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".xls":
        df = pd.read_excel(file_path)
    else:
        raise ValueError('{file} is an unsupported file format.'.format(file=file_path))
    return df