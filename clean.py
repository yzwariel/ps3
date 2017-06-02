import pandas as pd


def check_missing_data(df):
    """
    Calculates how many values for each variable is null for a given DataFrame
    
    """ 
    df_lng = pd.melt(df) #Convert to long data
    null_variables = df_lng.value.isnull()
    
    return pd.crosstab(df_lng.variable, null_variables)



#for c in proj.columns:
    #if proj[c].count() < len(proj):
        #missing_perc = ((len(proj) - proj[c].count()) / float(len(proj))) * 100.0
        #print("%.1f%% missing from: Column %s" %(missing_perc, c))


def clean(df,var,fill_method):
    '''
    In this assignment,this function is limited to fill in missing value for variables that have
    missing value showed in last function 'check_missing_data'.
    more filling method at: http://pandas.pydata.org/pandas-docs/stable/missing_data.html#cleaning-filling-missing-data
    Input:
        variable(string): the variable you want to fill in missing value (indicated in'check_missing_data' )
        fill_method(string): the method to fill in NA values. Avaliable methods are:
                             'drop', 'mean','zero',pad','ffill'.
    Return:
        df after cleaning
    '''
    if fill_method == "zero":
        df[var] = df[var].fillna(0)
    elif fill_method == "mean":
        df[var] = df[var].fillna(df[var].mean())
    elif fill_method == "drop":
        df[var] = df[var].dropna()
    elif fill_method == "pad" or fill_method == "ffill":
        df[var] = df[var].fillna(method=fill_method)
    else: 
        raise ValueError('{method} not avaliable'.format(method=fill_method))
    
    return df