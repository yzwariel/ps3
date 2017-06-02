import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def explore(df):
    '''
    Generate summary statisitcs and list of features(exclude dependent var)
    for the whole dataset
    Return:
        d: a dictionary contains summary statistics and list of feature for whole dataset
    '''
    d = {}
    summary = df.describe()
    features = list(df)[LEAD_VAR:]
    d["summary"] = summary
    d["features"] = features
    
    return d
    
def explore_var(df,var,graph_type):
    '''
    Generate distribution graph for specific variable
    Input:
        df: pd dataframe
        variable(string): the variable/attribute you want to explore
        graph_type(string): the type of graph you want to draw
    Return:
        d_var: a dictionary contains distribution for the selected attribute
        and the corresponding garph and feature list for that attribute. 
    '''
    d_var = {}
    cols = [var, DEP_VAR]
    var_mean = df[cols].groupby(var).mean()
    graph = var_mean.plot(kind=graph_type,use_index=False,figsize=(8,4))
    
    d_var["distribution"] = var_mean
    d_var["graph"] = graph
    
    return d_var


def plots(df):
    '''
    Takes:
        data, a pd.dataframe 

    Generates histograms in a separate folder
    '''
    print('Check the current folder for default histograms of these features.')
    for feature in df.keys():
        unique_vals = len(df[feature].value_counts())
        figure = plt.figure()
        if unique_vals == 1:
            df.groupby(feature).size().plot(kind='bar')
        elif unique_vals < 15:
            bins = unique_vals
            df[feature].hist(xlabelsize=10, ylabelsize=10, bins=unique_vals)
        else:
            df[feature].plot.hist()

        plt.ylabel('Frequency')
        plt.title('{}'.format(feature))
        plt.savefig('histograms/{}'.format(feature) + '_hist')
        plt.close()


