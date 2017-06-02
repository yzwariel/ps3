import pandas as pd
import numpy as np
import math
import pylab as pl
from sklearn.ensemble import RandomForestClassifier


def find_features(df, features):
    '''
    Use scikit-learn lib to determine which variables are the best at predicting risk.
    Then, from the calculated importances, order them from most to least important
    and make a barplot to visualize what is/isn't important
    '''
    clf = RandomForestClassifier()
    clf.fit(df[features], df[DEP_VAR])
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")



def binning(df, feature, type_cut, quantiles = 0.5, bins = 1):
    '''
    Takes:
        feature, a string with the name of the feature to put into bins
        type_cut, quantiles or bins
        quantiles, an int or array of quantiles
        bins, an int or sequence of scalars
    '''
    valid_cuts = ['quantiles', 'bins']
    assert type_cut in valid_cuts

    bins = 'bins_{}'.format(feature)
    if type_cut == 'quantiles':
        df[bins] = pd.qcut(df[feature], quantiles, labels=False)
    elif type_cut == 'n':
        df[bins] = pd.cut(df[feature], bins, labels=False)


def binarize(df, feature, control):
    '''
    Takes:
        feature, a string of the name of the feature to binarize
        control, the value of a feature that will have a zero value in the
        new binarized feature 
    '''
    df[feature] = df[feature].apply(lambda x: 0 if x == control else 1)


def generate_bins(df,var,size):
    '''
    Generate a list of bin boundary for categorical var
    
    '''
    lb = int(math.floor(df[var].describe()[3]))
    lb2 = int(math.floor(df[var].describe()[4]))
    ub = int(math.ceil(df[var].describe()[7]))
    ub2_temp = int(math.ceil(df[var].describe()[6]))
    bins = int(math.ceil((ub2_temp-lb2)/size))
    ub2 = int(lb2+size*(bins))
    bins = [lb] + range(lb2,ub2+size,size) + [ub]
    return bins

def build_category(df,var,bins):
    '''
    Discretize a continous variable
    '''
    new_name = var + '_bucket'
    df[new_name] = pd.cut(df[var], bins, labels=False,include_lowest=True)
    return df

def create_dummy(df,var):
    '''
    Take categorical var and create binary/dummy variables from it
    '''
    dummy_df = pd.get_dummies(df[var],prefix=var)
    return dummy_df