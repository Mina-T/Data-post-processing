import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

def read_metrics(path):
    '''
    Given a path, converts the metrics.dat into pd.df
    '''
    df= pd.read_csv(path + '/metrics.dat', delim_whitespace = True )
    df = df.groupby(['#epoch'], as_index=False).last()
    return df

def read_dat(path, _file):
    '''
    Given a path, converts the *.dat into pd.df
    '''
    if path[-1] != '/':
        path = path + '/'
    df= pd.read_csv(path + _file, delim_whitespace = True )
    return df
