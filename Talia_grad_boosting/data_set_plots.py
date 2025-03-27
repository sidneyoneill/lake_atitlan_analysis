# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:48:34 2025

@author: talia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_clean_lake_data(data_set,varis,loc=None,depth=None):
    
    start_matrix = data_set.iloc[:,:].values
    start_matrix = start_matrix[:,varis]
    depths = ('Surface','Mid-Depth','Lower Photic','Deep')
    locs = {'WG', 'SA', 'WP'}
    if loc == None:
        temp_depth = start_matrix[:,:]
    else:
        temp_depth = start_matrix[start_matrix[:,1]==loc,:]
    temp_depth=temp_depth[temp_depth[:,2]==depth,:]
    
    return temp_depth

def plot_graph(svd_matrix,feature_matrix,i,features):
    plt.figure()
    plt.plot(svd_matrix[:,0],svd_matrix[:,i])
    plt.plot(feature_matrix[:,0],feature_matrix[:,i])
    plt.ylabel(features[i])
    plt.xlabel('Date')
    plt.legend(['svd','feature'])

def accuracy(test_y,pred_y):
    n = len(test_y)
    true_mean = np.sum(test_y)/n
    e = test_y - pred_y
    mse = (np.sum(e**2)/n) 
    try:
        mape = (100/n) * np.sum(abs((e)/test_y))
    except:
        mape = np.nan
    mae = np.sum(abs(e))/n
    r2 = 1 - (np.sum(e**2))/(np.sum((test_y-true_mean)**2))
    evs = 1-(np.var(e)/np.var(test_y)) #explained varience score
    
    return np.array([round(mse,2),round(mape,2),round(mae,2),round(r2,2),round(evs,2)])

feature = pd.read_csv('SID_LIMNO_processed_V2.csv', header=0, encoding = "cp850")
feature['date'] = pd.to_datetime(feature['date'], format='mixed',dayfirst=True)
feature_dates = feature['date']
feature_sort = [2,0,1]+[i for i in range(3,len(list(feature.columns.values)))]

svd = pd.read_csv('Lake_data_clean_final.csv', header=0, encoding = "cp850")
svd['Fecha'] = pd.to_datetime(svd['Fecha'], format='mixed',dayfirst=True)
svd_dates = svd['Fecha']
svd_sort = [1,0,2,3,4,5,7,10,9,8,6,11,12,13,14]

svd_matrix = get_clean_lake_data(svd,svd_sort,'WG','0-10 m')
feature_matrix = get_clean_lake_data(feature,feature_sort,'WG','Surface')

features = list(feature.columns.values)
accuracies = np.zeros(5)
for i in [i for i in range(3,len(list(feature.columns.values)))]:
    svd_data = svd_matrix[svd_matrix[:,0]>=feature_matrix[0,0],i]
    feature_data = feature_matrix[feature_matrix[:,0]>=svd_matrix[0,0],i]
    while pd.isna(feature_data[0]) or pd.isna(svd_data[0]):
        feature_data = feature_data[1:]
        svd_data = svd_data[1:]
    accuracies = np.row_stack((accuracies,accuracy(svd_data,feature_data)))
accuracies = accuracies[1:]
accuracy_measures = np.array(['MSE','MAPE','MAE','R2','EVS'])
accurate_table =  pd.DataFrame(accuracies, columns=accuracy_measures, index=features[3:])
print(accurate_table)

