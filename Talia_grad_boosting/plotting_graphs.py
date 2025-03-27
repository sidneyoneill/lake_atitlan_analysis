# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:19:21 2025

@author: talia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from functions import load_data

def get_clean_lake_data(data_set,varis,loc=None):
    
    start_matrix = data_set.iloc[:,:].values
    start_matrix = start_matrix[:,varis]
	
    locs = {'WG', 'SA', 'WP'}
    if loc == None:
        temp_depth = start_matrix[:,:]
    else:
        temp_depth = start_matrix[start_matrix[:,0]=='WG',:]
    for i in range(len(temp_depth[:,2])):
        if not pd.isna(temp_depth[i,2]):
            try:
                temp_depth[i,2] = round(temp_depth[i,2])
            except:
                try:
                    temp_depth[i,2] = int(temp_depth[i,2][:2])
                except:
                    temp_depth[i,2] = int(temp_depth[i,2][:1])
    
    return temp_depth


def selecting_variables(temp_depth, variables,d):
    
    times = list(set(temp_depth[:,1])) #times is in column 1
    data = np.zeros((len(times),len(variables)+1)) #create matrix of zeros
    data[:,0] = times #second column is time
    if d == 0:
        surface = temp_depth[temp_depth[:,2]<11,:] #surface is any data above 10m
    elif d == 10:
        surface = temp_depth[temp_depth[:,2]>=11,:]
        surface = temp_depth[temp_depth[:,2]<31,:]
    else:
        surface = temp_depth[temp_depth[:,2]>30,:]
    for t in range(len(times)): #for every time
        temp_surface = surface[surface[:,1]==times[t],:] 
        #temp_surface is just surface values at that time
        
        for i in range(len(variables[1:])):
            total = 0
            num = 0
            for j in range(len(temp_surface[:,variables[i]])):
                if not pd.isna(temp_surface[j,variables[i]]):
                    total += temp_surface[j,variables[i]]
                    num += 1
            if num > 0:
                data[t,i+1] = total/num
            else:
                data[t,i+1] = np.nan
    return data, times

def fill_sechi(data, ind=10):
    imp = data[:,[0,1,ind]]
    imp = imp[~pd.isna(imp[:,2]), :]
    for i in imp:
        data[np.all(data[:,:2]==i[:2],axis=1),ind] = i[2]
    return data

contents = pd.read_csv('limno_set.csv', header=0, encoding = "cp850")
contents = contents.sort_values(by='month_year')

df = load_data('lake_data_final.xlsx')
dates = df['Fecha']
df['month_year'] = (dates.dt.year - 2014) * 12 + dates.dt.month
df=df.sort_values(by='month_year')

locs = ['WG', 'SA', 'WP']
depths = [0,10,30]
varis = [0,1]+[i for i in range(6,20)]
varis2 = [0,17,1]+[i for i in range(4,17)]

var = ['Depth','Month','Temp','Chl-a','pH','Turbidity','DO','TDS','DBO','Sechi','NO3','PO4','NH4','PT','NT']

for loc in locs:
    svd_data = get_clean_lake_data(df, varis2, loc)
    data_matrix = get_clean_lake_data(contents,varis,loc)
    print(data_matrix[0])
    data_matrix = fill_sechi(data_matrix)
    print(data_matrix[0])
    for depth in depths:
        actual_data, times = selecting_variables(data_matrix, np.arange(1,16), depth)
        #exp_data = svd_data[svd_data[:,2]==depth,:]
        for v in range(len(var[2:])):
            #plt.figure()
            #plt.plot(exp_data[:,1],exp_data[:,v+3])
            plt.plot(times,actual_data[:,v+3])
            
            plt.title(var[v+2]+"/"+str(depth)+"/"+loc)
            plt.xlabel('months')
            plt.ylabel(var[v+2])
            plt.legend(['svd_prediction','actual'])