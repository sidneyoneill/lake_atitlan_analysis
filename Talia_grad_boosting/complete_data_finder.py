# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:26:41 2025

@author: talia
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_existing_data(data_matrix, cols):
    n=0
    new_data =np.zeros((1,len(cols)))
    for i in range(len(data_matrix[:,1])):
        exists = True
        for j in cols:
            if pd.isna(data_matrix[i,j]):
                exists = False
        if exists:
            n+=1
            new_data = np.vstack([new_data, data_matrix[i,cols]])
    return new_data[1:,:]

def get_clean_lake_data(variables):
    contents = pd.read_csv('limno_set.csv', header=0, encoding = "cp850")
    contents = contents.sort_values(by='month_year')
    start_matrix = contents.iloc[:,:].values
    start_matrix = start_matrix[:,[0,1]+[i for i in range(6,20)]]
   	
    locs = {'WG', 'SA', 'WP'}
   	
    temp_depth = start_matrix[start_matrix[:,0]=='WG',:]
    for i in range(len(temp_depth[:,2])):
   		if not pd.isna(temp_depth[i,2]):
   			temp_depth[i,2] = round(temp_depth[i,2])
   	
    times = list(set(temp_depth[:,1]))
    data = np.zeros((len(times),len(variables)+1))
    data[:,0] = times
    surface = temp_depth[temp_depth[:,2]<11,:]
    for t in range(len(times)):
        temp_surface = surface[surface[:,1]==times[t],:]
        for i in range(len(variables)):
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
    complete_data = get_existing_data(data, [i for i in range(len(data[0]))]) 
    return complete_data

var = ['Location','Depth','Month','Temp','Chl-a','pH','Turbidity','DO','TDS','DBO','Secchi','NO3','PO4','NH4','PT','NT']
variables = [15]
data = get_clean_lake_data(variables)
dates = data[:,0]
months = dates%12 +1
years = (dates-dates%12)/12 +2014
string = 'Dates: '
pos = list(set(years))
pos.sort()
for year in pos:
    print(f'{int(year)}: {len(months[years[:]==year])} months')