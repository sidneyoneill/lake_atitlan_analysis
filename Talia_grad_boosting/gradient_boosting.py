# -*- coding: utf-8 -*-
"""
Created 21:42 Sun 19 Jan 2025

@author: talia

Cleaning Lake Data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functions import load_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from functions import plot_results

def get_existing_data(data_matrix, cols):
	new_data =np.zeros((1,len(cols)))
	for i in range(len(data_matrix[:,1])):
		exists = True
		for j in cols:
			if pd.isna(data_matrix[i,j]):
				exists = False
			
		if exists:
			new_data = np.vstack([new_data, data_matrix[i,cols]])
			
	return new_data[1:,:]

def get_clean_lake_data(data_set,varis,loc=None,depth=None):
    
    start_matrix = data_set.iloc[:,:].values
    start_matrix = start_matrix[:,varis]
    depths = ['0-10 m', '10-30 m', '30+ m']
    locs = {'WG', 'SA', 'WP'}
    if loc == None:
        temp_depth = start_matrix[:,:]
    else:
        temp_depth = start_matrix[start_matrix[:,1]==loc,:]
    temp_depth=temp_depth[temp_depth[:,3]==depth,:]
    
    return temp_depth
	
    
def selecting_variables(temp_depth, variables):
    
    times = list(set(temp_depth[:,1])) #times is in column 1
    data = np.zeros((len(times),len(variables)+1)) #create matrix of zeros
    data[:,0] = times #first column is time
    surface = temp_depth[temp_depth[:,2]<11,:] #surface is any data above 10m
    for t in range(len(times)): #for every time
        temp_surface = surface[surface[:,1]==times[t],:] 
        #temp_surface is just surface values at that time
        
		
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

    use_data = complete_data[complete_data[:,0]>0,:]
    use_data_f = use_data[:,1:]

    return np.transpose(use_data_f.astype(float)), use_data[:,0]

def fill_sechi(data, ind=10):
    imp = data[:,[0,1,ind]]
    imp = imp[~pd.isna(imp[:,2]), :]
    for i in imp:
        data[np.all(data[:,:2]==i[:2],axis=1),ind] = i[2]
    return data



def gradient_boosting(X, y):
    split_index = int(0.7 * X.shape[0])
    
    # Split the matrix
    new_X, new_y = X[:split_index, :], y[:split_index]  # Rows 0 to split_index-1
    test_X, test_y = X[split_index:, :],y[split_index:]

    tscv = TimeSeriesSplit(n_splits=5)  # Define time series split with 5 folds
    rmse_opt = 900  # Set initial RMSE to a high value
    best_hp = None  # Store best hyperparameter
    best_model = None  # Store best model

    for hp in range(10, 100, 2):  # Iterate over max_depth values
        rmse_list = []

        for train_index, val_index in tscv.split(new_X):
            train_X, val_X = new_X[train_index], new_X[val_index]
            train_y, val_y = new_y[train_index], new_y[val_index]

            # Instantiate and train Gradient Boosting Regressor
            gbr = GradientBoostingRegressor(loss='absolute_error',
                                            learning_rate=0.1,
                                            n_estimators=100,
                                            max_depth=hp)
            gbr.fit(train_X, train_y)

            # Predict and compute RMSE
            y_pred = gbr.predict(val_X)
            rmse = root_mean_squared_error(val_y, y_pred)
            rmse_list.append(rmse)

        avg_rmse = np.mean(rmse_list)  # Compute mean RMSE over splits

        # Store best hyperparameter based on lowest RMSE
        if avg_rmse < rmse_opt:
            x_test_opt = test_X
            y_test_opt = test_y
            rmse_opt = avg_rmse
            best_hp = hp
            best_model = gbr
    return x_test_opt, y_test_opt, best_hp, best_model

def accuracy(test_y,pred_y):
    n = len(test_y)
    true_mean = np.sum(test_y)/n
    e = test_y - pred_y
<<<<<<< HEAD
    rmse = (np.sum(e**2)/n) ** (1/2)
=======
    mse = (np.sum(e**2)/n)
    try:
        mape = (100/n) * np.sum(abs((e)/test_y))
    except:
        mape = np.nan
>>>>>>> 5085d64bb69e700d878b307ef10936f263cc1b95
    mae = np.sum(abs(e))/n
    nse = 1 - (np.sum(e**2))/(np.sum((test_y-true_mean)**2))
    #nse = 1-(np.sum(e**2)/np.sum(test_y-true_mean)) #explained varience score
    
<<<<<<< HEAD
    return np.array([round(rmse,4),round(mae,2),round(nse,2)])
=======
    return np.array([mse,mape,mae,r2,evs])
>>>>>>> 5085d64bb69e700d878b307ef10936f263cc1b95


depth_group1 = ['Surface','Mid-Depth','Lower Photic','Deep']
varis1 = [2, 0,15,1]+[i for i in range(3,15)]
varis2 = [1,0,16,2]+[i for i in range(3,16)] 
depth_group2 = ['0-10m', '10-30m', '30m+']
all_data = False
#def begin(all_data=True):        
contents = pd.read_csv('SID_LIMNO_no_outliers.csv', header=0, encoding = "cp850")
contents['date'] = pd.to_datetime(contents['date'], format='mixed',dayfirst=True)
dates = contents['date']
contents['month_year'] = (dates.dt.year - 2014) * 12 + dates.dt.month
varis = varis1
accuracies = np.zeros(6)
for loc in ['WG']:#,'SA','WP']:
    for depth in depth_group2:
        print(depth)
        data_matrix = get_clean_lake_data(contents,varis,loc,depth)
        #print(data_matrix[0])
        data_matrix = fill_sechi(data_matrix)
        if all_data:
            data_matrix = get_existing_data(data_matrix, [i for i in range(len(data_matrix[0]))])
        dates = data_matrix[:,0]
        data_matrix = data_matrix[:,1:]
        #variables = [1,3,4,5,6,7,10]
        #data,time = selecting_variables(data_matrix, variables)
        #plt.plot(data_matrix[:,1],data_matrix[:,3])
        
        # df = load_data('lake_data_svd.xlsx')
        # dates = df['Fecha']
        # df['month_year'] = (dates.dt.year - 2014) * 12 + dates.dt.month
        # df = df.sort_values(by='month_year')
        # varis = [0,17,1]+[i for i in range(4,17)]
        # data_matrix2 = get_clean_lake_data(df,varis,'WG')
        # data_matrix2 = data_matrix2
        # surface2 = data_matrix2[data_matrix2[:,2]==0,:]
        # plt.plot(surface2[6:,1],surface2[6:,5])
        
        vals = [3,4,5,6,7,8]
        var = ['Temperature','Chl-a','pH','DO','Secchi','BDO']
        if all_data:
            vals = [i for i in range(3,len(data_matrix[0]))]
            var = ['temp','Chlorophyll Amount','pH','Dissolved Oxygen','secchi','BDO','TDS','turbidity',
                   'nitrate','phosphate','ammonium','phosphorus']
        accuracy_measures = np.array(['feature','depth','RMSE','MAE','NSE','hyper_param'])
        for n in [4,6]:#vals:
            vales = [1]+vals
            vales.remove(n)
            X, y = data_matrix[:,vales],data_matrix[:,n]
            test_X, test_y, hp, gbr = gradient_boosting(X,y)
            pred_y = gbr.predict(test_X)
            # test set accuracy measures | 
            accu = accuracy(test_y, pred_y)
            
            # Print rmse
            variable = var[n-3]
            #print(f'Root mean Square error {variable}: {accu[0]}')
            
            #plt.figure()
            #plt.plot(train_X[:,0],gbr.predict(train_X),'darkorange')
            #plt.plot(test_X[:,0],gbr.predict(test_X),'r')
            #plt.plot(surface2[:,1],surface2[:,4])
            #plt.plot(data_matrix[:,1],data_matrix[:,n])
            #plt.legend(['training data','testing data','exact'])
            #plt.figure()
            #plt.plot(test_X[:,0],pred_y)
            #plt.plot(test_X[:,0],test_y)
            #plt.legend(['prediction','exact'])
            #plt.xlabel('time')
            #plt.ylabel(var[n-3])
            accuracies = np.row_stack((accuracies,np.append(np.append([var[n-3],depth],accu),hp)))
            y_test_opt = test_y
            y_pred_opt = pred_y
            #print(accuracies)
            print(accuracies)
            plot_results(y, gbr.predict(X), int(len(dates)*0.7), dates, var[n-3])#[int(len(dates)*0.7):])
            test_train_split = int(len(dates)*0.7)
            test_val_split = int(test_train_split*0.7)
            #plt.axvline(x=dates[test_val_split], label= 'train-validation split', linestyle='dotted')
            #plt.axvline(x=dates[test_train_split], label= 'validation-test split',linestyle='dotted',color='r')
            #plt.legend()
accuracies = accuracies[1:]
accurate_table =  pd.DataFrame(accuracies, columns=accuracy_measures)
print(accurate_table)
#return accurate_table

#df =begin(False)
accurate_table.to_csv("grad_boost_output.csv", index=False)

print("DataFrame has been written to 'output.csv'")
# plt.figure()
# for i in [3,4,5,6,7,8]:
#     plt.plot(data_matrix[:,1],data_matrix[:,i])
# plt.legend(['temp','Chl-a','pH','DO','secchi','BDO'])

