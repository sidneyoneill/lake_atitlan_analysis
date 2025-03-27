# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:56:05 2025

@author: talia
"""

import pandas as pd
import numpy as np

contents = pd.read_csv('atitlan_river_inflow.csv', header=0, encoding = "cp850")
contents = contents.sort_values(by='month_year')
start_matrix = contents.iloc[:,:].values

locations_list = ['Catarata','Francisco','Quiscab','Buenaventura','Tzunun']
def clean_river_data(data_matrix,locations_list):
	for i in range(len(data_matrix[:,6])): #Location
		for loc in range(len(locations_list)):
			if locations_list[loc] in data_matrix[i,6]:
				data_matrix[i,6] = loc
				break
			
	for i in range(len(data_matrix[:,16])): #Depth
		if not pd.isna(data_matrix[i,16]):
			try:
				data_matrix[i,16] = float(data_matrix[i,16])
			except:
				data_matrix[i,16] = 120
				
	for i in range(len(data_matrix[:,18])): #sediments
		if not pd.isna(data_matrix[i,18]):
			try:
				data_matrix[i,18] = float(data_matrix[i,18])
			except:
				data_matrix[i,18] = 0
	
	for i in range(len(data_matrix[:,27])): #Turbidity
		if not pd.isna(data_matrix[i,27]):
			try:
				data_matrix[i,27] = float(data_matrix[i,27])
			except:
				data_matrix[i,27] = 161
	
			
	for i in range(len(data_matrix[:,28])): #Fecal Coliforms
		if not pd.isna(data_matrix[i,28]):
			try:
				data_matrix[i,28] = float(data_matrix[i,28])
			except:
				data_matrix[i,28] = float(data_matrix[i,28][1:])
				
	for i in range(len(data_matrix[:,33])): #Water Quality
		if not pd.isna(data_matrix[i,33]):
			if data_matrix[i,33] == 'Mala':
				data_matrix[i,33] = 2
			elif data_matrix[i,33] == 'Regular':
				data_matrix[i,33] = 1
			else:
				data_matrix[i,33] = 3
				
	for i in range(len(data_matrix[:,29])): #BOD
		if not pd.isna(data_matrix[i,29]):
			try:
				data_matrix[i,29] = float(data_matrix[i,29])
			except:
				data_matrix[i,29] = np.nan
	return data_matrix

def get_existing_data(data_matrix, cols):
	new_data =np.zeros((1,len(cols)))
	for i in range(len(data_matrix[:,1])):
		exists = True
		for j in cols:
			if pd.isna(data_matrix[i,j]) or j==12 and data_matrix[i,j] == 0:
				exists = False
			
		if exists:
			new_data = np.vstack([new_data, data_matrix[i,cols]])
			
	return new_data[1:,:]	

def get_matrix_X():

	headings = ['month_year','Year','Month (num)','Month','Day','date','Location','Municipality',
	 'X','Y','Altitude','Time','Temperature','Ambient T','Temp change','pH','depth',
	 'Total_Dissolved_Solids','Sediments','O2','O2.1','Salinity','Conductivity','Flow',
	 'flow','phosphate','Nitrates','Turbidity','Fecal_Coliforms','BOD','Total_Nitrogen',
	 'Total_Phosphorus','ICA','Quality','Colour']
	selection = [0,6,10]+[i for i in range(12,32)]
	used_headings = [headings[i] for i in selection]
	cleaned_data = clean_river_data(start_matrix,locations_list)
	complete_matrix = get_existing_data(cleaned_data, selection)
	complete_matrix_F = complete_matrix[complete_matrix[:,1]==1,:]
	complete_matrix_Q = complete_matrix[complete_matrix[:,1]==2,:]
	complete_matrix_Q_n = np.transpose(complete_matrix_Q[:,[i for i in range(3,len(complete_matrix[1]))]])
	complete_matrix_F_n = np.transpose(complete_matrix_F[:,[i for i in range(3,len(complete_matrix[1]))]])
	complete_matrix_Q_n = complete_matrix_Q_n.astype(float)
	complete_matrix_F_n = complete_matrix_F_n.astype(float)
	
	final_month = complete_matrix_F[-1,0]
	
	clean_data = cleaned_data[:,selection]
	cleaned_data_new = clean_data[cleaned_data[:,0]>final_month,:]
	cleaned_data_new = cleaned_data_new[cleaned_data_new[:,0]<final_month+13,:]
	cleaned_data_new_F = cleaned_data_new[cleaned_data_new[:,1]==1,:]
	cleaned_data_new_F = np.transpose(cleaned_data_new_F.astype(float))
	final_month = complete_matrix_Q[-1,0]
	cleaned_data_new = clean_data[cleaned_data[:,0]>final_month,:]
	cleaned_data_new = cleaned_data_new[cleaned_data_new[:,0]<final_month+13,:]
	cleaned_data_new_Q = cleaned_data_new[cleaned_data_new[:,1]==1,:]
	cleaned_data_new_Q = np.transpose(cleaned_data_new_Q.astype(float))
	
	return complete_matrix_Q_n[:,2:], complete_matrix_F_n[:,2:],cleaned_data_new_Q,cleaned_data_new_F,used_headings