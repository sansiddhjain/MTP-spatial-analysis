from __future__ import division
import geopandas as gpd
import pandas as pd
from haversine import haversine
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

####Cumulative  binning
# A function for creating bins. Bins are subsets of the interval [0, 1], with the least count of the size parameter. Bins can either be cumulative, not
# Cumulative example (0, 0.1, 0.05); (0, 0.2, 0.1); (0, 0.3, 0.15)... 
# Non-Cumulative example (0, 0.1, 0.05); (0.1, 0.2, 0.15); (0.2, 0.3, 0.25)... 
def binit(cumulative, start, end, size):
	df_temp=pd.DataFrame(columns=['From','To','Mid'])
	k=start
	i=0
	while k+size < end:
		if cumulative:
			df_temp.loc[i,'From']=0
		else:
			df_temp.loc[i,'From']=k
		df_temp.loc[i,'To']= k+size
		df_temp.loc[i,'Mid']=(k+size)/2
		k=k+size
		i=i+1
	return df_temp

### Make Bins
# df_bin=binit(True, 0, 1, 0.1)
# df_bin=binit(True, 0, 1, 1/59)
df_bin=binit(True, 0, 1, 1/59)

# Census data at a village level granularity
df_Spatial = pd.read_csv("village-level-metrics.csv")
# Take a slice of the data to deal with data from only 1 state
# df_Spatial = df_Spatial[df_Spatial["State"] == 32]

### Merging bins with districts
df_x= df_bin.copy(deep=True)
df_y= pd.DataFrame(df_Spatial['District'].unique())
df_x['key']=1
df_y['key']=1
df_distFinal=df_x.merge(df_y, on='key')
df_distFinal.rename(columns={0:'District'}, inplace=True)
del df_x, df_y, df_distFinal['key']
print(df_distFinal)

df_distFinal['Floor'] = np.zeros(df_distFinal.shape[0])
df_distFinal['Top'] = np.zeros(df_distFinal.shape[0])
for i in range(len(df_distFinal)):
	vildist_temp=df_Spatial.loc[df_Spatial['District']==df_distFinal.loc[i,'District'],'Dist']
	floor = vildist_temp.quantile(df_distFinal.loc[i,'From'])
	top = vildist_temp.quantile(df_distFinal.loc[i,'To'])
	df_distFinal.loc[i,'Floor']= floor
	df_distFinal.loc[i,'Top']= top

df_temp = df_Spatial.loc[:, :'District'].groupby('District').mean()
df_temp.reset_index(inplace=True)
df_distFinal = df_distFinal.merge(df_temp, on="District")

df_distFinal = pd.read_csv('spatial-cdf-raw-values.csv')
del df_distFinal['Unnamed: 0']

# attributes = ["BF", "CHH", "FC", "MSW", "MSL"]
attributes = ["MSL"]
for attribute in attributes:
	levels = np.sort(df_Spatial["Village_HHD_Cluster_"+attribute].unique())
	print(levels)
	if attribute == "EMP":
		subscript_dict = {0 : "UN", 1 : "AL", 2 : "NAL"}
	else:
		subscript_dict = {0 : "RUD", 1 : "INT", 2 : "ADV"}

	for j in ["RUD", "INT", "ADV"]:
		# if j>= 3:
		# 	continue

		### Fill the final data frame 

		# df_distFinal[attribute+"_"+subscript_dict[j]] = np.zeros(df_distFinal.shape[0])
		# for i in range(len(df_distFinal)):
		# 	floor = df_distFinal.loc[i,'Floor']
		# 	top = df_distFinal.loc[i,'Top']
		# 	df_temp = df_Spatial.loc[(df_Spatial['District']==df_distFinal.loc[i,'District']) & (df_Spatial['Dist']>=floor) & (df_Spatial['Dist']<=top),:]
		# 	if ((attribute == "CHH") and (j == 2)):
		# 		att_villages = np.sum(np.logical_or((df_temp["Village_HHD_Cluster_"+attribute] == levels[j]), (df_temp["Village_HHD_Cluster_"+attribute] == levels[j+1])))
		# 		total_villages = (df_temp.shape[0])
		# 		val = np.sum(np.logical_or((df_temp["Village_HHD_Cluster_"+attribute] == levels[j]), (df_temp["Village_HHD_Cluster_"+attribute] == levels[j+1])))/(df_temp.shape[0])
		# 	else:
		# 		att_villages = np.sum(df_temp["Village_HHD_Cluster_"+attribute] == levels[j])
		# 		total_villages = (df_temp.shape[0])
		# 		val = (np.sum(df_temp["Village_HHD_Cluster_"+attribute] == levels[j])*100)/(df_temp.shape[0])
		# 	# print(str(val))
		# 	df_distFinal.loc[i, attribute+"_"+subscript_dict[j]] = val

		for i in range(len(df_distFinal)):
			floor = df_distFinal.loc[i,'Floor']
			top = df_distFinal.loc[i,'Top']
			df_temp = df_Spatial.loc[(df_Spatial['District']==df_distFinal.loc[i,'District']) & (df_Spatial['Dist']>=floor) & (df_Spatial['Dist']<=top),:]
			if (attribute == "CHH"):
				if j == 'RUD':
					val = np.mean(df_temp[attribute+"_ADV"])
				if j == 'ADV':
					val = np.mean(df_temp[attribute+"_RUD"])
			if (attribute == 'MSL') & (j == 'RUD'):
				val = np.mean(100 - df_temp["MSL_ADV"] - df_temp["MSL_INT"])
			else:
				val = np.mean(df_temp[attribute+"_"+j])

			df_distFinal.loc[i, attribute+"_"+j] = val
		

	df_distFinal.to_csv("spatial-cdf-raw-values.csv")

# df = pd.read_csv('spatial-cdf-verbose.csv')
# del df['Unnamed: 0']
# df['To'] = pd.Series(list(map(lambda x: round(x, 1), df['To'])))
# df['Mid'] = pd.Series(list(map(lambda x: round(x, 2), df['Mid'])))

# att_list = []
# for main_att in ["BF", "CHH", "FC", "MSW", "MSL", "EMP"]:
#     if main_att != 'EMP':
#         att_list.append(main_att+'_RUD')
#         att_list.append(main_att+'_INT')
#         att_list.append(main_att+'_ADV')
#     else:
#         att_list.append(main_att+'_UN')
#         att_list.append(main_att+'_AL')
#         att_list.append(main_att+'_NAL') 

# iterables = [att_list, ['vill%_s', 'vill%_f', 'distance']]
# columns = pd.MultiIndex.from_product(iterables, names=['attribute', 'salient_info'])
# df_metadata = pd.DataFrame(index = pd.unique(df['District']), columns = columns)

# districts = pd.unique(df['District'])

# for distr in districts:
#     for attribute in att_list:
#         row = df[np.logical_and(df['District'] == distr, df['To'] == 0.1)]
#         df_metadata.loc[distr, (attribute, 'vill%_s')] = round(row.loc[row.index[0], attribute], 2)
#         row = df[np.logical_and(df['District'] == distr, df['To'] == 1)]
#         df_metadata.loc[distr, (attribute, 'vill%_f')] = round(row.loc[row.index[0], attribute], 2)
#         df_metadata.loc[distr, (attribute, 'distance')] = round(row.loc[row.index[0], 'Top'], 2)

#     print('district '+str(distr)+ ' done.')

# df_metadata.to_csv('imp-metadata.csv')