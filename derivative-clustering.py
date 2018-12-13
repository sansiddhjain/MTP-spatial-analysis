from __future__ import division
# import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


df_Spatial = pd.read_csv("data/Spatial2.csv")

# Takes input of attribute, and produces the vectors for clustering
attribute = sys.argv[1]
# Normalising sequence of vectors such that mean value is 0, max is 1
df_distFinal = df_Spatial.loc[:, ['From', 'To', 'Mid', 'District', attribute]]
for i in range(len(df_distFinal)):
	df_temp=df_distFinal.loc[df_distFinal['District']==df_distFinal.loc[i,'District'], attribute]
	df_distFinal.loc[i,attribute+'_shape']=0
	if ((df_temp.max()-df_temp.min()) > 0):
		df_distFinal.loc[i,attribute+'_shape']=(df_distFinal.loc[i, attribute]-df_temp.min())/(df_temp.max()-df_temp.min())

df_ShapeAttr=df_distFinal.pivot(index='District', columns='To')[attribute+'_shape']

# Normalising across every dimension (making variance for all same)
for i in list(df_ShapeAttr):
    std=df_ShapeAttr[i].std()
    df_ShapeAttr[i]=df_ShapeAttr[i]/std

# Extracting only the data out of the dataframe
arr_ShapeAttr=np.array(df_ShapeAttr)

# df.apply(lambda row: func(row), axis=1)
df_ShapeAttr['derivative_sum'] = df_ShapeAttr.apply( lambda row: np.sum(np.ediff1d(row)), axis = 1 )
df_ShapeAttr['label'] = df_ShapeAttr.apply( lambda row: int(row['derivative_sum'] >= 0), axis = 1 )
df_ShapeAttr.columns = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', 'derivative_sum', 'label']
if not os.path.exists('derivative-clustering/'+attribute+'/'):
	os.makedirs('derivative-clustering/'+attribute+'/')
df_ShapeAttr['label'].to_csv('derivative-clustering/'+attribute+'/labels.csv')

print("preprocessing done. Starting plotting")

for i in range(len(df_ShapeAttr)):
	district = df_ShapeAttr.index[i]
	print(district)
	label = df_ShapeAttr.loc[district, 'label']
	print(label)
	plt.figure()
	sliced = df_ShapeAttr.loc[district, '0.1':'1.0']
	plt.plot(sliced.index, sliced, '-')
	plt.xlabel('Distance Bin - To')
	plt.ylabel(attribute)
	plt.title('Distance variation of '+attribute+' for district '+str(district))
	if not os.path.exists('derivative-clustering/'+attribute+'/plots/'+str(label)+'/'):
		os.makedirs('derivative-clustering/'+attribute+'/plots/'+str(label)+'/')
	plt.savefig('derivative-clustering/'+attribute+'/plots/'+str(label)+'/'+str(district)+'.png')
	plt.close()
	print(i)

