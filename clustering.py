from __future__ import division
import geopandas as gpd
import pandas as pd
from haversine import haversine
import math
import numpy as np
from ggplot import *
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn import metrics
from scipy.spatial.distance import cdist, euclidean
import os
from fastdtw import fastdtw
import sys

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm


df_Spatial = pd.read_csv("data/Spatial2.csv")

# Takes input of attribute, and produces the vectors for clustering
def preprocess(attribute):
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
	df_ShapeAttr.reset_index(inplace=True)
	return df_distFinal, df_ShapeAttr, arr_ShapeAttr

def computeDistanceMatrix(vectors):
	
	distanceMatrix = np.zeros(( len(vectors), len(vectors) ))
	for i in range(len(vectors)):
		for j in range(len(vectors)):
			if i == j:
				continue
			distance, path = fastdtw(vectors[i, :], vectors[j, :], dist=euclidean)
			distanceMatrix[i, j] = distance
	return distanceMatrix

def KMeansCluster(attribute, dataframe, vectors):
	# print("Staring silhouette study")
	# agg_models = [KMeans(n_clusters=i) for i in range(2, 21)]
	
	# model_variances = [agg_models[i].fit(vectors).score(vectors) for i in range(19)]
	# elbow_matrix = np.vstack((np.asarray(range(2, 21)), np.asarray(model_variances))).T
	# plt.figure()
	# plt.plot(elbow_matrix[:, 0], elbow_matrix[:, 1], '-')
	# plt.xlabel('Number of Clusters')
	# plt.ylabel('Variance')
	# plt.title(attribute)
	# if not(os.path.exists("KMeans/Elbow/")):
	# 	os.makedirs("KMeans/Elbow/")
	# plt.savefig('KMeans/Elbow/elbow-curve-'+attribute+'.png')
	# plt.close()
	# np.savetxt(fname='KMeans/Elbow/elbow-values-'+attribute+'.txt', X=elbow_matrix, delimiter=' ', fmt="%f")
	
	# avg_silhouette_scores = []
	# for i in range(19):
	# 	cluster_labels = agg_models[i].fit_predict(vectors)
	# 	silhouette_avg = silhouette_score(vectors, cluster_labels)
	# 	avg_silhouette_scores.append(silhouette_avg)
	# print("Calculated silhouette scores")
	# silhouette_matrix = np.vstack((np.asarray(range(2, 21)), np.asarray(avg_silhouette_scores))).T
	# print(silhouette_matrix)
	# plt.figure()
	# plt.plot(silhouette_matrix[:, 0], silhouette_matrix[:, 1], '-')
	# plt.xlabel('Number of Clusters')
	# plt.ylabel('Avg Silhouette Score')
	# plt.title(attribute)
	# plt.savefig('KMeans/slhte-curve-'+attribute+'.png')
	# np.savetxt(fname='KMeans/slhte-values-'+attribute+'.txt', X=silhouette_matrix, delimiter=' ', fmt="%f")
	# print("Saved graph and matrix")
	# plt.close()
	
	clusteringModel = KMeans(n_clusters=2).fit(vectors)
	df_labels = pd.DataFrame(data={'District':dataframe['District'], 'labels':clusteringModel.labels_})
	if not(os.path.exists(attribute+"/KMeans/")):
		os.makedirs(attribute+"/KMeans/")
	df_labels.to_csv(attribute+"/KMeans/labels.csv")
	return df_labels

def DBSCANCluster(attribute, dataframe, vectors):
	clusteringModel = DBSCAN(eps=1).fit(vectors)
	df_labels = pd.DataFrame(data={'District':dataframe['District'], 'labels':clusteringModel.labels_})
	if not(os.path.exists(attribute+"/DBSCAN/")):
		os.makedirs(attribute+"/DBSCAN/")
	df_labels.to_csv(attribute+"/DBSCAN/labels.csv")
	return df_labels

def AggCluster(attribute, dataframe, vectors, distanceMatrix):
	print("Staring silhouette study")
	agg_models = [AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage='average') for i in range(2, 21)]
	avg_silhouette_scores = []
	for i in range(19):
		# print i+2
		cluster_labels = agg_models[i].fit_predict(distanceMatrix)
		# print cluster_labels
		silhouette_avg = silhouette_score(vectors, cluster_labels)
		# print silhouette_avg
		avg_silhouette_scores.append(silhouette_avg)
	print("Calculated silhouette scores")
	silhouette_matrix = np.vstack((np.asarray(range(2, 21)), np.asarray(avg_silhouette_scores))).T
	print(silhouette_matrix)
	plt.figure()
	plt.plot(silhouette_matrix[:, 0], silhouette_matrix[:, 1], '-')
	plt.xlabel('Number of Clusters')
	plt.ylabel('Avg Silhouette Score')
	plt.title(attribute)
	if not(os.path.exists("Agglo/")):
		os.makedirs("Agglo/")
	plt.savefig('Agglo/slhte-curve-'+attribute+'.png')
	np.savetxt(fname='Agglo/slhte-values-'+attribute+'.txt', X=silhouette_matrix, delimiter=' ', fmt="%f")
	print("Saved graph and matrix")
	plt.close()

	clusteringModel = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average').fit(distanceMatrix)
	df_labels = pd.DataFrame(data={'District':dataframe['District'], 'labels':clusteringModel.labels_})
	if not(os.path.exists(attribute+"/Agglo/")):
		os.makedirs(attribute+"/Agglo/")
	df_labels.to_csv(attribute+"/Agglo/labels.csv")
	return df_labels

def plot_tsne(attribute, vectors, df_labels, ctype, perplexity=30, early_exaggeration=12, n_components=2):
	vectors_embed = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration).fit_transform(vectors)
	plt.figure()
	plt.scatter(vectors_embed[:, 0], vectors_embed[:, 1], c=df_labels['labels'])
	plt.title('tsne for '+ctype)
	plt.savefig(attribute+"/tsne-"+str(ctype)+"-bins10-p"+str(perplexity)+"-ee"+str(early_exaggeration)+".png")

def plot_graphs(attribute, ctype, df_labels, vectors):
	for i in range(len(df_labels)):
		district = df_labels.loc[i, 'District']
		# print district
		label = df_labels.loc[i, 'labels']
		# print label
		matrix = np.vstack((0.1*np.arange(1.0, 11.0), np.asarray(vectors[i, :]))).T
		plt.figure()
		plt.plot(0.1*np.arange(1.0, 11.0), vectors[i, :], '-')
		plt.xlabel('Distance Bin - To')
		plt.ylabel(attribute)
		plt.title('Distance variation of '+attribute+' for district '+str(district))
		if not os.path.exists(attribute+'/'+ctype+'/plots/'+str(label)+'/'):
			os.makedirs(attribute+'/'+ctype+'/plots/'+str(label)+'/')
		plt.savefig(attribute+'/'+ctype+'/plots/'+str(label)+'/'+str(district)+'.png')
		plt.close()
		if not os.path.exists(attribute+'/'+ctype+'/data/'+str(label)+'/'):
			os.makedirs(attribute+'/'+ctype+'/data/'+str(label)+'/')
		np.savetxt(fname=attribute+'/'+ctype+'/data/'+str(label)+'/'+str(district)+'.txt', X=matrix, delimiter=' ', fmt="%f")
		print i

attributes = list(df_Spatial.columns[7:])
attributes.pop(2)
attributes.pop(5)
attributes = ["BF_ADV", "CHH_ADV"] + attributes

os.chdir("cdf-clustering/")

def plot_sillhouette_graphs(attribute, X, n_clusters):
	# Create a subplot with 1 row and 2 columns
	fig, (ax1) = plt.subplots(1, 1)
	fig.set_size_inches(18, 7)

	# The 1st subplot is the silhouette plot
	# The silhouette coefficient can range from -1, 1 but in this example all
	# lie within [-0.1, 1]
	ax1.set_xlim([-0.5, 1])
	# The (n_clusters+1)*10 is for inserting blank space between silhouette
	# plots of individual clusters, to demarcate them clearly.
	ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

	# Initialize the clusterer with n_clusters value and a random generator
	# seed of 10 for reproducibility.
	clusterer = KMeans(n_clusters=n_clusters, random_state=10)
	cluster_labels = clusterer.fit_predict(X)

	# The silhouette_score gives the average value for all the samples.
	# This gives a perspective into the density and separation of the formed
	# clusters
	silhouette_avg = silhouette_score(X, cluster_labels)
	print("For n_clusters =", n_clusters,
		  "The average silhouette_score is :", silhouette_avg)

	# Compute the silhouette scores for each sample
	sample_silhouette_values = silhouette_samples(X, cluster_labels)

	y_lower = 10
	for i in range(n_clusters):
		# Aggregate the silhouette scores for samples belonging to
		# cluster i, and sort them
		ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

		ith_cluster_silhouette_values.sort()

		size_cluster_i = ith_cluster_silhouette_values.shape[0]
		y_upper = y_lower + size_cluster_i

		color = cm.nipy_spectral(float(i) / n_clusters)
		ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

		# Label the silhouette plots with their cluster numbers at the middle
		ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

		# Compute the new y_lower for next plot
		y_lower = y_upper + 10  # 10 for the 0 samples

	ax1.set_title("The silhouette plot for the various clusters.")
	ax1.set_xlabel("The silhouette coefficient values")
	ax1.set_ylabel("Cluster label")

	# The vertical line for average silhouette score of all the values
	ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

	ax1.set_yticks([])  # Clear the yaxis labels / ticks
	ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

	# 2nd Plot showing the actual clusters formed
	# colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
	# ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

	# Labeling the clusters
	# centers = clusterer.cluster_centers_
	# Draw white circles at cluster centers
	# ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

	# for i, c in enumerate(centers):
	# 	ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

	# ax2.set_title("The visualization of the clustered data.")
	# ax2.set_xlabel("Feature space for the 1st feature")
	# ax2.set_ylabel("Feature space for the 2nd feature")

	plt.suptitle(("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
	plt.savefig(attribute+"/sillhouette-clusters-"+str(n_clusters)+".png")
	print str(n_clusters) + " done."

for attribute in attributes:
	
# attribute = sys.argv[1]
	df_distFinal, df_ShapeAttr, arr_ShapeAttr = preprocess(attribute)

	kmeans_labels = KMeansCluster(attribute, df_ShapeAttr, arr_ShapeAttr)
	plot_sillhouette_graphs(attribute, arr_ShapeAttr, 2)
	plot_sillhouette_graphs(attribute, arr_ShapeAttr, 3)
	plot_sillhouette_graphs(attribute, arr_ShapeAttr, 4)
	plot_sillhouette_graphs(attribute, arr_ShapeAttr, 5)
	plot_sillhouette_graphs(attribute, arr_ShapeAttr, 6)
	# dbscan_labels = DBSCANCluster(attribute, df_ShapeAttr, arr_ShapeAttr)

	plot_graphs(attribute, 'KMeans', kmeans_labels, arr_ShapeAttr)
	# plot_graphs(attribute, 'DBSCAN', dbscan_labels, arr_ShapeAttr)

	# plot_tsne(attribute, arr_ShapeAttr, kmeans_labels, 'kmeans')
	# plot_tsne(attribute, arr_ShapeAttr, dbscan_labels, 'dbscan', perplexity=70)

	# distanceMatrix = computeDistanceMatrix(arr_ShapeAttr)
	# dtw_labels = AggCluster(attribute, df_ShapeAttr, arr_ShapeAttr, distanceMatrix)
	# plot_tsne(attribute, arr_ShapeAttr, dtw_labels, 'dtw', perplexity=80)
	# plot_graphs(attribute, 'Agglo', dtw_labels, arr_ShapeAttr)

	print attribute + "Done. "


