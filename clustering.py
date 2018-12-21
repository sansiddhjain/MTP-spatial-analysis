import geopandas as gpd
import pandas as pd
from haversine import haversine
import math
from math import sqrt, log
import numpy as np
from numpy.random import random_sample
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
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

def KMeansCluster(attribute, dataframe, vectors, n_clusters):
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
	
	clusteringModel = KMeans(n_clusters=n_clusters).fit(vectors)
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
	agg_models = [AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage='complete') for i in range(2, 21)]
	avg_silhouette_scores = []
	for i in range(19):
		cluster_labels = agg_models[i].fit_predict(distanceMatrix)
		silhouette_avg = silhouette_score(vectors, cluster_labels)
		avg_silhouette_scores.append(silhouette_avg)
	print("Calculated silhouette scores")
	silhouette_matrix = np.vstack((np.asarray(list(range(2, 21))), np.asarray(avg_silhouette_scores))).T
	print(silhouette_matrix)
	plt.figure()
	plt.plot(silhouette_matrix[:, 0], silhouette_matrix[:, 1], '-')
	plt.xlabel('Number of Clusters')
	plt.ylabel('Avg Silhouette Score')
	plt.title(attribute)
	plt.show()
	if not(os.path.exists("Agglo/")):
		os.makedirs("Agglo/")
	# plt.savefig('Agglo/slhte-curve-'+attribute+'.png')
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
		print(i)

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
	print(("For n_clusters =", n_clusters,
		  "The average silhouette_score is :", silhouette_avg))

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
	print((str(n_clusters) + " done."))


# returns series of random values sampled between min and max values of passed col
def get_rand_data(col):
	rng = col.max() - col.min()
	return pd.Series(random_sample(len(col))*rng + col.min())

def iter_kmeans(df, n_clusters, num_iters=500):
	rng = list(range(1, num_iters + 1))
	vals = pd.Series(index=rng)
	for i in rng:
		k = KMeans(n_clusters=n_clusters, n_init=3)
		k.fit(df)
		# print "Ref k: %s" % k.get_params()['n_clusters']
		vals[i] = k.inertia_
	return vals

def iter_agglo(distanceMatrix, n_clusters, num_iters=500):
	rng = list(range(1, num_iters + 1))
	vals = pd.Series(index=rng)
	for i in rng:
		labels = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average').fit_predict(distanceMatrix)
		# if (i % 10 == 0):
		# 	print("random copies "+str(i)+" done.")
		# print "Ref k: %s" % k.get_params()['n_clusters']
		vals[i] = calculate_agglo_inertia(distanceMatrix, labels, n_clusters)
	return vals

def gap_statistic_kmeans(attribute, df, max_k=12):
	gaps = pd.Series(index = list(range(1, max_k + 1)))
	s = pd.Series(index = list(range(1, max_k + 1)))
	kmeans_inertia = pd.Series(index = list(range(1, max_k + 1)))
	final_test = pd.Series(index = list(range(1, max_k + 1)))
	for k in range(1, max_k + 1):
		km_act = KMeans(n_clusters=k, n_init=3)
		km_act.fit(df)
		kmeans_inertia[k] = km_act.inertia_
		# get ref dataset
		ref = df.apply(get_rand_data)
		ref_inertia_vec = np.log(np.asarray(iter_kmeans(ref, n_clusters=k)))
		ref_inertia = ref_inertia_vec.mean()
		s[k] = ref_inertia_vec.std()*(sqrt(1 + 1/k))
		gap = ref_inertia - log(km_act.inertia_)

		print("k: %s	Ref: %s   Act: %s  Gap: %s" % (k, ref_inertia, km_act.inertia_, gap))
		gaps[k] = gap

	# plt.figure()
	# plt.plot(range(1, max_k + 1), kmeans_inertia, '-')
	# plt.xlabel('# of clusters')
	# plt.ylabel('Inertia')
	# plt.title(attribute)
	# plt.show()
	
	b = False
	n_clusters_gap = 0
	for k in range(1, max_k):
		final_test[k] = gaps[k] - gaps[k+1] + s[k+1]
		print("k: %s	final_test val: %s 	" % (k, final_test[k]))
		if ((final_test[k] > 0) & (not b)):
			n_clusters_gap = k
			print("k: %s 	true" % (k))
			b = True
			break

	return gaps, n_clusters_gap

def calculate_agglo_inertia(distanceMatrix, labels, k):
	total_inertia = 0
# 	print(k)
# 	print(labels)
	for cluster_no in range(k):
		bool_arr = labels == cluster_no
# 		print(bool_arr)
		count = sum(bool_arr)
# 		print(count)
		submat = np.zeros((count, count))
		for idx in range(count):
			submat[idx, :] = distanceMatrix[bool_arr][idx][bool_arr]
		sumbat = np.triu(submat)
# 		print(submat)
		cluster_inertia = np.sum(np.sum(submat))
# 		print(cluster_inertia)
		total_inertia += cluster_inertia/count

	return total_inertia

def gap_statistic_agglo(attribute, df, distanceMatrix, max_k=12):
	gaps = pd.Series(index = list(range(2, max_k + 1)))
	s = pd.Series(index = list(range(2, max_k + 1)))
	final_test = pd.Series(index = list(range(2, max_k + 1)))
	for k in range(2, max_k + 1):
		labels = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average').fit_predict(distanceMatrix)
		inertia = calculate_agglo_inertia(distanceMatrix, labels, k)
		# print("starting reference computation")
		# get ref dataset
		ref = df.apply(get_rand_data)
# 		print(ref)
		distanceMatrix = computeDistanceMatrix(np.asarray(ref))
		ref_inertia_vec = np.log(np.asarray(iter_agglo(distanceMatrix, n_clusters=k)))
		ref_inertia = ref_inertia_vec.mean()
		s[k] = ref_inertia_vec.std()*(sqrt(1 + 1/k))
		# print("Reference Inertia - %s " % (ref_inertia))
		# print("Real Inertia - %s " % (inertia))
		gap = ref_inertia - log(inertia)

		print("k: %s	Ref: %s   Act: %s  Gap: %s" % (k, ref_inertia, inertia, gap))
		gaps[k] = gap
	
	b = False
	n_clusters_gap = 0
	for k in range(2, max_k):
		final_test[k] = gaps[k] - gaps[k+1] + s[k+1]
		print("k: %s	final_test val: %s 	" % (k, final_test[k]))
		if ((final_test[k] > 0) & (not b)):
			n_clusters_gap = k
			print("k: %s 	true" % (k))
			b = True
			break

	return gaps, n_clusters_gap

attributes = list(df_Spatial.columns[7:])

df = pd.read_csv('n_clusters-kmeans-gap-method-agg.csv')

if not os.path.exists("temp/cdf-clustering/"):
	os.makedirs("temp/cdf-clustering/")
os.chdir("temp/cdf-clustering/")

print(attributes)
n_clusters = np.zeros(len(attributes))
iterator = 0
# df = pd.read_csv('n_clusters-kmeans-gap-method.csv')
for attribute in attributes:
	
# attribute = sys.argv[1]
	df_distFinal, df_ShapeAttr, arr_ShapeAttr = preprocess(attribute)
	row = df.loc[df['attribute'] == attribute, :]
	no_of_clusters = int(row.loc[row.index[0], 'no_of_clusters'])
	print(no_of_clusters)
	kmeans_labels = KMeansCluster(attribute, df_ShapeAttr, arr_ShapeAttr, no_of_clusters)
	# distanceMatrix = computeDistanceMatrix(arr_ShapeAttr)
	# gaps, n_clusters[iterator] = gap_statistic_agglo(attribute, pd.DataFrame(data=arr_ShapeAttr), distanceMatrix)
	# gaps, n_clusters[iterator] = gap_statistic_kmeans(attribute, pd.DataFrame(data=arr_ShapeAttr))
	# iterator += 1
	# plot_sillhouette_graphs(attribute, arr_ShapeAttr, 2)
	# plot_sillhouette_graphs(attribute, arr_ShapeAttr, 3)
	# plot_sillhouette_graphs(attribute, arr_ShapeAttr, 4)
	# plot_sillhouette_graphs(attribute, arr_ShapeAttr, 5)
	# plot_sillhouette_graphs(attribute, arr_ShapeAttr, 6)
	# dbscan_labels = DBSCANCluster(attribute, df_ShapeAttr, arr_ShapeAttr)

	plot_graphs(attribute, 'KMeans', kmeans_labels, arr_ShapeAttr)
	# plot_graphs(attribute, 'DBSCAN', dbscan_labels, arr_ShapeAttr)

	# plot_tsne(attribute, arr_ShapeAttr, kmeans_labels, 'kmeans')
	# plot_tsne(attribute, arr_ShapeAttr, dbscan_labels, 'dbscan', perplexity=70)

	# distanceMatrix = computeDistanceMatrix(arr_ShapeAttr)
	# dtw_labels = AggCluster(attribute, df_ShapeAttr, arr_ShapeAttr, distanceMatrix)
	# plot_tsne(attribute, arr_ShapeAttr, dtw_labels, 'dtw', perplexity=80)
	# plot_graphs(attribute, 'Agglo', dtw_labels, arr_ShapeAttr)
	# gaps, n_clusters[iterator] = gap_statistic_agglo(attribute, pd.DataFrame(data=arr_ShapeAttr), distanceMatrix)

	print((attribute + "Done. "))

	# data = np.vstack((np.asarray(attributes), n_clusters)).T
	# df = pd.DataFrame(data = data, columns=['attribute', 'no_of_clusters'])
	# df.to_csv('n_clusters-kmeans-gap-method-'+itr+'.csv')
	# print("saved dataframe")