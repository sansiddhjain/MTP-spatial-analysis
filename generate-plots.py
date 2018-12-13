import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import os

os.chdir('cdf-clustering/Agglo/')

f = [j for j in os.listdir(os.getcwd()) if '.txt' in j]
X = np.zeros((19, 2))
counter = 0
plt.figure()
plt.xlabel('Number of Clusters')
plt.ylabel('Avg Silhouette Score')
plt.title('Avg Silhouette Scores for DTW Clustering')
for file in f:
	attribute = file.split('slhte-values-')[1].split('.txt')[0]
	mat = np.genfromtxt(file, delimiter=' ')
	plt.plot(mat[:, 0], mat[:, 1], '-', label=attribute)
	X = X + mat
	counter += 1
plt.legend(loc='best')
plt.savefig('slhte-values-agglo-var-att.png')
plt.close()

X = X / counter
print X

os.chdir('../KMeans/')

f = [j for j in os.listdir(os.getcwd()) if '.txt' in j]
Y = np.zeros((19, 2))
counter = 0
plt.figure()
plt.xlabel('Number of Clusters')
plt.ylabel('Avg Silhouette Score')
plt.title('Avg Silhouette Scores for KMeans Clustering')
for file in f:
	attribute = file.split('slhte-values-')[1].split('.txt')[0]
	mat = np.genfromtxt(file, delimiter=' ')
	plt.plot(mat[:, 0], mat[:, 1], '-', label=attribute)
	Y = Y + mat
	counter += 1
plt.legend(loc='best')
plt.savefig('slhte-values-kmeans-var-att.png')
plt.close()

Y = Y / counter
print Y

plt.figure()
plt.xlabel('Number of Clusters')
plt.ylabel('Avg Silhouette Score')
plt.title('Avg Avg Silhouette Scores - KMeans vs DTW')
plt.plot(X[:, 0], X[:, 1], '-', label='DTW')
plt.plot(Y[:, 0], Y[:, 1], '-', label='KMeans')
plt.legend(loc='best')
plt.savefig('../slhte-values-kmeans-vs-agglo.png')
plt.close()

os.chdir('Elbow/')

f = [j for j in os.listdir(os.getcwd()) if '.txt' in j]
Y = np.zeros((19, 2))
counter = 0
plt.figure()
plt.xlabel('Number of Clusters')
plt.ylabel('Variance')
plt.title('Elbow Curves for KMeans Clustering')
for file in f:
	attribute = file.split('elbow-values-')[1].split('.txt')[0]
	mat = np.genfromtxt(file, delimiter=' ')
	plt.plot(mat[:, 0], mat[:, 1], '-', label=attribute)
	Y = Y + mat
	counter += 1
plt.legend(loc='best')
plt.savefig('all-elbow-curves.png')
plt.close()

Y = Y / counter
print Y

plt.figure()
plt.xlabel('Number of Clusters')
plt.ylabel('Variance')
plt.title('Average Elbow Curve Across All Clusters')
plt.plot(Y[:, 0], Y[:, 1], '-')
# plt.legend(loc='best')
plt.savefig('avg-elbow-curve.png')
plt.close()

