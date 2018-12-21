import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy.random import random_sample
from math import sqrt, log

from sklearn.datasets import load_iris

# famous iris data set
iris = load_iris()
iris_data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_target = iris['target']


# returns series of random values sampled between min and max values of passed col
def get_rand_data(col):
	rng = col.max() - col.min()
	return pd.Series(random_sample(len(col))*rng + col.min())

def iter_kmeans(df, n_clusters, num_iters=30):
	rng = range(1, num_iters + 1)
	vals = pd.Series(index=rng)
	for i in rng:
		k = KMeans(n_clusters=n_clusters, n_init=3)
		k.fit(df)
		# print "Ref k: %s" % k.get_params()['n_clusters']
		vals[i] = k.inertia_
	return vals

def gap_statistic(df, max_k=21):
	gaps = pd.Series(index = range(1, max_k + 1))
	s = pd.Series(index = range(1, max_k + 1))
	final_test = pd.Series(index = range(1, max_k + 1))
	for k in range(1, max_k + 1):
		km_act = KMeans(n_clusters=k, n_init=3)
		km_act.fit(df)

		# get ref dataset
		ref = df.apply(get_rand_data)
		ref_inertia_vec = np.log(np.asarray(iter_kmeans(ref, n_clusters=k)))
		ref_inertia = ref_inertia_vec.mean()
		s[k] = ref_inertia_vec.std()*(sqrt(1 + 1/k))
		gap = ref_inertia - log(km_act.inertia_)

		print "k: %s	Ref: %s   Act: %s  Gap: %s" % (k, ref_inertia, km_act.inertia_, gap)
		gaps[k] = gap

	for k in range(1, max_k):
		final_test[k] = gaps[k] - gaps[k+1] + s[k+1]
		print "k: %s	final_test val: %s 	" % (k, final_test[k])
	return gaps