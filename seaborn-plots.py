import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os

sns.set(style="darkgrid")

attributes = pd.read_csv("attributes.csv").index
print(attributes)
for attribute in attributes:
	f = os.listdir("temp/cdf-clustering/"+attribute[1:-1]+"/KMeans/data/")
	os.chdir("temp/cdf-clustering/"+attribute[1:-1]+"/KMeans/data/")

	df = pd.DataFrame(columns=["percentile", "value", "cluster", "district"])

	for cluster in f:
		for file in os.listdir(cluster+"/"):
			data = np.genfromtxt(cluster+"/"+file, dtype=float, delimiter=' ')
			df1 = pd.DataFrame(data=data, columns=["percentile", "value"])
			df1['cluster'] = int(cluster)
			df1['district'] = int(file.split('.')[0])
			df = df.append(df1)
		print("cluster "+ cluster + " done. ")

	os.chdir("../../../../../")
	if not os.path.exists("seaborn/plots/KMeans-ci/"):
		os.makedirs("seaborn/plots/KMeans-ci/")
	df.to_csv("seaborn/"+attribute[1:-1].lower()+".csv")

	print(len(f))
	plot = sns.lineplot(x="percentile", y="value", hue="cluster", palette=sns.color_palette("hls", len(f)), data=df)
	fig = plot.get_figure()
	fig.savefig("seaborn/plots/KMeans-ci/"+attribute[1:-1].lower()+"-all.png")
	fig.savefig("temp/cdf-clustering/"+attribute[1:-1]+"/KMeans/"+attribute[1:-1].lower()+"-all.png")
	plt.clf()

	for cluster in f:
		print(cluster)
		print(df)
		print(df[df["cluster"] == int(cluster)])
		plot = sns.lineplot(x="percentile", y="value", legend="full", data=df[df["cluster"] == int(cluster)]).set_title(attribute[1:-1]+" - cluster "+cluster)
		fig = plot.get_figure()
		fig.savefig("temp/cdf-clustering/"+attribute[1:-1]+"/KMeans/"+attribute[1:-1].lower()+"-"+cluster+".png")
		plt.clf()
			