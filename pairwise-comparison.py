import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# attribute1 = sys.argv[1]
# attribute2 = sys.argv[2]

attributes = pd.read_csv("attributes.csv").index

# attribute1 = "BF_ADV"
# attribute2 = "FC_ADV"

# for i in range(len(attributes)):
for i in range(0, 1):
	for j in range(i+1, len(attributes)):
		attribute1 = attributes[i][1:-1]
		attribute2 = attributes[j][1:-1]

		method_dict = {0 : "KMeans", 1 : "Agglo"}

		method = method_dict[0]

		# d3f2 = pd.read_csv("temp/cdf-clustering/"+attribute2+"/"+method+"/labels.csv")

		df1 = pd.read_csv("temp/cdf-clustering/"+attribute1+"/"+method+"/labels.csv")
		df2 = pd.read_csv("temp/cdf-clustering/"+attribute2+"/"+method+"/labels.csv")

		del df1['Unnamed: 0']
		del df2['Unnamed: 0']

		df1.columns = ["District", attribute1]
		df2.columns = ["District", attribute2]

		df = df1.merge(df2, how='inner', on='District')


		mat = confusion_matrix(df[attribute1], df[attribute2])
		np.sum(mat)

		mat = np.vstack((mat, np.sum(mat, axis = 0)))
		mat = np.hstack((mat, np.sum(mat, axis = 1).reshape(len(np.sum(mat, axis = 1)), 1)))
		# print(mat)
		np.savetxt("confusion-matrix/"+method+"/"+attribute1+"_"+attribute2+".csv", mat, delimiter=", ", fmt="%d")

		df_at1_sns = pd.read_csv("seaborn/"+attribute1.lower()+".csv")
		del df_at1_sns["Unnamed: 0"]
		df_at2_sns = pd.read_csv("seaborn/"+attribute2.lower()+".csv")
		del df_at2_sns["Unnamed: 0"]
		df_at1_sns['attribute'] = attribute1
		df_at2_sns['attribute'] = attribute2

		# print(df1)
		at1_clusters = np.unique(np.asarray(df1[attribute1]))
		at2_clusters = np.unique(np.asarray(df2[attribute2]))

		if not os.path.exists("confusion-matrix/"+method+"/"+attribute1+"_"+attribute2+"/"):
			os.makedirs("confusion-matrix/"+method+"/"+attribute1+"_"+attribute2+"/")

		for cluster1 in at1_clusters:
			for cluster2 in at2_clusters:
				df_at1_sns_temp = df_at1_sns[df_at1_sns["cluster"] == cluster1]
				att1_dists = len(df_at1_sns_temp)/10
				df_at2_sns_temp = df_at2_sns[df_at2_sns["cluster"] == cluster2]
				att2_dists = len(df_at2_sns_temp)/10

				districts = np.intersect1d(np.asarray(df_at1_sns_temp["district"]), np.asarray(df_at2_sns_temp["district"]))

				df1 = df_at1_sns[df_at1_sns['district'].isin(districts)]
				df2 = df_at2_sns[df_at2_sns['district'].isin(districts)]
				df = df1.append(df2)
				
				plot = sns.lineplot(x="percentile", y="value", hue="attribute", data=df).set_title(attribute1+" cluster "+str(int(cluster1))+" - #"+str(int(att1_dists))+ "; " + attribute2+" cluster "+str(cluster2)+" - #"+str(int(att2_dists))+"; Common - #"+str(len(districts)))
				fig = plot.get_figure()
				fig.savefig("confusion-matrix/"+method+"/"+attribute1+"_"+attribute2+"/"+attribute1+"-"+str(cluster1)+"-"+attribute2+"-"+str(cluster2)+".png")
				plt.clf()

		print(attribute1+", "+attribute2+" done.")
