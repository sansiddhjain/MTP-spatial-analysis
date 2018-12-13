import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# attribute = sys.argv[1]
attributes = ['BF_ADV', 'BF_INT', 'BF_RUD', 'CHH_ADV', 'CHH_INT', 'CHH_RUD', 'FC_ADV', 'FC_INT', 'MSW_ADV', 'MSW_INT', 'MSW_RUD']

for attribute in attributes:
	os.chdir('cdf-clustering/'+attribute+'/KMeans/data/')

	for label in ['0', '1']:
		f = [j for j in os.listdir(os.getcwd()+'/'+label+'/') if '.txt' in j]
		X = np.zeros((10, 2))
		counter = 0
		plt.figure()
		plt.xlabel('Bins')
		plt.ylabel(attribute)
		plt.title('Spatial Curve (All Districts) : '+attribute+', Label - '+label)
		for file in f:
			# district = file.split('.txt')[0]
			mat = np.genfromtxt(label+'/'+file, delimiter=' ')
			plt.plot(mat[:, 0], mat[:, 1], '-')
			X = X + mat
			counter += 1

		plt.savefig(label+'/'+'all-curves.png')
		plt.close()

		X = X / counter
		print X
		print str(X[0, 1]) + ', ' + str(X[9, 1])

		plt.figure()
		plt.xlabel('Bins')
		plt.ylabel(attribute)
		plt.title('Average Spatial Curve : '+attribute+', Label - '+label)
		plt.plot(X[:, 0], X[:, 1], '-')
		# plt.legend(loc='best')
		plt.savefig(label+'/'+'avg-curve.png')
		plt.close()

	os.chdir('../../../../')

