import numpy as np
import sys, time
from dtinduce import DecisionTree, Node

def main(train_file, minfreq, model_file):
	data_index =np.array([10808 , 5726, 33336, 27566, 31878 , 3818, 24156, 54592, 23816, 46362, 40764, 36464
	, 31818, 48843, 10576, 20129, 23771, 52184, 27277, 34892, 13932, 55600, 16934 , 2284
	, 32759 , 8356, 43502, 26068, 33402, 48234, 48940, 22340, 14247, 50053, 39648, 59443
	, 45355, 56270, 22540 , 2763, 38772])

	data = np.genfromtxt(train_file, delimiter=',')
	X_train = data[data_index][:, 1:]
	y_train = data[data_index][:, 0].astype(int)

	print(X_train[:, 5])
	print(y_train)
	attributes = np.array([5])
	start = time.time()
	model = DecisionTree(minfreq).fit(X_train, y_train, attributes)
	#model.save(model_file)
	end = time.time()
	print("Time: ", start-end, "seconds")

if __name__ == '__main__':
    train_file = sys.argv[1]
    minfreq = sys.argv[2]
    model_file= sys.argv[3]
    main(train_file, minfreq, model_file)