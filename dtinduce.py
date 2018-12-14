import numpy as np
import sys

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.attribute = None
        self.value = None
        self.label = None

class DecisionTree():
    def __init__(self, minfreq):
        self.minfreq = int(minfreq)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.num = X.shape[0]             # number of instances
        self.feature_num = X.shape[1]     # number of features
        self.labels = np.unique(y)
        self.label_num = self.labels.size

        data_index = np.arange(self.num)
        attributes = np.arange(self.feature_num)
        
        self.root = self.TreeGrowth(data_index, attributes)

        return self

    def TreeGrowth(self, _data_index, _attributes):
        if self.stop_cond(_data_index, _attributes) == True:
            leaf = Node();
            leaf.label = self.classify(_data_index)
            return leaf

        else:
            root = Node();
            root.attribute, root.value = self.find_best_split(_data_index, _attributes)

            attributes = np.delete(_attributes, root.attribute)
            
            left_data_index = _data_index
            left_child = self.TreeGrowth(left_data_index, attributes)
            root.left = left_child

            right_data_index = _data_index
            right_child = self.TreeGrowth(right_data_index, attributes)
            root.right = right_child

        return root

    def stop_cond(self, data_index, _attributes):
        print("Check stop conditions:")
        if _attributes.size == 0:
            print("No more features. Stop growth.")
            return True
        elif data_index.size < self.minfreq:
            print("Data points less than minfreq. Stop growth.")
            return True
        elif np.unique(self.y[data_index]).size == 1:
            print("Data points belong to the same class. Stop growth.")
            return True
        else:
            print("Keep growing...")
            return False
        
    def find_best_split(self, data_index, attributes):
        num = data_index.size
        print("There are ",num, "data points at this node")
        max_gini = 0
        attribute = 0
        value = 0
        for i in attributes:
            # sort the split values
            sorted_index = data_index[self.X[:,i].argsort()]
            
            sorted_val = self.X[sorted_index][:, i]
            sorted_label = self.y[sorted_index]
            print(sorted_val)
            print(sorted_label)

            # initialize gini table, all data points at right side
            gini_table = np.zeros((self.label_num, 2), dtype=int)   #       <=val, >val
                                                                    #labels      ,
            labels, counts = np.unique(sorted_label, return_counts=True)
            print(counts)

            gini_table[:,1] = counts
            print(gini_table)

            current_val = sorted_val[0]
            # for all data points
            for j in range(num):
                val = sorted_val[j]
                label = sorted_label[j]
                index = np.argwhere(self.labels == label)[0][0]
                print(index)
                # left side decrease 1, right side increase 1
                gini_table[index][0] -= 1
                gini_table[index][1] += 1
                print(gini_table)
                print(self.calculate_gini(gini_table))
                sys.exit()
                
                if val != current_val:
                    current_val = val
                    gini = self.calculate_gini(gini_table)
                    if gini > max_gini:
                        max_gini = gini
                        attribute = i
                        value = val

        return attribute, value

    def calculate_gini(self, gini_table):
        left_total = gini_table[:,0].sum()
        right_total = gini_table[:,1].sum()
        total = left_total + right_total

        l = 1
        r = 1
        for i in range(gini_table.shape[0]):
            if left_total != 0:
                l -= (gini_table[i][0]/left_total)**2
            if right_total != 0:
                r -= (gini_table[i][1]/right_total)**2

        gini_index = left_total/total * l + right_total/total * r
        return gini_index

    def classify(self, data):
        
        return label

    def save(self, model_file):
        print("Model saved")


def main(train_file, minfreq, model_file):
    data = np.genfromtxt(train_file, delimiter=',')
    X_train = data[:, 1:]
    y_train = data[:, 0].astype(int)
    print(X_train.shape)
    print(y_train.shape)

    model = DecisionTree(minfreq).fit(X_train, y_train)
    #model.save(model_file)


if __name__ == '__main__':
    train_file = sys.argv[1]
    minfreq = sys.argv[2]
    model_file= sys.argv[3]
    main(train_file, minfreq, model_file)