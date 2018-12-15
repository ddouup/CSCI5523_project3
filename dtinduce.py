import numpy as np
import sys, time, os

class Node:
    def __init__(self):
        self.index = None
        self.left = None
        self.right = None
        self.attribute = None
        self.value = None
        self.label = None

class DecisionTree():
    def __init__(self, minfreq):
        self.minfreq = int(minfreq)
        self.tree_index = 0

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
            leaf.index = self.tree_index
            self.tree_index +=1
            return leaf

        else:
            root = Node();
            root.attribute, root.value, left_data_index, right_data_index = self.find_best_split(_data_index, _attributes)
            root.index = self.tree_index
            self.tree_index += 1

            attributes = np.setdiff1d(_attributes, root.attribute)
            
            left_child = self.TreeGrowth(left_data_index, attributes)
            root.left = left_child

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
        print("There are ",data_index.size, "data points at this node")
        print("There are ",attributes.size, "attributes at this node")
        print(attributes)
        min_gini = 1
        attribute = 0
        value = 0
        left_data_index = np.array([], dtype=int)
        for i in attributes:
            print("The",i,"attribute.")
            # sort the split values
            sorted_index = data_index[self.X[data_index][:, i].argsort()]
            
            sorted_val = self.X[sorted_index][:, i]
            sorted_label = self.y[sorted_index]
            #print(sorted_val)
            #print(sorted_label)

            labels, counts = np.unique(sorted_label, return_counts=True)
            #print(counts)

            # initialize gini table, all data points at right side
            gini_table = np.zeros((labels.size, 2), dtype=int)   #       <=val, >val
                                                                    #labels      ,
            gini_table[:,1] = counts
            #print(gini_table)

            current_val = sorted_val[0]
            current_label = sorted_label[0]
            l_data_index = np.array([],dtype=int)
            # for each split value
            for j in range(sorted_val.size):
                #print("The", j, "data point")
                val = sorted_val[j]
                #print(val)
                label = sorted_label[j]
                
                if val != current_val:
                    current_val = val
                    if label != current_label:
                        current_label = label
                        gini = self.calculate_gini(gini_table)
                        #print(gini)
                        if gini < min_gini:
                            min_gini = gini
                            attribute = i
                            value = val
                            left_data_index = l_data_index
                            min_gini_table = np.copy(gini_table)
                            #print("min gini:", min_gini)
                            #print("attribute:", attribute)
                            #print("value:",value)
                            #print(min_gini_table)
                            #print()

                # left side decrease 1, right side increase 1
                index = np.argwhere(labels == label)[0][0]
                gini_table[index][0] += 1
                gini_table[index][1] -= 1
                l_data_index = np.append(l_data_index, sorted_index[j])

        right_data_index = np.setdiff1d(data_index, left_data_index)
        print("min gini:", min_gini)
        print("attribute:", attribute)
        print("value:",value)
        print(min_gini_table)
        print("left data index:",left_data_index)
        print("right data index:",right_data_index)
        print()
        return attribute, value, left_data_index, right_data_index

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

    def classify(self, data_index):
        labels, counts = np.unique(self.y[data_index], return_counts=True)
        label = labels[np.argmax(counts)]
        print("Label for this leaf:", label)
        return label

    def save(self, model_file):
        with open(model_file, 'w') as file:
            self.printPreorder(self.root, file)
        print("Model saved")

    def printPreorder(self, node, file):
        if node.label == None:
            file.write(str(node.index)+','+str(node.attribute)+','+str(node.value)+'\n')
            self.printPreorder(node.left, file)
            self.printPreorder(node.right, file)
        else:
            file.write(str(node.index)+','+str(node.label)+'\n')

def main(train_file, minfreq, model_file):
    data = np.genfromtxt(train_file, delimiter=',')
    X_train = data[:, 1:]
    y_train = data[:, 0].astype(int)
    print(X_train.shape)
    print(y_train.shape)

    start = time.time()
    model = DecisionTree(minfreq).fit(X_train, y_train)
    end = time.time()
    print("Time: ", start-end, "seconds")

    model.save(os.path.splitext(model_file)[0]+'_'+str(minfreq)+os.path.splitext(model_file)[1])


if __name__ == '__main__':
    train_file = sys.argv[1]
    minfreq = sys.argv[2]
    model_file= sys.argv[3]
    main(train_file, minfreq, model_file)