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
    def __init__(self, model_file):
        with open(model_file, 'r') as file:
            lines = file.readlines()
            self.preIndex = 0
            self.root = constructTree(lines, lines[0])

    def constructTree(self, lines):
        node = Node()
        line = lines[self.preIndex].split(',')
        self.preIndex += 1
        
        # leaf node
        if len(line).size == 2:
            node.index = line[0]
            node.label = line[1]

        # non-leaf node
        else:
            node.index = line[0]
            node.attribute = line[1]
            node.value = line[2]
            node.left = constructTree(lines) 
            node.right = constructTree(lines) 
      
        return node 


    def predict(X_test):
        self.X_test = X_test
        data_index = np.arange(X_test.shape[0])
        y_pred = np.array((X_test.shape[0],1))



def main(model_file, test_file, predictions):
    data = np.genfromtxt(test_file, delimiter=',')
    X_test = data[:, 1:]
    y_test = data[:, 0].astype(int)
    print(X_test.shape)
    print(y_test.shape)

    model = DecisionTree(model_file)
    model.predict(X_test)


if __name__ == '__main__':
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    predictions= sys.argv[3]
    main(model_file, test_file, predictions)