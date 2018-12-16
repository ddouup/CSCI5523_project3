import numpy as np
import sys, os

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
            self.root = self.constructTree(lines)
            print('Number of nodes:', self.preIndex)

    def constructTree(self, lines):
        node = Node()
        line = lines[self.preIndex].split(',')
        self.preIndex += 1
        
        # leaf node
        if len(line) == 2:
            node.index = line[0].strip()
            node.label = line[1].strip()

        # non-leaf node
        else:
            node.index = line[0].strip()
            node.attribute = line[1].strip()
            node.value = line[2].strip()
            node.left = self.constructTree(lines) 
            node.right = self.constructTree(lines) 
      
        return node 


    def predict(self, X_test):
        test_num = X_test.shape[0]
        data_index = np.arange(test_num, dtype=int)
        y_pred = np.array([], dtype=int)
        for i in range(test_num):
            node = self.root
            x = X_test[i]
            while node.label == None:
                if x[int(node.attribute)] < float(node.value):
                    node = node.left
                else:
                    node = node.right

            y_pred = np.append(y_pred, node.label)

        return y_pred


def main(model_file, test_file, pred_file):
    data = np.genfromtxt(test_file, delimiter=',')
    X_test = data[:, 1:]
    y_test = data[:, 0].astype(int)
    print(X_test.shape)
    print(y_test.shape)

    model = DecisionTree(model_file)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    assert y_test.size == y_pred.size

    path = os.path.splitext(model_file)[0]+'_'+pred_file
    with open(path, 'w') as file:
        for i in range(y_test.size):
            file.write(str(y_test[i])+','+str(y_pred[i])+'\n')


if __name__ == '__main__':
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    pred_file= sys.argv[3]
    main(model_file, test_file, pred_file)