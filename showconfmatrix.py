import numpy as np
import sys, time

def main(pred_file):
    data = np.genfromtxt(pred_file, delimiter=',')
    confusion_matrix = np.zeros((10,10), dtype=int)
    for line in data:
        confusion_matrix[int(line[0])][int(line[1])] += 1

    total = np.sum(confusion_matrix)
    accuracy = np.sum(np.diag(confusion_matrix)) / total

    print('Total number: ', total)

    print('The confusion matrix:')
    print(confusion_matrix)
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
    pred_file = sys.argv[1]
    main(pred_file)