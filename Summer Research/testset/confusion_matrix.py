import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
def my_confusion_matrix(y, y_pred):
    '''
        True CU    False non-CU
        False CU   Tue non-CU
    '''
    n = len(np.unique(y))
    matrix = np.zeros((n,n))
    print(zip(y,y_pred))
    for i, j in zip(y, y_pred):
        matrix[i][j] += 1
    return matrix

def accuracy(y, y_pred):
    accurate = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            accurate += 1
    return accurate/len(y)

def main():
    file = '/Volumes/Personal/ykang23/Summer/testset/closeup_identification.csv'
    data = np.genfromtxt(file, delimiter = ',', dtype = str)
    data = data[1:,:]
    #print(data)
    actual_labels = data[:,1]
    derived_labels = data[:,2]
    #print(actual_labels)
    #print(derived_labels)
    matrix = confusion_matrix(actual_labels, derived_labels)
    acc = accuracy(actual_labels, derived_labels)
    print(matrix)
    #print(acc)
    print("Accurately classified CUs: ", matrix[0][0])
    print("CUs classified as non-CUs: ", matrix[0][1])
    print("Accurately classified non-CUs: ", matrix[1][1])
    print("non_CUs classified as CUs: ", matrix[1][0])
    print("Accuracy: ", acc)
    #plt.matshow(matrix)
    
if __name__ == "__main__":
    main()

    
