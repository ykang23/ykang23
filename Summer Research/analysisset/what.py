import sys
import torch
import os
import numpy as np
from Program4 import TransferLearning

def main():
    dataset = '/personal/ykang23/Summer'
    tl = TransferLearning(dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test a particular model on a test image
    model = torch.load("{}_model_{}.pt".format(dataset, 29))

    c_path = os.path.join(dataset, 'testset/CU')
    n_path = os.path.join(dataset, 'testset/non-CU')
    
    cu_files = os.listdir(c_path)
    n_files = os.listdir(n_path)
    
    prediction1 = []
    prediction2 = []
    score1 = []
    score2 = []
    imgs = []
    labels = []
    for cu in cu_files:
        file = os.path.join(c_path, cu)
        predictions, scores = tl.predict(model, file)
        prediction1.append(predictions[0])
        prediction2.append(predictions[1])
        score1.append(scores[0])
        score2.append(scores[1])
        imgs.append(cu)
        labels.append("CU")
        
    for non_cu in n_files:
        file = os.path.join(n_path, non_cu)           
        predictions, scores = tl.predict(model,file)
        prediction1.append(predictions[0])
        prediction2.append(predictions[1])
        score1.append(scores[0])
        score2.append(scores[1])
        imgs.append(non_cu)
        labels.append("non-CU")

    information = np.array([imgs, labels, prediction1, score1, prediction2, score2])
    #print(information.shape)
    #print(information[:,:5])
    information = information.T
    np.savetxt("closeup_identification.csv", information, delimiter = ",", header = "Image Title, Actual Label , Prediction I, Score I, Prediction II, Score II", fmt = ['%s','%s','%s','%s','%s','%s'],comments='')

if __name__ == '__main__':
    main()
