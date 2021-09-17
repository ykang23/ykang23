import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from PIL import Image

class TransferLearning:
    def __init__(self, dataset):
        '''
        dataset : address of the dataset
        '''
        self.dataset = dataset
        self.image_transforms = { 
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                #transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        }

        train_directory = os.path.join(self.dataset, 'trainset-tl')
        valid_directory = os.path.join(self.dataset, 'validset-tl')
        test_directory = os.path.join(self.dataset, 'testset-tl')

        # Batch size
        self.bs = 32

        # Number of classes
        self.num_classes = len(os.listdir(valid_directory))

        # Load Data from folders
        self.data = {
            'train': datasets.ImageFolder(root=train_directory, transform=self.image_transforms['train']),
            'valid': datasets.ImageFolder(root=valid_directory, transform=self.image_transforms['valid']),
            'test': datasets.ImageFolder(root=test_directory, transform=self.image_transforms['test'])
        }
        self.idx_to_class = {v: k for k, v in self.data['train'].class_to_idx.items()}

        # Size of Data, to be used for calculating Average Loss and Accuracy
        self.train_data_size = len(self.data['train'])
        self.valid_data_size = len(self.data['valid'])
        self.test_data_size = len(self.data['test'])

        # Create iterators for the Data loaded using DataLoader module
        self.train_data_loader = DataLoader(self.data['train'], batch_size=self.bs, shuffle=True)
        self.valid_data_loader = DataLoader(self.data['valid'], batch_size=self.bs, shuffle=True)
        self.test_data_loader = DataLoader(self.data['test'], batch_size=self.bs, shuffle=True)


    def train_and_validate(self, model, loss_criterion, optimizer, epochs=25):
        '''
        Function to train and validate
        Parameters
            :param model: Model to train and validate
            :param loss_criterion: Loss Criterion to minimize
            :param optimizer: Optimizer for computing gradients
            :param epochs: Number of epochs (default=25)
    
        Returns
            model: Trained Model with best validation accuracy
            history: (dict object): Having training loss, accuracy and validation loss, accuracy
        '''
        
        start = time.time()
        history = []
        best_loss = 100000.0
        best_epoch = None

        for epoch in range(epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch+1, epochs))
            
            # Set to training mode
            model.train()
            
            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0
            
            valid_loss = 0.0
            valid_acc = 0.0

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for i, (inputs, labels) in enumerate(self.train_data_loader):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Clean existing gradients
                optimizer.zero_grad()
                
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)
                
                # Compute loss
                loss = loss_criterion(outputs, labels)
                
                # Backpropagate the gradients
                loss.backward()
                
                # Update the parameters
                optimizer.step()
                
                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)
                
                # Compute the accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                
                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                
                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)
                
                #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
            # Validation - No gradient tracking needed
            with torch.no_grad():

                # Set to evaluation mode
                model.eval()

                # Validation loop
                for j, (inputs, labels) in enumerate(self.valid_data_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass - compute outputs on input data using the model
                    outputs = model(inputs)

                    # Compute loss
                    loss = loss_criterion(outputs, labels)

                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)

                    # Calculate validation accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)

                    #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch

            # Find average training loss and training accuracy
            avg_train_loss = train_loss/self.train_data_size 
            avg_train_acc = train_acc/self.train_data_size

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss/self.valid_data_size 
            avg_valid_acc = valid_acc/self.valid_data_size

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                    
            epoch_end = time.time()
        
            print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
            
            # Save if the model has best accuracy till now
            torch.save(model, self.dataset+'_model_'+str(epoch)+'.pt')
                
        return model, history, best_epoch

    def computeTestSetAccuracy(self, model, loss_criterion, device):
        '''
        Function to compute the accuracy on the test set
        Parameters
            :param model: Model to test
            :param loss_criterion: Loss Criterion to minimize
        '''

        test_acc = 0.0
        test_loss = 0.0

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(self.test_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                test_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                test_acc += acc.item() * inputs.size(0)

                print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average test loss and test accuracy
        avg_test_loss = test_loss/self.test_data_size 
        avg_test_acc = test_acc/self.test_data_size

        print("Test accuracy : " + str(avg_test_acc))

    def predict(self, model, test_image_name):
        '''
        Function to predict the class of a single test image
        Parameters
            :param model: Model to test
            :param test_image_name: Test image

        '''
        prediction = []
        scores = []
        
        transform = self.image_transforms['test']


        test_image = Image.open(test_image_name)
        plt.imshow(test_image)
        
        test_image_tensor = transform(test_image)
        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
        
        with torch.no_grad():
            model.eval()
            # Model outputs log probabilities
            out = model(test_image_tensor)
            ps = torch.exp(out)

            topk, topclass = ps.topk(2, dim=1)
            cls = self.idx_to_class[topclass.cpu().numpy()[0][0]]
            score = topk.cpu().numpy()[0][0]

            for i in range(2):
                #print("Predcition", i+1, ":", self.idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
                prediction.append(self.idx_to_class[topclass.cpu().numpy()[0][i]])
                scores.append(topk.cpu().numpy()[0][i])
        return prediction, scores 


def main(argv):
    if len(sys.argv) < 2:
        print("Enter image")
        exit()
        
    dataset = '/personal/ykang23/Summer'
    tl = TransferLearning(dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet50 Model
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = resnet50.to(device)

    # Freeze model parameters
    for param in resnet50.parameters():
        param.requires_grad = False

    # Change the final layer of ResNet50 Model for Transfer Learning
    fc_inputs = resnet50.fc.in_features

    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, tl.num_classes), # Since 10 possible outputs
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )

    # Convert model to be used on GPU
    resnet50 = resnet50.to(device)

    # Define Optimizer and Loss Function
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet50.parameters())

    # Train the model for 30 epochs
    num_epochs = 30
    trained_model, history, best_epoch = tl.train_and_validate(resnet50, loss_func, optimizer, num_epochs)
    torch.save(history, dataset+'_history.pt')

    # Test a particular model on a test image
    model = torch.load("{}_model_{}.pt".format(dataset, best_epoch))

    img = sys.argv[1]
    tl.predict(model, img)


if __name__ == '__main__':
    main(sys.argv)
