import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.autograd import Variable
import time

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size_ = 8

#load daily gesture data
class SkeletionSensorDataset(Dataset):
    def __init__(self,file_num):
        self.total_data = pd.read_csv('Dataset/csv/'+ str(file_num) + '.csv')
        self.total_data.fillna(0, inplace=True)

         
        self.data = self.total_data.iloc[:,:-1].to_numpy()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.normal_data = scaler.fit_transform(self.data)

        self.label = self.total_data.iloc[:,-1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.normal_data[idx]) #MinMaxScaler
        y = torch.FloatTensor(self.label.iloc[idx])
        x = x.to(device)
        y = y.to(device)

        return x,y

#save classification data
def classification_report_csv(report,file_num,fold):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
     
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])


        report_data.append(row)
        if(row_data[1] == '   ' + str(9.0)):
            break
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('results2/file_num_' + str(file_num)+ '_' + str(fold) + '.csv', index = False)




class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  #internal state
        # Propagate input through LSTM

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

num_epochs = 300 #1000 epochs
count = 0



hidden_size = 512  # number of features in hidde
num_layers = 1 #number of stacked lstm layers

num_classes = 10 #number of output classes 



# K-fold Cross Validation model evaluation
for file_num in range(11):
    #file_num = file_num + 6
    df = pd.read_csv('Dataset/csv/'+ str(file_num) + '.csv')
    print(len(df.columns))
    input_size = len(df.columns)-1
    dataset = SkeletionSensorDataset(file_num)
    k_folds = 10
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        start_time = time.time()
        print("fold",fold)

        # Lists for visualization of loss and accuracy 
        loss_list = []
        iteration_list = []
        accuracy_list = []

        # Lists for knowing classwise accuracy
        predictions_list = []
        labels_list = []

        input_size = input_size  # number of features
        start = time.time()
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size_,sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=batch_size_, sampler=test_subsampler)

        model = LSTM1(num_classes, input_size, hidden_size, num_layers) #our lstm class 
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()    # mean-squared error for regression
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)

        print(model)
        print(len(train_loader))
        print(len(test_loader))

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(train_loader):
                y = y-1
                x = x.reshape(-1, 1, input_size).to(device)
                y = y.view(y.shape[0]*y.shape[1])
                outputs = model(x)

                loss = criterion(outputs,y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 1 == 0:
                   print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        # Test the model
        #model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_loader:
                x = x.reshape(-1, 1, input_size).to(device)
                y = y-1
                y = y.view(y.shape[0]*y.shape[1])
                labels_list.append(y)
                outputs = model(x)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)

                correct += (predictions == y).sum()
                #print("correct",correct)

                total += len(y)
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            print ('Epoch [{}/{}], Loss: {:.4f},Accuracy: {}%' 
                    .format(epoch+1,num_epochs, loss.data, accuracy))
            end = time.time()
            print("working time", end - start)
        with open('results2/file_num_' + str(file_num)+ '_' + str(fold) + '.txt', 'w') as f:
            f.write(str(accuracy))
            f.write('\n')
            f.write(str(time.time() - start_time))

      # Print fold results
        print('K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        print(accuracy_list)
        print('Average:', sum(accuracy_list)/len(accuracy_list))



    # Print precision recall f1-socre
        from itertools import chain 

        predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
        labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
        predictions_l = list(chain.from_iterable(predictions_l))
        labels_l = list(chain.from_iterable(labels_l))

        import sklearn.metrics as metrics

        confusion_matrix(labels_l, predictions_l)
        report = metrics.classification_report(labels_l, predictions_l)
        classification_report_csv(report,file_num,fold)
        print("Classification report for CNN :\n%s\n"
              % (metrics.classification_report(labels_l, predictions_l)))

