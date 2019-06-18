__author__ = "Manuel Vasquez"

import torch.nn as nn
import torch.nn.functional as F


class Model_1(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        # ======================================================================
        # One fully connected layer.
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 10)
        )
        # self.fc1 = nn.Linear(input_dim, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, 10)
        # self.sig = nn.Sigmoid()

        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features
        
        # x = self.sig(self.fc1(x))
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)

        return self.features(x)

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()


class Model_2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        
        # self.maxp = nn.MaxPool2d(2, stride=1)
        # self.sig = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(1, 40, 5, 1)
        # self.conv2 = nn.Conv2d(40, 40, 5, 1)
        # self.fc1 = nn.Linear(8*8*40, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, 10)

        # self.conv1 = nn.Conv2d(1, 40, 5, 1)       # 10*rows*cols => 40*rows*cols
        # self.sig = nn.Sigmoid()
        # self.maxp = nn.MaxPool2d(2, stride=1)     # 40*rows*cols => 40*rows*cols/4
        # self.conv2 = nn.Conv2d(40, 40, 5, 1)        # 40*rows*cols/4
        # nn.Sigmoid(),
        # nn.MaxPool2d(2, stride=1),      # 40*rows*cols/16

        
        # self.fc1 = nn.Linear(40*18*18, hidden_size)   # 40*rows*cols/16
        # nn.Sigmoid(),
        # self.fc2 = nn.Linear(hidden_size, 10)

        self.features1 = nn.Sequential(
            nn.Conv2d(1, 40, 5, 1),
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(40, 40, 5, 1),
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=1)
        )

        self.features2 = nn.Sequential(
            nn.Linear(40*18*18, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 10)
        )
        
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features

        # x = self.sig(self.conv1(x))
        # x = self.maxp(x)
        # x = self.sig(self.conv2(x))
        # x = self.maxp(x)
        # x = x.view(-1, 40*18*18)
        # x = self.sig(self.fc1(x))
        # return self.fc2(x)

        x = self.features1(x)
        x = x.view(-1, 40*18*18)
        return self.features2(x)

        # print(np.shape(x))
        # x = self.seq1(x)
        # print(np.shape(x))
        # x = x.view(-1, 40*7*7)
        # return self.seq2(x)

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()


class Model_3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.

        self.features1 = nn.Sequential(
            nn.Conv2d(1, 40, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(40, 40, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )

        self.features2 = nn.Sequential(
            nn.Linear(40*18*18, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, 10)
        )

        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features
        
        x = self.features1(x)
        x = x.view(-1, 40*18*18)
        return self.features2(x)

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

class Model_4(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        
        self.features1 = nn.Sequential(
            nn.Conv2d(1, 40, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(40, 40, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(40*18*18, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

        # self.features2 = nn.Sequential(
        #     nn.MSELoss(),
        #     nn.Linear(40*18*18, hidden_size),
        #     nn.ReLU(),
        #     nn.MSELoss(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 10)
        # )

        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features

        x = self.features1(x)
        x = x.view(-1, 40*18*18)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        return self.fc3(x)

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

class Model_5(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        
        self.features1 = nn.Sequential(
            nn.Conv2d(1, 40, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(40, 40, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )

        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(40*18*18, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)


        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features
        
        x = self.features1(x)
        x = x.view(-1, 40*18*18)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        return self.fc3(x)

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()


class Net(nn.Module):
    def __init__(self, mode, args):
        super(Net, self).__init__()
        self.mode = mode
        self.hidden_size = args.hidden_size
        self.soft = nn.Softmax(dim=1)

        # model 1: base line
        if mode == 1:
            in_dim = 28*28  # input image size is 28x28
            self.model = Model_1(in_dim, self.hidden_size)

        # model 2: use two convolutional layer
        if mode == 2:
            self.model = Model_2(self.hidden_size)

        # model 3: replace sigmoid with relu
        if mode == 3:
            self.model = Model_3(self.hidden_size)

        # model 4: add one extra fully connected layer
        if mode == 4:
            self.model = Model_4(self.hidden_size)
            self.soft = nn.LogSoftmax(dim=1)

        # model 5: utilize dropout
        if mode == 5:
            self.model = Model_5(self.hidden_size)
            self.soft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if self.mode == 1:
            x = x.view(-1, 28 * 28)
            x = self.model(x)
        if self.mode in [2, 3, 4, 5]:
            x = self.model(x)

        # ======================================================================
        # Define softmax layer, use the features.

        logits = self.soft(x)
        return logits

        # Remove NotImplementedError and assign calculated value to logits after code implementation.

