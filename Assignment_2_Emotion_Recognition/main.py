import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

data_cv = pd.read_csv('data/fer2013/fer2013/fer2013.csv').to_numpy()
print(data_cv.shape)

data = np.zeros((data_cv.shape[0], len(data_cv[0, 1].split(' ')) + 2))
data[:, 0] = data_cv[:, 0]

unique_labels = np.unique(data_cv[:, 2])
number_dict = dict(zip(unique_labels, range(len(unique_labels))))
data_cv[:, 2] = np.array([number_dict[k] for k in data_cv[:, 2]])
data[:,  len(data_cv[0, 1].split(' ')) + 1] = data_cv[:, 2]
data[:, 1:  len(data_cv[0, 1].split(' ')) + 1] = np.array([s.split(' ') for s in data_cv[:, 1]]).astype(int)

# images: shape 48 x 48


# training, test, validation
train = data[np.where(data[:, -1] == 2)]  # Training
x_train = torch.from_numpy(train[:, 1:-1])
y_train = torch.from_numpy(train[:, 0])

img = x_train[0].reshape(48, 48)
plt.imshow(img, cmap='gray')
plt.show()

test = data[np.where(data[:, -1] == 0)]  # PrivateTest
x_test = torch.from_numpy(test[:, 1:])
y_test = torch.from_numpy(test[:, 0])

valid = data[np.where(data[:, -1] == 1)]  # PublicTest
x_valid = torch.from_numpy(valid[:, 1:])
y_valid = torch.from_numpy(valid[:, 0])

# data shape: sample size, channel size, height, width

# input layer size, hidden layer size, output layer size and batch size (N)
input_size = x_train.shape[1]
hidden_size = 1
output_size = 7
batch_size = 64

# TODO: Ideas:
# 1D Conv and 2D Conv
# Hidden size
# Number layers
# Visualization


class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #, num_classes=N):
        super(ConvNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_size, hidden_size, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(hidden_size, output_size, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


