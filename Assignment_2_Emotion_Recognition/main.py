import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


data_cv = pd.read_csv('data/fer2013/fer2013/fer2013.csv').to_numpy()
print(data_cv.shape)

# convert "Training", "PrivateTest", "PublicTest" to ints
unique_labels = np.unique(data_cv[:, 2])
number_dict = dict(zip(unique_labels, range(len(unique_labels))))
data_cv[:, 2] = np.array([number_dict[k] for k in data_cv[:, 2]])

# convert string pixel values to ints
data = np.zeros((data_cv.shape[0], len(data_cv[0, 1].split(' ')) + 2))
data[:, 0] = data_cv[:, 0]
data[:,  len(data_cv[0, 1].split(' ')) + 1] = data_cv[:, 2]
data[:, 1:  len(data_cv[0, 1].split(' ')) + 1] = np.array([s.split(' ') for s in data_cv[:, 1]]).astype(int)

# training, test, validation (x data shape: sample size, channel size, height, width)
train = np.expand_dims(data[np.where(data[:, -1] == 2)], axis=2)  # Training
x_train = torch.from_numpy(train[:, 1:-1, :])
y_train_int = torch.from_numpy(train[:, 0, :])
y_train = torch.nn.functional.one_hot(y_train_int[:, 0].to(torch.int64))

test = np.expand_dims(data[np.where(data[:, -1] == 0)], axis=2)  # PrivateTest
x_test = torch.from_numpy(test[:, 1:-1, :])
y_test_int = torch.from_numpy(test[:, 0, :])
y_test = torch.nn.functional.one_hot(y_test_int[:, 0].to(torch.int64))

valid = np.expand_dims(data[np.where(data[:, -1] == 1)], axis=2)  # PublicTest
x_valid = torch.from_numpy(valid[:, 1:-1, :])
y_valid_int = torch.from_numpy(valid[:, 0, :])
y_valid = torch.nn.functional.one_hot(y_valid_int[:, 0].to(torch.int64))

# images: shape 48 x 48
img = x_train[0].reshape(48, 48)
plt.imshow(img, cmap='gray')
plt.show()

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


