import torch.nn as nn


class Conv1DNet(nn.Module):
    def __init__(self,
                 input_channel,
                 channel_layer1,
                 kernel_layer1,
                 stride_layer1,
                 padding_layer1,
                 channel_layer2,
                 kernel_layer2,
                 stride_layer2,
                 padding_layer2,
                 channel_linear,
                 num_classes
                 ):
        super(Conv1DNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 1D convolution layer
            nn.Conv1d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm1d(channel_layer1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(24*24*channel_layer1, channel_linear),
            nn.Linear(channel_linear, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet(nn.Module):
    def __init__(self,
                 input_channel,
                 channel_layer1,
                 kernel_layer1,
                 stride_layer1,
                 padding_layer1,
                 channel_layer2,
                 kernel_layer2,
                 stride_layer2,
                 padding_layer2,
                 channel_linear,
                 num_classes):
        super(Conv2DNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer2),#nn.Linear(24*24*32, 32),
            nn.Linear(channel_layer2, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


