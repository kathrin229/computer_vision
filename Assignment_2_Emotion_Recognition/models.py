import torch.nn as nn


class Conv1DNet1Layer(nn.Module):
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
        super(Conv1DNet1Layer, self).__init__()

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


class Conv1DNet2Layer(nn.Module):
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
        super(Conv1DNet2Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 1D convolution layer
            nn.Conv1d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm1d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining a 1D convolution layer
            nn.Conv1d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2,padding=padding_layer2),
            nn.BatchNorm1d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer2),
            nn.Linear(channel_layer2, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet1Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet1Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1,
                      padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer1),
            nn.Linear(channel_layer1, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet2Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet2Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer2),
            nn.Linear(channel_layer2, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet3Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet3Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer3),
            nn.Linear(channel_layer3, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet4Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet4Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer4),
            nn.Linear(channel_layer4, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet5Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet5Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer5),
            nn.Linear(channel_layer5, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet6Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet6Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer6),
            nn.Linear(channel_layer6, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet7Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet7Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer6, channel_layer7, kernel_size=kernel_layer7, stride=stride_layer7,padding=padding_layer7),
            nn.BatchNorm2d(channel_layer7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer7),
            nn.Linear(channel_layer7, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet8Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet8Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer6, channel_layer7, kernel_size=kernel_layer7, stride=stride_layer7,padding=padding_layer7),
            nn.BatchNorm2d(channel_layer7),
            nn.ReLU(inplace=True),


            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer7, channel_layer8, kernel_size=kernel_layer8, stride=stride_layer8, padding=padding_layer8),
            nn.BatchNorm2d(channel_layer8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer8),
            nn.Linear(channel_layer8, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet9Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet9Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer6, channel_layer7, kernel_size=kernel_layer7, stride=stride_layer7,padding=padding_layer7),
            nn.BatchNorm2d(channel_layer7),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer7, channel_layer8, kernel_size=kernel_layer8, stride=stride_layer8, padding=padding_layer8),
            nn.BatchNorm2d(channel_layer8),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer8, channel_layer9, kernel_size=kernel_layer9, stride=stride_layer9, padding=padding_layer9),
            nn.BatchNorm2d(channel_layer9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer9),
            nn.Linear(channel_layer9, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet10Layer(nn.Module):
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
                 channel_layer3,
                 kernel_layer3,
                 stride_layer3,
                 padding_layer3,
                 channel_layer4,
                 kernel_layer4,
                 stride_layer4,
                 padding_layer4,
                 channel_layer5,
                 kernel_layer5,
                 stride_layer5,
                 padding_layer5,
                 channel_layer6,
                 kernel_layer6,
                 stride_layer6,
                 padding_layer6,
                 channel_layer7,
                 kernel_layer7,
                 stride_layer7,
                 padding_layer7,
                 channel_layer8,
                 kernel_layer8,
                 stride_layer8,
                 padding_layer8,
                 channel_layer9,
                 kernel_layer9,
                 stride_layer9,
                 padding_layer9,
                 channel_layer10,
                 kernel_layer10,
                 stride_layer10,
                 padding_layer10,
                 channel_linear,
                 num_classes):
        super(Conv2DNet10Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer6, channel_layer7, kernel_size=kernel_layer7, stride=stride_layer7,padding=padding_layer7),
            nn.BatchNorm2d(channel_layer7),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer7, channel_layer8, kernel_size=kernel_layer8, stride=stride_layer8, padding=padding_layer8),
            nn.BatchNorm2d(channel_layer8),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer8, channel_layer9, kernel_size=kernel_layer9, stride=stride_layer9, padding=padding_layer9),
            nn.BatchNorm2d(channel_layer9),
            nn.ReLU(inplace=True),

            # # Defining another 2D convolution layer
            nn.Conv2d(channel_layer9, channel_layer10, kernel_size=kernel_layer10, stride=stride_layer10,padding=padding_layer10),
            nn.BatchNorm2d(channel_layer10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer10),
            nn.Linear(channel_layer10, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x