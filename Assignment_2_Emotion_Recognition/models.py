import torch.nn as nn

######################################################
# defining Conv1D Models with 1 and 2 Layers
######################################################
class Conv1DNet1Layer(nn.Module):
    def __init__(self, input_channel=1,
                 channel_layer1=32, kernel_layer1=5, stride_layer1=2, padding_layer1=2,
                 channel_linear=6 * 6 * 64, num_classes=7):
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
            nn.ReLU(inplace=True),
            nn.Linear(channel_linear, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv1DNet2Layer(nn.Module):
    def __init__(self, input_channel=1,
                 channel_layer1=32, kernel_layer1=5, stride_layer1=2, padding_layer1=2,
                 channel_layer2=16, kernel_layer2=5, stride_layer2=2, padding_layer2=2,
                 channel_linear=6 * 6 * 64, num_classes=7):
        super(Conv1DNet2Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 1D convolution layer
            nn.Conv1d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm1d(channel_layer1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Defining a 1D convolution layer
            nn.Conv1d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2,padding=padding_layer2),
            nn.BatchNorm1d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer2),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer2, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


#####################################################################
# defining Conv2D Models with 1, 2, 3, 4, 5, 6, 7, 8, 9 and 10 Layers
#####################################################################
class Conv2DNet1Layer(nn.Module):
    def __init__(self, input_channel=1,
                 channel_layer1=16, kernel_layer1=5, stride_layer1=2, padding_layer1=2,
                 channel_linear=6 * 6 * 64, num_classes=7):
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
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer1, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet2Layer(nn.Module):
    def __init__(self, input_channel=1,
                 channel_layer1=64, kernel_layer1=5, stride_layer1=2, padding_layer1=2,
                 channel_layer2=64, kernel_layer2=5, stride_layer2=2, padding_layer2=2,
                 channel_linear=6 * 6 * 64, num_classes=7):
        super(Conv2DNet2Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer2),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer2, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet3Layer(nn.Module):
    def __init__(self, input_channel=1,
                 channel_layer1=64, kernel_layer1=5, stride_layer1=2, padding_layer1=2,
                 channel_layer2=64, kernel_layer2=5, stride_layer2=2, padding_layer2=2,
                 channel_layer3=64, kernel_layer3=5, stride_layer3=2, padding_layer3=2,
                 channel_linear=6 * 6 * 16, num_classes=7):
        super(Conv2DNet3Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer3),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer3, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet4Layer(nn.Module):
    def __init__(self, input_channel=1,
                 channel_layer1=64, kernel_layer1=5, stride_layer1=2, padding_layer1=2,
                 channel_layer2=64, kernel_layer2=5, stride_layer2=2, padding_layer2=2,
                 channel_layer3=64, kernel_layer3=5, stride_layer3=2, padding_layer3=2,
                 channel_layer4=64, kernel_layer4=5, stride_layer4=2, padding_layer4=2,
                 channel_linear=8 * 8, num_classes=7):
        super(Conv2DNet4Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer4),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer4, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet5Layer(nn.Module):
    def __init__(self, input_channel=1,
                 channel_layer1=128, kernel_layer1=5, stride_layer1=2, padding_layer1=2,
                 channel_layer2=128, kernel_layer2=5, stride_layer2=2, padding_layer2=2,
                 channel_layer3=128, kernel_layer3=5, stride_layer3=2, padding_layer3=2,
                 channel_layer4=128, kernel_layer4=5, stride_layer4=2, padding_layer4=2,
                 channel_layer5=64, kernel_layer5=5, stride_layer5=2, padding_layer5=2,
                 channel_linear=8 * 8, num_classes=7):
        super(Conv2DNet5Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer5),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer5, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet6Layer(nn.Module):
    def __init__(self, input_channel=1,
                 channel_layer1=256, kernel_layer1=3, stride_layer1=1, padding_layer1=1,
                 channel_layer2=256, kernel_layer2=3, stride_layer2=1, padding_layer2=1,
                 channel_layer3=128, kernel_layer3=3, stride_layer3=1, padding_layer3=1,
                 channel_layer4=128, kernel_layer4=3, stride_layer4=1, padding_layer4=1,
                 channel_layer5=64, kernel_layer5=3, stride_layer5=1, padding_layer5=1,
                 channel_layer6=64, kernel_layer6=3, stride_layer6=1, padding_layer6=1,
                 channel_linear=6*6*64, num_classes=7
                 ):
        super(Conv2DNet6Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer6),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer6, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet7Layer(nn.Module):
    def __init__(self, input_channel,
                 channel_layer1=256, kernel_layer1=3, stride_layer1=1, padding_layer1=1,
                 channel_layer2=256, kernel_layer2=3, stride_layer2=1, padding_layer2=1,
                 channel_layer3=128, kernel_layer3=3, stride_layer3=1, padding_layer3=1,
                 channel_layer4=128, kernel_layer4=3, stride_layer4=1, padding_layer4=1,
                 channel_layer5=64, kernel_layer5=3, stride_layer5=1, padding_layer5=1,
                 channel_layer6=64, kernel_layer6=3, stride_layer6=1, padding_layer6=1,
                 channel_layer7=32, kernel_layer7=3, stride_layer7=1, padding_layer7=1,
                 channel_linear=6*6*32, num_classes=7
                 ):
        super(Conv2DNet7Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer6, channel_layer7, kernel_size=kernel_layer7, stride=stride_layer7,padding=padding_layer7),
            nn.BatchNorm2d(channel_layer7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer7),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer7, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet8Layer(nn.Module):
    def __init__(self, input_channel,
                 channel_layer1=256, kernel_layer1=3, stride_layer1=1, padding_layer1=1,
                 channel_layer2=256, kernel_layer2=3, stride_layer2=1, padding_layer2=1,
                 channel_layer3=128, kernel_layer3=3, stride_layer3=1, padding_layer3=1,
                 channel_layer4=128, kernel_layer4=3, stride_layer4=1, padding_layer4=1,
                 channel_layer5=64, kernel_layer5=3, stride_layer5=1, padding_layer5=1,
                 channel_layer6=64, kernel_layer6=3, stride_layer6=1, padding_layer6=1,
                 channel_layer7=32, kernel_layer7=3, stride_layer7=1, padding_layer7=1,
                 channel_layer8=32, kernel_layer8=3, stride_layer8=1, padding_layer8=1,
                 channel_linear=3*3*32, num_classes=7
                 ):
        super(Conv2DNet8Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer6, channel_layer7, kernel_size=kernel_layer7, stride=stride_layer7,padding=padding_layer7),
            nn.BatchNorm2d(channel_layer7),
            nn.ReLU(inplace=True),


            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer7, channel_layer8, kernel_size=kernel_layer8, stride=stride_layer8, padding=padding_layer8),
            nn.BatchNorm2d(channel_layer8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer8),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer8, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet9Layer(nn.Module):
    def __init__(self, input_channel,
                 channel_layer1=256, kernel_layer1=3, stride_layer1=1, padding_layer1=1,
                 channel_layer2=256, kernel_layer2=3, stride_layer2=1, padding_layer2=1,
                 channel_layer3=128, kernel_layer3=3, stride_layer3=1, padding_layer3=1,
                 channel_layer4=128, kernel_layer4=3, stride_layer4=1, padding_layer4=1,
                 channel_layer5=64, kernel_layer5=3, stride_layer5=1, padding_layer5=1,
                 channel_layer6=64, kernel_layer6=3, stride_layer6=1, padding_layer6=1,
                 channel_layer7=32, kernel_layer7=3, stride_layer7=1, padding_layer7=1,
                 channel_layer8=32, kernel_layer8=3, stride_layer8=1, padding_layer8=1,
                 channel_layer9=32, kernel_layer9=3, stride_layer9=1, padding_layer9=1,
                 channel_linear=6*6*32, num_classes=7
                 ):
        super(Conv2DNet9Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer6, channel_layer7, kernel_size=kernel_layer7, stride=stride_layer7,padding=padding_layer7),
            nn.BatchNorm2d(channel_layer7),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer7, channel_layer8, kernel_size=kernel_layer8, stride=stride_layer8, padding=padding_layer8),
            nn.BatchNorm2d(channel_layer8),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer8, channel_layer9, kernel_size=kernel_layer9, stride=stride_layer9, padding=padding_layer9),
            nn.BatchNorm2d(channel_layer9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer9),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer9, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv2DNet10Layer(nn.Module):
    def __init__(self, input_channel,
                 channel_layer1=256, kernel_layer1=3, stride_layer1=1, padding_layer1=1,
                 channel_layer2=256, kernel_layer2=3, stride_layer2=1, padding_layer2=1,
                 channel_layer3=128, kernel_layer3=3, stride_layer3=1, padding_layer3=1,
                 channel_layer4=128, kernel_layer4=3, stride_layer4=1, padding_layer4=1,
                 channel_layer5=64, kernel_layer5=3, stride_layer5=1, padding_layer5=1,
                 channel_layer6=64, kernel_layer6=3, stride_layer6=1, padding_layer6=1,
                 channel_layer7=32, kernel_layer7=3, stride_layer7=1, padding_layer7=1,
                 channel_layer8=32, kernel_layer8=3, stride_layer8=1, padding_layer8=1,
                 channel_layer9=32, kernel_layer9=3, stride_layer9=1, padding_layer9=1,
                 channel_layer10=32, kernel_layer10=3, stride_layer10=1, padding_layer10=1,
                 channel_linear=6*6*32, num_classes=7
                 ):
        super(Conv2DNet10Layer, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(input_channel, channel_layer1, kernel_size=kernel_layer1, stride=stride_layer1, padding=padding_layer1),
            nn.BatchNorm2d(channel_layer1),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer1, channel_layer2, kernel_size=kernel_layer2, stride=stride_layer2, padding=padding_layer2),
            nn.BatchNorm2d(channel_layer2),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer2, channel_layer3, kernel_size=kernel_layer3, stride=stride_layer3, padding=padding_layer3),
            nn.BatchNorm2d(channel_layer3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer3, channel_layer4, kernel_size=kernel_layer4, stride=stride_layer4,padding=padding_layer4),
            nn.BatchNorm2d(channel_layer4),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer4, channel_layer5, kernel_size=kernel_layer5, stride=stride_layer5,padding=padding_layer5),
            nn.BatchNorm2d(channel_layer5),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer5, channel_layer6, kernel_size=kernel_layer6, stride=stride_layer6, padding=padding_layer6),
            nn.BatchNorm2d(channel_layer6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer6, channel_layer7, kernel_size=kernel_layer7, stride=stride_layer7,padding=padding_layer7),
            nn.BatchNorm2d(channel_layer7),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer7, channel_layer8, kernel_size=kernel_layer8, stride=stride_layer8, padding=padding_layer8),
            nn.BatchNorm2d(channel_layer8),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer8, channel_layer9, kernel_size=kernel_layer9, stride=stride_layer9, padding=padding_layer9),
            nn.BatchNorm2d(channel_layer9),
            nn.ReLU(inplace=True),

            # Defining another 2D convolution layer
            nn.Conv2d(channel_layer9, channel_layer10, kernel_size=kernel_layer10, stride=stride_layer10,padding=padding_layer10),
            nn.BatchNorm2d(channel_layer10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.drop_out = nn.Dropout()  # default 0.5

        self.linear_layers = nn.Sequential(
            nn.Linear(channel_linear, channel_layer10),
            nn.ReLU(inplace=True),
            nn.Linear(channel_layer10, num_classes),
            nn.LogSoftmax(dim=1),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x