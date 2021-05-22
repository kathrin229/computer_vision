import torch.nn as nn


class Conv1DNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Conv1DNet, self).__init__()

    #     self.cnn_layers = nn.Sequential(
    #         # Defining a 2D convolution layer
    #         nn.Conv2d(input_size, hidden_size, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
    #         nn.BatchNorm2d(hidden_size),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #
    #         # Defining another 2D convolution layer
    #         nn.Conv2d(hidden_size, num_classes, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)),
    #         nn.BatchNorm2d(num_classes),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #     )
    #
    #     self.linear_layers = nn.Sequential(
    #         nn.Linear(84, hidden_size),
    #         nn.Linear(hidden_size, num_classes)
    #     )
    #
    # # Defining the forward pass
    # def forward(self, x):
    #     x = self.cnn_layers(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.linear_layers(x)
    #     return x


class Conv2DNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Conv2DNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Defining another 2D convolution layer
            # nn.Conv2d(hidden_size, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=1, stride=1),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(24*24*32, 32),
            nn.Linear(hidden_size, num_classes)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


