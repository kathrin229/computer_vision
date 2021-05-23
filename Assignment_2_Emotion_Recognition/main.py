import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import dataset
from models import Conv1DNet, Conv2DNet
from sklearn.metrics import precision_recall_fscore_support

# TODO check seed - reproducibility
torch.seed()
torch.manual_seed(0)

architecture = Conv1DNet
num_epochs = 1
learning_rate = 0.0001
batch_size = 64
model_args = {
    'in_channels': 1,    # grayscale = 1
    'out_channels': 64,  # num of filters
    'hidden_size': 32,   # linear layer
    'num_classes': 7
}


data = dataset.load_data()
train_loader, valid_loader, test_loader = dataset.get_data_loader(data, batch_size, architecture=architecture,
                                                                  shuffle=True, drop_last=True)
print("Finished loading data.")

model = architecture(**model_args)
print(model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # training loop (iterates over training batches)
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_train.float())
        loss_train = loss(y_pred, y_train)
        if epoch % 10 == 9:
            print('Epoch {}  -  Train loss: {}'.format(epoch, loss_train.item()))
        loss_train.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x_valid, y_valid in valid_loader:
            # validation loop
            y_pred = model(x_valid.float())
            # loss
            loss_valid = loss(y_pred, y_valid)
            # accuracy
            _, predicted = torch.max(y_pred.data, 1)
            total += y_valid.size(0)
            correct += (predicted == y_valid).sum().item()
        print('Epoch {}  -  Validation Accuracy: {}%'.format(epoch, 100 * correct / total))
        precision, recall, fscore, support = precision_recall_fscore_support(y_valid, predicted, average='macro')
        print('Precision (macro): {}  -  Recall (macro): {}  -  F-score (macro): {}%'.format(precision, recall, fscore))

# # images: shape 48 x 48
# img = x_train[0].flatten().reshape(48, 48)
# plt.imshow(img, cmap='gray')
# plt.show()

# # TODO: Ideas:
# # 1D Conv and 2D Conv
# # Hidden size
# # Number layers
# # Visualization

