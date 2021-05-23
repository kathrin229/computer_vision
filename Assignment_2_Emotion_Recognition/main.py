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
num_epochs = 10
learning_rate = 0.0001
batch_size = 64
model_args = {
    'input_channel': 1,

    'channel_layer1': 32,
    'kernel_layer1': 5,
    'stride_layer1': 2,
    'padding_layer1': 2,

    'channel_layer2': 64,
    'kernel_layer2': 5,
    'stride_layer2': 2,
    'padding_layer2': 2,

    'channel_linear': 3*3*64,
    'num_classes': 7
}


data = dataset.load_data()
train_loader, valid_loader, test_loader = dataset.get_data_loader(data, batch_size, architecture=architecture,
                                                                  shuffle=True, drop_last=True)
print("Finished loading data.\n")

model = architecture(**model_args)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)
print("\n")

min_valid_loss = np.inf
for epoch in range(num_epochs):
    train_loss = 0.0
    ######################################################
    # training loop (iterates over training batches)
    ######################################################
    for x_train, y_train in train_loader:
        # clear the old gradients from optimized variables
        optimizer.zero_grad()
        # forward pass: feed inputs to the model to get outputs
        y_pred = model(x_train.float())
        # calculate the training batch loss
        loss_train = loss(y_pred, y_train)
        # backward: perform gradient descent of the loss w.r. to the model params
        loss_train.backward()
        # update the model parameters by performing a single optimization step
        optimizer.step()
        # accumulate the training loss
        train_loss += loss_train.item()

    ######################################################
    # validation loop
    ######################################################
    # set the model to eval mode
    model.eval()
    valid_loss = 0.0
    # turn off gradients for validation
    with torch.no_grad():
        for x_valid, y_valid in valid_loader:
            # forward pass
            y_pred = model(x_valid.float())
            # validation batch loss
            loss_valid = loss(y_pred, y_valid)
            # accumulate the valid_loss
            valid_loss += loss_valid.item()

    # print epoch results
    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    print(f'Epoch: {epoch+1}/{num_epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')

######################################################
# test loop
######################################################

# set the model to eval mode
model.eval()
# turn off gradients for validation
with torch.no_grad():
    correct = 0
    total = 0
    for x_test, y_test in test_loader:
        # forward pass
        y_pred = model(x_test.float())
        # validation batch loss
        loss_test = loss(y_pred, y_test)
        # accumulate the valid_loss
        total += y_test.size(0)
        # calculate the accuracy
        predicted = torch.argmax(y_pred, 1)
        correct += (predicted == y_test).sum().item()
    print('Test Accuracy: {}%'.format(100 * correct / total))

loss_test /= len(test_loader)
accuracy = correct / len(test_loader)
print(f'Test loss: {loss_test}.. Test Accuracy: {accuracy}')

precision, recall, fscore, support = precision_recall_fscore_support(y_valid, predicted, average='macro')
print(f'Precision (macro): {precision}.. Recall (macro): {recall}.. F-score (macro): {fscore}')


# # images: shape 48 x 48
# img = x_train[0].flatten().reshape(48, 48)
# plt.imshow(img, cmap='gray')
# plt.show()

# # TODO: Ideas:
# # 1D Conv and 2D Conv
# # Hidden size
# # Number layers
# # Visualization

