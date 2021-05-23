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

architecture = Conv2DNet
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
print("Finished loading data.")

model = architecture(**model_args)
print(model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


min_valid_loss = np.inf
for epoch in range(num_epochs):
    train_loss = 0.0
    # training loop (iterates over training batches)
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_train.float())
        loss_train = loss(y_pred, y_train)
        if epoch % 10 == 9:
            print('Epoch {}  -  Train loss: {}'.format(epoch, loss_train.item()))
        loss_train.backward()
        optimizer.step()

        train_loss = loss_train.item() * len(batch[0])

    valid_loss = 0.0
    for batch in valid_loader:
        x_valid = batch[0]
        y_valid = batch[1]

        optimizer.zero_grad()
        y_pred = model(x_valid.float())
        loss_valid = loss(y_pred, y_valid)

        loss_valid.backward()
        optimizer.step()

        valid_loss = loss_valid.item() * len(batch[0])

    print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')

    # model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for x_valid, y_valid in valid_loader:
    #         # validation loop
    #         y_pred = model(x_valid.float())
    #         # loss
    #         loss_valid = loss(y_pred, y_valid)
    #         # accuracy
    #         _, predicted = torch.max(y_pred.data, 1)
    #         total += y_valid.size(0)
    #         correct += (predicted == y_valid).sum().item()
    #     print('Epoch {}  -  Validation Accuracy: {}%'.format(epoch, 100 * correct / total))
    #     precision, recall, fscore, support = precision_recall_fscore_support(y_valid, predicted, average='macro')
    #     print('Precision (macro): {}  -  Recall (macro): {}  -  F-score (macro): {}%'.format(precision, recall, fscore))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for img, labels in test_loader:
        outputs = model(img.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy: {}%'.format(100 * correct / total))

# # images: shape 48 x 48
# img = x_train[0].flatten().reshape(48, 48)
# plt.imshow(img, cmap='gray')
# plt.show()

# # TODO: Ideas:
# # 1D Conv and 2D Conv
# # Hidden size
# # Number layers
# # Visualization

