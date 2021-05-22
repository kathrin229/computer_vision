import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import dataset
from models import Conv1DNet, Conv2DNet

# TODO check seed - reproducibility
torch.seed()
torch.manual_seed(0)

architecture = Conv2DNet
num_epochs = 10
learning_rate = 0.0001
batch_size = 64
model_args = {
    'input_size': 48,
    'hidden_size': 32,
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

print(model)

for epoch in range(num_epochs):
    for batch in train_loader:
        x_train = batch[0]
        y_train = batch[1]

        optimizer.zero_grad()
        y_pred = model(x_train.float())
        # y_valid = model()
        loss_train = loss(y_pred, y_train)

        if epoch % 10 == 9:
            print('Epoch {}:  Train loss: {}'.format(epoch, loss_train.item()))

        loss_train.backward()
        optimizer.step()

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

