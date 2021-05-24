import os
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import dataset
from models import Conv1DNet, Conv2DNet
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns

# TODO check seed - reproducibility
torch.seed()
torch.manual_seed(0)

architecture = Conv2DNet
num_epochs = 3
learning_rate = 0.0001
batch_size = 64
patience = 15
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

    'channel_linear': 3 * 3 * 64,
    'num_classes': 7
}

DATA_DIR = "./data/"
dataset_path = os.path.join(DATA_DIR, 'data.npy')

data = dataset.load_data(src='data/fer2013/fer2013/fer2013.csv', dest=dataset_path)
train_loader, valid_loader, test_loader = dataset.get_data_loader(data, batch_size, architecture=architecture,
                                                                  shuffle=True, drop_last=True)
print("Finished loading data.\n")

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print('Current device:', torch.cuda.get_device_name(device), '\n')
else:
    print('Failed to find GPU. Will use CPU.\n')
    device = 'cpu'

model = architecture(**model_args).to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)
print("\n")

min_valid_loss = np.inf
stopping = 0
max_val_acc = 0
train_loss_all = []
valid_loss_all = []

print("Fit model...")
for epoch in range(num_epochs):
    train_loss, train_correct, train_total = 0, 0, 0
    train_epoch_loss = []
    ######################################################
    # training loop (iterates over training batches)
    ######################################################
    for batch in train_loader:
        x_train = batch[0].to(device)
        y_train = batch[1].to(device)
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
        train_epoch_loss.append(loss_train.item())
        # calculate the accuracy
        predicted = torch.argmax(y_pred, 1)
        train_total += y_train.size(0)
        train_correct += (predicted == y_train).sum().item()

    train_loss_all.append(sum(train_epoch_loss) / len(train_epoch_loss))

    ######################################################
    # validation loop
    ######################################################
    # set the model to eval mode
    model.eval()
    valid_loss, valid_correct, valid_total = 0, 0, 0
    valid_epoch_loss = []
    # turn off gradients for validation
    with torch.no_grad():
        for batch in valid_loader:
            x_valid = batch[0].to(device)
            y_valid = batch[1].to(device)
            # forward pass
            y_pred = model(x_valid.float())
            # validation batch loss
            loss_valid = loss(y_pred, y_valid)
            # accumulate the validation loss
            valid_loss += loss_valid.item()
            valid_epoch_loss.append(loss_valid.item())
            # calculate the accuracy
            predicted = torch.argmax(y_pred, 1)
            valid_total += y_valid.size(0)
            valid_correct += (predicted == y_valid).sum().item()

    valid_loss_all.append(sum(valid_epoch_loss) / len(valid_epoch_loss))

    # print epoch results
    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    train_accuracy = train_correct / len(train_loader)
    valid_accuracy = valid_correct / len(valid_loader)
    print(f'Epoch: {epoch + 1}/{num_epochs}.. '
          f'Training loss: {train_loss}.. Validation Loss: {valid_loss}.. '
          f'Training accuracy: {train_accuracy}.. Validation accuracy: {valid_accuracy}')

    # early stopping
    if max_val_acc < valid_accuracy:
        max_val_acc = valid_accuracy
        weights = copy.deepcopy(model.state_dict())
        stopping = 0
    else:
        stopping = stopping + 1

    if stopping == patience:
        print('Early stopping...')
        print('Restoring best weights')
        model.load_state_dict(weights)
        break


def plot_train_val(epochs, loss_train, loss_val, metric, IMG_DIR):
    sns.set_style("darkgrid")
    epochs_twice = np.tile(epochs, 2)
    hue_train = ['training' for l in loss_train]
    hue_val = ['validation' for v in loss_val]
    hue = hue_train + hue_val
    loss = loss_train + loss_val
    df = pd.DataFrame({'loss': loss, 'epochs': epochs_twice, 'hue': hue})
    ax = sns.lineplot(x='epochs', y='loss', hue='hue', data=df)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(str(metric))
    ax.set_xticks(epochs)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=[h for h in handles if handles.index(h) != 0])  # remove subtitle for hue entries from legend
    if metric == "Cross Entropy":
        name = "ce"
    else:
        name = "acc"
        ax.set_yticks(np.linspace(0, 1, 11))
    plt.savefig(f"{IMG_DIR}_{name}_plot.png")
    plt.show()


# plotting training and validation loss
plot_train_val(np.linspace(1, num_epochs, num_epochs).astype(int), train_loss_all, valid_loss_all,
               metric="Cross Entropy", IMG_DIR=f'./img/{model.__class__.__name__}')

######################################################
# test loop
######################################################
print("Test model...")
# set the model to eval mode
model.eval()
# turn off gradients for validation
with torch.no_grad():
    test_loss, test_correct, test_total = 0, 0, 0
    for batch in test_loader:
        x_test = batch[0].to(device)
        y_test = batch[1].to(device)
        # forward pass
        y_pred = model(x_test.float())
        # test batch loss
        loss_test = loss(y_pred, y_test)
        # accumulate the test loss
        test_loss += loss_test.item()
        # calculate the accuracy
        predicted = torch.argmax(y_pred, 1)
        test_total += y_test.size(0)
        test_correct += (predicted == y_test).sum().item()
    print('Test Accuracy: {}%'.format(100 * test_correct / test_total))

test_loss /= len(test_loader)
accuracy = test_correct / len(test_loader)
print(f'Test loss: {loss_test}.. Test Accuracy: {accuracy}')

precision, recall, fscore, support = precision_recall_fscore_support(y_valid, predicted, average='macro')
print(f'Precision (macro): {precision}.. Recall (macro): {recall}.. F-score (macro): {fscore}')

######################################################
# visualization of feature maps for single image
######################################################
# reference: https://androidkt.com/how-to-visualize-feature-maps-in-convolutional-neural-networks-using-pytorch/

IMG_DIR = "./img/"

img = x_train[0].unsqueeze(0).type(torch.FloatTensor).to(device)

# accessing convolutional layers
num_layers = 0
conv_layers = []
model_children = list(model.children())

for child in model_children:
    if type(child) == nn.Conv2d:
        num_layers += 1
        conv_layers.append(child)
    elif type(child) == nn.Sequential:
        for layer in child.children():
            if type(layer) == nn.Conv2d:
                num_layers += 1
                conv_layers.append(layer)

# pass image through network and store results
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))
outputs = results

# plot image
plt.imshow(x_train[0].cpu().flatten().reshape(48, 48), cmap='gray')
plt.show()

# visualize feature maps of network
for num_layer in range(len(outputs)):
    fig = plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    title = "Layer %s" % (num_layer + 1)
    print(title)
    for i, conv_filter in enumerate(layer_viz):
        if i == 16:
            break
        plt.subplot(2, 8, i + 1)
        plt.imshow(conv_filter.cpu(), cmap='gray')
        plt.axis("off")
        st = fig.suptitle(title, fontsize=50)
        # shift subplots down:
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)
    plt.savefig(IMG_DIR + model.__class__.__name__ + "_layer%s_feature_maps.png" % str(num_layer + 1))
    plt.show()
    plt.close()
