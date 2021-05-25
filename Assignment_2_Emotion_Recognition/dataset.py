import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


###############################################################################
# Loading the dataset and saving the an np array with the right shape and types
###############################################################################
def load_data(src, dest):
    print('Load data...')
    dataset_path = dest
    if not os.path.exists(dataset_path):

        data_cv = pd.read_csv(src).to_numpy()
        print(data_cv.shape)

        # convert "Training", "PrivateTest", "PublicTest" to ints
        unique_labels = np.unique(data_cv[:, 2])
        number_dict = dict(zip(unique_labels, range(len(unique_labels))))
        data_cv[:, 2] = np.array([number_dict[k] for k in data_cv[:, 2]])

        # convert string pixel values to ints
        data = np.zeros((data_cv.shape[0], len(data_cv[0, 1].split(' ')) + 2))
        data[:, 0] = data_cv[:, 0]
        data[:, len(data_cv[0, 1].split(' ')) + 1] = data_cv[:, 2]
        data[:, 1:len(data_cv[0, 1].split(' ')) + 1] = np.array([s.split(' ') for s in data_cv[:, 1]]).astype(int)

        np.save(dataset_path, data)

        return data

    else:
        return np.load(dataset_path)


###############################################################################
# Preprocessing the data array for 1D or 2D convolutional layers
###############################################################################
def preprocess_data(data, architecture):

    # training, test, validation (x data shape: sample size, channel size, height, width)
    train = np.expand_dims(data[np.where(data[:, -1] == 2)], axis=2)  # Training
    x_train = torch.from_numpy(train[:, 1:-1, :])
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
    y_train = torch.from_numpy(train[:, 0, :].flatten()).type(torch.LongTensor)

    test = np.expand_dims(data[np.where(data[:, -1] == 0)], axis=2)  # PrivateTest
    x_test = torch.from_numpy(test[:, 1:-1, :])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])
    y_test = torch.from_numpy(test[:, 0, :].flatten()).type(torch.LongTensor)

    valid = np.expand_dims(data[np.where(data[:, -1] == 1)], axis=2)  # PublicTest
    x_valid = torch.from_numpy(valid[:, 1:-1, :])
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[2], x_valid.shape[1])
    y_valid = torch.from_numpy(valid[:, 0, :].flatten()).type(torch.LongTensor)

    if architecture.__name__.startswith("Conv2D"):

        # reshape 3-dim input to 4-dim input
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],
                                  int(np.sqrt(x_train.shape[2])), int(np.sqrt(x_train.shape[2])))

        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],
                                int(np.sqrt(x_test.shape[2])), int(np.sqrt(x_test.shape[2])))

        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1],
                                  int(np.sqrt(x_valid.shape[2])), int(np.sqrt(x_valid.shape[2])))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


###############################################################################
# Returns pytorch dataloader for Train, Test and Validation Set
###############################################################################
def get_data_loader(data, batch_size, architecture, shuffle, drop_last):

    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess_data(data, architecture)

    train_set = TensorDataset(x_train, y_train)
    valid_set = TensorDataset(x_valid, y_valid)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return train_loader, valid_loader, test_loader
