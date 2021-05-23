import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import Conv1DNet, Conv2DNet

def load_data():
    print('Load data...')
    DATA_DIR = "./data/"
    dataset_path = os.path.join(DATA_DIR, 'data.npy')
    if not os.path.exists(dataset_path):

        data_cv = pd.read_csv('data/fer2013/fer2013/fer2013.csv').to_numpy()
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


def preprocess_data(data, architecture):

    # training, test, validation (x data shape: sample size, channel size, height, width)
    train = np.expand_dims(data[np.where(data[:, -1] == 2)], axis=2)  # Training
    x_train = torch.from_numpy(train[:, 1:-1, :])
    y_train = torch.from_numpy(train[:, 0, :].flatten()).type(torch.LongTensor)
    # y_train = torch.nn.functional.one_hot(y_train_int[:, 0].to(torch.int64))

    test = np.expand_dims(data[np.where(data[:, -1] == 0)], axis=2)  # PrivateTest
    x_test = torch.from_numpy(test[:, 1:-1, :])
    y_test = torch.from_numpy(test[:, 0, :].flatten()).type(torch.LongTensor)
    # y_test = torch.nn.functional.one_hot(y_test_int[:, 0].to(torch.int64))

    valid = np.expand_dims(data[np.where(data[:, -1] == 1)], axis=2)  # PublicTest
    x_valid = torch.from_numpy(valid[:, 1:-1, :])
    y_valid = torch.from_numpy(valid[:, 0, :].flatten()).type(torch.LongTensor)
    # y_valid = torch.nn.functional.one_hot(y_valid_int[:, 0].to(torch.int64))

    if architecture == Conv2DNet:

        # reshape 3-dim input to 4-dim input
        x_train_2D = x_train.reshape(x_train.shape[0], int(np.sqrt(x_train.shape[1])), int(np.sqrt(x_train.shape[1])),
                                     x_train.shape[2])
        x_train_2D = x_train_2D.reshape(x_train_2D.shape[0], x_train_2D.shape[3], x_train_2D.shape[1], x_train_2D.shape[2])

        x_test_2D = x_test.reshape(x_test.shape[0], int(np.sqrt(x_test.shape[1])), int(np.sqrt(x_test.shape[1])),
                                   x_test.shape[2])
        x_test_2D = x_test_2D.reshape(x_test_2D.shape[0], x_test_2D.shape[3], x_test_2D.shape[1], x_test_2D.shape[2])

        x_valid_2D = x_valid.reshape(x_valid.shape[0], int(np.sqrt(x_valid.shape[1])), int(np.sqrt(x_valid.shape[1])),
                                     x_valid.shape[2])
        x_valid_2D = x_valid_2D.reshape(x_valid_2D.shape[0], x_valid_2D.shape[3], x_valid_2D.shape[1], x_valid_2D.shape[2])


        return x_train_2D, y_train, x_valid_2D, y_valid, x_test_2D, y_test
    else:
        return x_train, y_train, x_valid, y_valid, x_test, y_test


def get_data_loader(data, batch_size, architecture, shuffle, drop_last):

    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess_data(data, architecture)

    train_set = TensorDataset(x_train, y_train)
    valid_set = TensorDataset(x_valid, y_valid)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return train_loader, valid_loader, test_loader
