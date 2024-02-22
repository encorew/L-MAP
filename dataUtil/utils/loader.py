import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import random_split

from data_preprocess import read_data, split_data_by_window


def generate_data_iter(X, Y, batch_size, validation_split=0, shuffle=False, drop_last=True):
    print(X.shape, Y.shape)
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    if len(Y.shape) == 2:
        Y = Y.reshape(Y.shape[0], 1, Y.shape[1])
    data_set = Data.TensorDataset(X, Y)
    # 没有验证集,只生成训练集
    if validation_split == 0:
        data_iter = Data.DataLoader(data_set, batch_size, shuffle, drop_last=drop_last)
        return data_iter
    else:
        data_length = len(data_set)
        split = [data_length - data_length // validation_split, data_length // validation_split]
        train_dataset, test_dataset = random_split(
            dataset=data_set,
            lengths=split
        )
        train_iter = Data.DataLoader(train_dataset, batch_size, shuffle, drop_last=drop_last)
        validation_iter = Data.DataLoader(test_dataset, batch_size, shuffle, drop_last=drop_last)
        return train_iter, validation_iter


def generate_data_iter_from_time_series(raw_data, window_size, batch_size, mul2mul=True, validation_split=0,
                                        shuffle=False):
    # window of points -> window of points
    if mul2mul:
        sequence_list = split_data_by_window(raw_data, window_size)
        train_data = [list_data[:-1] for list_data in sequence_list]
        train_label = [list_data[1:] for list_data in sequence_list]
    # window of points -> single point
    else:
        sequence_list = split_data_by_window(raw_data, window_size + 1)
        train_data = [list_data[:-1] for list_data in sequence_list]
        train_label = [list_data[-1] for list_data in sequence_list]
    train_data, train_label = np.array(train_data), np.array(train_label)
    return generate_data_iter(train_data, train_label, batch_size, validation_split, shuffle, drop_last=True)
