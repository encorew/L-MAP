import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import dataloader, random_split

from dataUtil.UEA_loader.loader import Multi_variate_data
from dataUtil.dataSet.dataset import TSCDataset


def replace_nan_in_data(raw_ts_dataset):
    nan_free_dataset = []
    for data in raw_ts_dataset:
        nan_free_dataset.append((np.nan_to_num(data[0]), data[1]))
    return nan_free_dataset


def series_info(train, test):
    print(f"-> dim: {train[0][0].shape[1]}")
    print(f"-> trainset length: {train[0][0].shape[0]} testset length:{test[0][0].shape[0]}")
    print(f"-> train num:{len(train)} test num:{len(test)} ratio {round(len(train) / len(test), 2)}:1.0")


def draw_series(dataset, dim=-1, show_classes=4):
    if dim == -1:
        dim = min(dataset[0][0].shape[1], 4)
        dim = [i for i in range(dim)]
    show_classes = min(show_classes, len(dataset))
    for i in range(show_classes):
        plt.subplot(show_classes * 100 + 10 + i + 1)
        plt.plot(dataset[i][0][:, dim])
    plt.show()


def concatenate_data_samples(dataset, concat_mode=0):
    prev_label = dataset[0][1]
    concatenated_series = {}
    prev_samples = []
    for series_sample, label in dataset:
        if label not in concatenated_series:
            concatenated_series[label] = series_sample
        else:
            concatenated_series[label] = np.concatenate([concatenated_series[label], series_sample], axis=0)
    concatenated_series = [(concatenated_series[label], label) for label in concatenated_series]
    return concatenated_series, len(concatenated_series)


def load_and_process(dataset_name, dataset_dir, show_info=True):
    train_ts_dataset = Multi_variate_data(dataset_name=dataset_name, data_dir=dataset_dir,
                                          train=True)
    raw_train_ts_dataset = train_ts_dataset.scattered_data
    raw_train_ts_dataset = replace_nan_in_data(raw_train_ts_dataset)
    raw_test_ts_dataset = Multi_variate_data(dataset_name=dataset_name, data_dir=dataset_dir,
                                             train=False).scattered_data
    raw_test_ts_dataset = replace_nan_in_data(raw_test_ts_dataset)
    print(f"current data:{dataset_name}")
    if show_info:
        series_info(raw_train_ts_dataset, raw_test_ts_dataset)
    return raw_train_ts_dataset, raw_test_ts_dataset, raw_train_ts_dataset[0][0].shape[1], \
           raw_train_ts_dataset[0][0].shape[0], train_ts_dataset.n_classes


def load_data_to_loader(train_data, test_data, batch_size, valid_portion=0.1):
    train_set = TSCDataset(train_data)
    test_set = TSCDataset(test_data)
    train_set, valid_set = random_split(dataset=train_set,
                                        lengths=[len(train_set) - int(len(train_set) * valid_portion),
                                                 int(len(train_set) * valid_portion)])
    train_iter = dataloader.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_iter = dataloader.DataLoader(dataset=valid_set, batch_size=1, shuffle=True,
                                       drop_last=True)
    test_iter = dataloader.DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=False)
    return train_iter, valid_iter, test_iter


if __name__ == '__main__':
    train_data, test_data, dimension, instance_length, n_classes = load_and_process('ArticularyWordRecognition',
                                                                                    '../classification_data/UEA')
    train_iter, valid_iter, test_iter = load_data_to_loader(train_data, test_data, batch_size=16)
    for train_sample, train_label in train_iter:
        print(train_label)
