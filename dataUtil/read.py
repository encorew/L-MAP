from matplotlib import pyplot as plt

from data_preprocess import read_data


def read_dataset(dataset_name, valid_portion=0, test_portion=0, show=True):
    if dataset_name == "electricity" or dataset_name == "elec":
        data_path = "dataset/prediction data/electricity"
        data_name = "electricity.csv"
    elif "ETT" in dataset_name:
        data_path = "dataset/prediction data/ETT-small"
        if dataset_name == "ETTh1":
            data_name = "ETTh1.csv"
        elif dataset_name == "ETTh2":
            data_name = "ETTh2.csv"
        elif dataset_name == "ETTm1":
            data_name = "ETTm1.csv"
        elif dataset_name == "ETTh2":
            data_name = "ETTh2.csv"
    elif dataset_name == "exchange_rate" or "exch" in dataset_name:
        data_path = "dataset/prediction data/exchange_rate"
        data_name = "exchange_rate.csv"
    elif dataset_name == "illness" or "ill" in dataset_name:
        data_path = "dataset/prediction data/illness"
        data_name = "national_illness.csv"
    elif dataset_name == "traffic":
        data_path = "dataset/prediction data/traffic"
        data_name = "traffic.csv"
    elif dataset_name == "weather":
        data_path = "dataset/prediction data/weather"
        data_name = "weather.csv"
    dataset = read_data(data_path, data_name).values[:, 1:]
    if show:
        print(dataset)
        plt.plot(dataset[:, 0])
        plt.show()
    if valid_portion == 0 and test_portion == 0:
        return dataset
    train_length = int(len(dataset) * (1 - valid_portion - test_portion))
    valid_length = int(len(dataset) * valid_portion)
    if valid_portion != 0 and test_portion != 0:
        return dataset[:train_length], dataset[train_length:train_length + valid_length], dataset[
                                                                                          train_length + valid_length:]
    else:
        return dataset[:train_length], dataset[train_length:]
