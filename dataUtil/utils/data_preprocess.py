import os
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


def normalization(dataset):
    normalizer = MinMaxScaler(feature_range=(-1, 1))
    if len(dataset.shape) == 1:
        dataset = dataset.reshape(-1, 1)
    return normalizer.fit_transform(dataset), normalizer


def inverse_normalization(dataset, normalizer):
    return normalizer.inverse_transform(dataset)


def read_data(data_dir, data_name, mode=None):
    if not mode:
        data_path = os.path.join(data_dir, data_name)
        extension = data_name.split('.')[-1]
        if extension == "pkl":
            fr = open(data_path, 'rb+')
            matrix_data = pickle.load(fr)
            return matrix_data
        elif extension == "csv":
            data_frame = pd.read_csv(data_path)
            return data_frame.values
        elif extension == "txt":
            return np.loadtxt(data_path)


def extract_raw_data_from_frame(csv_data_frame, columns_col_number, is_time=False):
    if not is_time:
        return np.array(csv_data_frame.iloc[:, columns_col_number])
    else:
        time_df = pd.to_datetime(csv_data_frame.iloc[:, columns_col_number].astype('str'))
        return np.array(time_df)


def split_data_by_window(data, window_size):
    total_num_steps = data.shape[0]
    sequence_list = [data[i:i + window_size] for i in range(total_num_steps - window_size + 1)]
    return sequence_list


def get_raw_data(data_dir, file_name, columns_num, time_column_num=0,
                 normalize=True):
    normalizer = None
    data_frame = read_data(data_dir, file_name)
    columns_name = data_frame.columns[columns_num]
    time = extract_raw_data_from_frame(data_frame, time_column_num, is_time=True)
    raw_data = extract_raw_data_from_frame(data_frame, columns_num)
    if normalize:
        raw_data, normalizer = normalization(raw_data)
    return raw_data, time, columns_name, normalizer


def save_pkl_data(save_dir, file_name, data):
    with open(os.path.join(save_dir, file_name), 'wb') as file:
        pickle.dump(data, file)


def add_noisy(data, ratio):
    total_num = len(data)
    noisy_count = total_num * ratio


def generate_anomaly_labels_for_noisy_data(length):
    anomaly_labels = []
    anomaly_counts = 0
    for i in range(500):
        anomaly_labels.append(0)
    i = 0
    anomaly_length = [1, 2, 3, 10]
    anomaly_interval = [30, 70, 100, 200]
    while i < length:
        x0 = random.randint(10, 20)
        x1 = random.randint(20, 50)
        x2 = random.randint(50, 100)
        x3 = random.randint(100, 200)
        if i % x0 == 0 or i % x1 == 0 or i % x2 == 0 or i % x3 == 0:
            current_anomaly_length = random.choice(anomaly_length)
            for j in range(min(current_anomaly_length, length - i)):
                anomaly_labels.append(1)
                anomaly_counts += 1
                i += 1
            current_interval = random.choice(anomaly_interval)
            for j in range(min(current_interval, length - i)):
                anomaly_labels.append(0)
                i += 1
        else:
            anomaly_labels.append(0)
            i += 1
    print(anomaly_counts / length)
    return anomaly_labels


def add_gaussian_noise(data, anomaly_labels):
    mu = 0
    sigma = 0.4
    for i in range(len(data)):
        if anomaly_labels[i] == 1:
            data[i] += random.gauss(mu, sigma)


if __name__ == "__main__":
    dir_path = "../data"
    pure_train_data = read_data(dir_path, "SMD_train.pkl")

    # data, time, _, _ = get_raw_data(dir_path, file_name + ".csv", [6], normalize=True)
    # train_data = data[:len(data) // 5 * 4]
    # test_data = data[len(data) // 5 * 4:]
    anomaly_labels = generate_anomaly_labels_for_noisy_data(len())
    # anomaly_labels = np.array(anomaly_labels)
    # add_gaussian_noise(test_data, anomaly_labels)
    # with open(os.path.join(dir_path, file_name + "_test.pkl"), 'wb') as file:
    #     pickle.dump(test_data, file)
    # with open(os.path.join(dir_path, file_name + "_test_label.pkl"), 'wb') as file:
    #     pickle.dump(anomaly_labels, file)
    # with open(os.path.join(dir_path, file_name + "_train.pkl"), 'wb') as file:
    #     pickle.dump(train_data, file)
    # draw_anomaly(test_data, anomaly_labels)
    # test_data = read_data(dir_path, file_name + "_test.pkl")
    # test_labels = read_data(dir_path, file_name + "_test_label2.pkl")
    # test_labels = test_labels[:len(test_data)]
    # with open(os.path.join(dir_path, file_name + "_test_label.pkl"), 'wb') as file:
    #     pickle.dump(test_labels, file)
    # draw_anomaly(test_data, test_labels)
