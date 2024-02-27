import os

import numpy as np
from matplotlib import pyplot as plt

from data_preprocess import read_data, save_pkl_data
from sklearn.preprocessing import MinMaxScaler


class Plot:
    def __init__(self, raw_data, prediction=np.array([0, 0]), time=np.array([0, 0]), anomaly_indice=np.array([0, 0])):
        self.raw_data = raw_data
        self.prediction = prediction
        self.anomaly_indice = anomaly_indice
        self.time = time
        if time == np.array([0, 0]):
            self.time = range(len(raw_data))

    def draw(self, select_dim=0):
        if len(self.raw_data.shape) > 1:
            plt.plot(self.time, self.raw_data[:, select_dim])
            if self.prediction != np.array([0, 0]):
                plt.plot(self.time, self.prediction[:, select_dim])
        else:
            plt.plot(self.time, self.raw_data)
            if self.prediction != np.array([0, 0]):
                plt.plot(self.time, self.prediction)
        plt.xticks(rotation=45)
        # plt.show()


def draw_anomaly(raw_data, labels):
    plt.plot(raw_data)
    anomaly_regions = []
    anomaly_region_start = False
    region_left = 0
    region_right = 0
    for idx, label in enumerate(labels):
        if not anomaly_region_start and label == 1:
            anomaly_region_start = True
            region_left = idx - 0.2
        elif anomaly_region_start and label == 0:
            anomaly_region_start = False
            region_right = idx - 0.8
            anomaly_regions.append((region_left, region_right))
    for region in anomaly_regions:
        plt.axvspan(xmin=region[0], xmax=region[1], facecolor="r", alpha=0.3)
    print(anomaly_regions)
    plt.show()


if __name__ == '__main__':
    # data = read_data('../../data', 'MSL_test.pkl')
    # labels = read_data('../../data', 'MSL_test_label.pkl')
    # draw_anomaly(data, labels)
    dataset_dir = "../data"
    sample_name = "ISONE"
    process = False
    if not process:
        train_data_name = sample_name + "_train.pkl"
        test_data_name = sample_name + "_test.pkl"
        test_label = sample_name + "_test_label.pkl"
        train_data = read_data(data_dir=dataset_dir, data_name=train_data_name)
        test_data = read_data(data_dir=dataset_dir, data_name=test_data_name)
        test_label = read_data(data_dir=dataset_dir, data_name=test_label)
        plt.plot(train_data[:, 0])
        plt.show()
        draw_anomaly(test_data[:, 0], test_label)
        anomaly_count = np.sum(test_label == 1)
        total_length = len(train_data) + len(test_data)
        print("total_length:", total_length, "anomaly ratio:", anomaly_count / len(test_data))
    else:
        train_dir_path = f"dataset\ECG and PowerDemand\\{dataset_dir}\labeled\\train"
        test_dir_path = f"dataset\ECG and PowerDemand\\{dataset_dir}\labeled\\test"
        save_file_path = f"dataset\ECG and PowerDemand\\{dataset_dir}\processed"
        os.makedirs(save_file_path, exist_ok=True)
        train_data = read_data(train_dir_path, sample_name + '.pkl')
        test_data = read_data(test_dir_path, sample_name + '.pkl')
        train_data, test_data = np.array(train_data), np.array(test_data)
        if np.max(test_data) > 10:
            min_max_scaler = MinMaxScaler()
            train_data = min_max_scaler.fit_transform(train_data)
            test_data = min_max_scaler.fit_transform(test_data)
        test_labels = test_data[:, 2]
        save_pkl_data(save_file_path, sample_name + '_train.pkl', train_data[:, :2])
        save_pkl_data(save_file_path, sample_name + '_test.pkl', test_data[:, :2])
        save_pkl_data(save_file_path, sample_name + '_test_label.pkl', test_labels)
        # data = read_data('../data','power_data.txt')
        # plt.plot(data)
        train_data = read_data(save_file_path, sample_name + "_train.pkl")
        test_data = read_data(save_file_path, sample_name + "_test.pkl")
        test_labels = read_data(save_file_path, sample_name + "_test_label.pkl")
        plt.title("train")
        plt.plot(train_data)
        plt.show()
        plt.title("test")
        draw_anomaly(test_data, test_labels)
