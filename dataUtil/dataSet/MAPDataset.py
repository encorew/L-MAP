import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from decomposition import multi_fft
from distance import normal_distance, avg_distance
from mp_algorithm import multi_mp_matrix
from plot_show import draw_anomaly
from sklearn.preprocessing import MinMaxScaler


class MAPDataset(data.Dataset):
    def __init__(self, dataname, dataset, window_length, window_stride, labels=None):
        self.dataname = dataname
        self.raw_dataset = dataset
        self.labels = labels
        self.total_length = dataset.shape[0]
        self.window_length = window_length
        self.window_stride = window_stride
        self.data = []
        for i in range(0, self.total_length, window_stride):
            if i + window_length >= self.total_length - window_stride:
                break
            self.data.append(dataset[i:i + window_length])
        self.mp_distances = self.freq_distance()
        self.mp_distances = self.normalization(self.mp_distances)
        self.draw_dataset()

    def freq_distance(self, load=False, save=True):
        self.freqs = []
        print("start generating freq")
        for sub_seq in tqdm(self.data):
            self.freqs.append(multi_fft(sub_seq))
        self.freqs = np.array(self.freqs)
        self.freqs = np.nan_to_num(self.freqs)
        distance_mat = []
        for i in tqdm(range(self.freqs.shape[0])):
            min_distance = float("inf")
            for j in range(self.freqs.shape[0]):
                if i == j:
                    continue
                min_distance = min(avg_distance(self.freqs[i], self.freqs[j]), min_distance)
            for j in range(self.window_stride):
                distance_mat.append(min_distance)
        distance_mat = np.array(distance_mat)
        if save:
            os.makedirs(f"saved_freq/{self.dataname}", exist_ok=True)
            np.save(f"saved_freq/{self.dataname}/window{self.window_length}stride{self.window_stride}.npy", self.freqs)
            np.save(f"saved_freq/{self.dataname}/window{self.window_length}stride{self.window_stride} distance.npy",
                    distance_mat)
        return distance_mat

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def draw_dataset(self):
        # multi_matrix_profile = multi_mp(self.raw_dataset, self.window_length)
        plt.subplot(3, 1, 1)
        plt.plot(self.mp_distances)
        plt.subplot(3, 1, 2)
        # plt.plot(multi_matrix_profile)
        plt.subplot(3, 1, 3)
        if self.labels is not None:
            draw_anomaly(self.raw_dataset[:, 0], self.labels)
        else:
            plt.plot(self.raw_dataset[:, 0])
        plt.show()

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).float(), torch.tensor(self.mp_distances[index]).float()

    def __len__(self):
        return len(self.data)
