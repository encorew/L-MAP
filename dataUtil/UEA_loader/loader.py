import random

import numpy as np

from dataUtil.UEA_loader import uea_ucr_datasets


class Multi_variate_data:
    def __init__(self, dataset_name, data_dir, train=False):
        processed_dataframe = uea_ucr_datasets.Dataset(dataset_name, data_dir, train)
        self.n_classes = processed_dataframe.n_classes
        self.scattered_data = processed_dataframe
        self.timeseries_set = {}
        for multi_variate_ts in self.scattered_data:
            class_no = multi_variate_ts[1]
            if class_no not in self.timeseries_set:
                self.timeseries_set[class_no] = []
            else:
                self.timeseries_set[class_no].append(multi_variate_ts[0])

    def ordered_concatenate(self):
        for class_no in self.timeseries_set:
            self.timeseries_set[class_no] = np.array(self.timeseries_set[class_no])
            self.timeseries_set[class_no] = np.concatenate(self.timeseries_set[class_no], axis=0)

    def random_concatenate(self):
        for class_no in self.timeseries_set:
            self.timeseries_set[class_no] = random.shuffle(self.timeseries_set[class_no])
            self.timeseries_set[class_no] = np.array(self.timeseries_set[class_no])
            self.timeseries_set[class_no] = np.concatenate(self.timeseries_set[class_no], axis=0)



if __name__ == '__main__':
    list1 = [1, 2, 3, 4, 5, 6]
    random.shuffle(list1)
    print(list1)
