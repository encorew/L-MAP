import random

import numpy as np

from dataUtil.UEA_loader import uea_ucr_datasets


class Multi_variate_data:
    '''
    Multi_variate_data
    这个类会调用uea_ucr_dataset，首先生成scatter_data类型，然后把同一类别的放到一个类里
    scatter_data结构 (array,class_no),(array,class_no),...(array,class_no)
    timeseries_set结构 {class1:[ts1],class2:[ts2],...,classn:[tsn]}
    '''

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
                #     把同class的多变量数据都放在一起
                self.timeseries_set[class_no].append(multi_variate_ts[0])

    def ordered_concatenate(self):
        for class_no in self.timeseries_set:
            self.timeseries_set[class_no] = np.array(self.timeseries_set[class_no])
            # 先转成numpy好进行后续的拼接
            self.timeseries_set[class_no] = np.concatenate(self.timeseries_set[class_no], axis=0)

    def random_concatenate(self):
        for class_no in self.timeseries_set:
            self.timeseries_set[class_no] = random.shuffle(self.timeseries_set[class_no])
            self.timeseries_set[class_no] = np.array(self.timeseries_set[class_no])
            # 先转成numpy好进行后续的拼接
            self.timeseries_set[class_no] = np.concatenate(self.timeseries_set[class_no], axis=0)


# print(multi_variate_dataframe)
# for multi_variate_ts in multi_variate_dataframe:
#     print(multi_variate_ts[1])

if __name__ == '__main__':
    # my_ts = Multi_variate_data('ArticularyWordRecognition', data_dir='../classification_data/UEA', train=True)
    # print(my_ts.timeseries_set)
    # my_ts.ordered_concatenate()
    # print(my_ts.timeseries_set)
    # for class_no in my_ts.timeseries_set:
    #     print(my_ts.timeseries_set[class_no].shape)
    list1 = [1, 2, 3, 4, 5, 6]
    random.shuffle(list1)
    print(list1)
