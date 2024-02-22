import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from mylsh.lshash import LSHash
from torch.utils.data import dataloader, random_split
from dataUtil.dataSet.MShapeDataset import MShapeDataset
from dataUtil.load import load_and_process, concatenate_data_samples
from frequency import multi_fft
from model.gcn import GCN_simple
from model.loss.triple_loss import TripleLoss
from sklearn.cluster import KMeans


def seq_distance(subseq1, subseq2):
    if len(subseq1) == len(subseq2):
        distance = subseq1 - subseq2
        return np.sum(np.square(distance))
    if len(subseq1) < len(subseq2):
        subseq1, subseq2 = subseq2, subseq1
    min_distance = float("inf")
    for i in range(len(subseq1) - len(subseq2)):
        distance = subseq1[i:i + len(subseq2)] - subseq2
        distance = np.sum(distance ** 2)
        min_distance = min(min_distance, distance)
    return min_distance


def generate_subsequence_candidates(train_data, length, window_stride, type="classification"):
    if type == "classification":
        candidate_dict = {}
        for series_sample, series_class in train_data:
            current_class_candidates = [series_sample[i:i + length].T for i in
                                        range(0, len(series_sample) - length, window_stride)]
            if series_class in candidate_dict:
                candidate_dict[series_class] = candidate_dict[series_class] + current_class_candidates
            else:
                candidate_dict[series_class] = current_class_candidates
        return candidate_dict
    else:
        candidates = [train_data[i:i + length].T for i in range(0, len(train_data) - length, window_stride)]
        return candidates


def generate_train_pair(samples_dict):
    train_pairs = []
    for series_class in samples_dict:
        for subseq in samples_dict[series_class]:
            min_in_class_pair_distance = float("inf")
            min_cross_class_pair_distance = float("inf")
            current_pair = [subseq, subseq, subseq]
            for compare_class in samples_dict:
                if compare_class != series_class:
                    for compare_seq in samples_dict[compare_class]:
                        cross_class_distance = seq_distance(subseq, compare_seq)
                        if cross_class_distance < min_cross_class_pair_distance:
                            min_cross_class_pair_distance = cross_class_distance
                            current_pair[1] = compare_seq
                else:
                    for compare_seq in samples_dict[compare_class]:
                        in_class_distance = seq_distance(subseq, compare_seq)
                        if 0 < in_class_distance < min_in_class_pair_distance:
                            min_in_class_pair_distance = in_class_distance
                            current_pair[2] = compare_seq
            train_pairs.append(current_pair)
    return train_pairs


def generate_train_triple_by_frequency(subseqs_set, hash_size=5, series_num=-1):
    train_triples = []
    if series_num != -1:
        subseqs = subseqs_set[series_num]
    else:
        subseqs = subseqs_set
    seq_length = subseqs[0].shape[1]
    dimension = subseqs[0].shape[0]
    lsh = LSHash(hash_size, (seq_length // 2 - 1) * dimension)
    for subseq in subseqs:
        current_freq = multi_fft(subseq)
        lsh.index(current_freq.flatten(), subseq.flatten())
    all_candidates = [tuple(subseq.flatten()) for subseq in subseqs]
    for subseq in tqdm(subseqs):
        current_freq = multi_fft(subseq)
        positive_samples = [candidate[0][1] for candidate in lsh.query(current_freq.flatten())]
        negative_samples = list(set(all_candidates) - set(positive_samples))
        positive_sample = positive_samples[0]
        if len(positive_samples) > 1:
            positive_sample = positive_samples[1]
        if not positive_sample:
            continue
        triple_positive_sample = np.array(positive_sample).reshape(subseq.shape[0], -1)
        triple_negative_sample = np.array(random.choice(negative_samples)).reshape(subseq.shape[0], -1)
        current_triple = [subseq, triple_negative_sample, triple_positive_sample]
        train_triples.append(current_triple)
    return train_triples


def triple_loss_train(model, train_pairs, learning_rate, batch_size, num_epochs, device, data_name, state_name):
    loss_fn = TripleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = MShapeDataset(train_pairs)
    train_dataset, valid_dataset = random_split(dataset=dataset,
                                                lengths=[len(dataset) - len(dataset) // 10, len(dataset) // 10])
    train_iter = dataloader.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_iter = dataloader.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()
    min_valid_loss = float("inf")
    for epoch in range(num_epochs):
        l = 0
        for anchor, negative, positive in train_iter:
            anchor, negative, positive = anchor.to(device), negative.to(device), positive.to(device)
            anchor_embed = model(anchor, batch_size, device)
            negative_embed = model(negative, batch_size, device)
            positive_embed = model(positive, batch_size, device)
            loss = loss_fn(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            l += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        # valid_loss = validate(model, valid_iter, loss_fn, 1, device)
        if len(valid_iter) < 1:
            valid_loss = l / len(train_iter)
        else:
            valid_loss = validate(model, valid_iter, loss_fn, batch_size, device) / len(valid_iter)
        print(f"\n==> current epoch:{epoch} loss:{l / len(train_iter)} valid_loss:{valid_loss}")
        if 0 <= valid_loss < min_valid_loss:
            print("save current state")
            min_valid_loss = valid_loss
            os.makedirs(f"saved_state/model_parameters/{data_name}", exist_ok=True)
            torch.save(model.state_dict(),
                       f"saved_state/model_parameters/{data_name}/{state_name}.pth")


def validate(model, valid_iter, loss_fn, batch_size, device):
    l = 0
    for anchor, negative, positive in valid_iter:
        anchor, negative, positive = anchor.to(device), negative.to(device), positive.to(device)
        anchor_embed = model(anchor, batch_size, device)
        negative_embed = model(negative, batch_size, device)
        positive_embed = model(positive, batch_size, device)
        loss = loss_fn(anchor_embed, positive_embed, negative_embed)
        l += loss.item()
    return l


def cluster(model, candidates, cluster_num, device, class_num=-1):
    embeddings = []
    if class_num != -1:
        candidates = candidates[class_num]
    for series in candidates:
        series = torch.tensor(series).to(device)
        series_embedding = model(series, 1, device)
        embeddings.append(series_embedding.detach().view(-1).cpu().numpy())
    embeddings = np.array(embeddings)
    k_means = KMeans(n_clusters=cluster_num, random_state=10)
    k_means.fit(embeddings)
    y_predict = k_means.predict(embeddings)
    cluster_centers = k_means.cluster_centers_
    # plt.show()
    return cluster_centers


def embedding_distance(embedding1, embedding2):
    return np.sum(np.square(embedding1 - embedding2), axis=1)


def cluster_distance(model, series_sample, cluster_centers, device):
    series = torch.tensor(series_sample.T).to(device)
    series_embedding = model(series, 1, device).detach().cpu().numpy()
    embedding_distances = embedding_distance(series_embedding, cluster_centers)
    min_distance = np.min(embedding_distances)
    cluster_num = np.argmin(embedding_distances)
    # out_class_distance = 0
    # for class_no in cluster_centers:
    #     if current_class != class_no:
    #         out_class_distance += np.sum(embedding_distance(series_embedding, cluster_centers[current_class]))
    return min_distance, cluster_num


def generate_profile_value(model, series, seq_length, cluster_centers, cluster_num, device):
    profile_value = []
    profile_pattern = []
    pattern_distance_dict = {}
    indice_dict = {}
    for i in range(cluster_num):
        pattern_distance_dict[i] = []
        indice_dict[i] = []
    for i in tqdm(range(len(series) - seq_length)):
        distance, match_pattern = cluster_distance(model, series[i:i + seq_length, :], cluster_centers, device)
        profile_value.append(distance)
        profile_pattern.append(match_pattern)
        pattern_distance_dict[match_pattern].append(distance)
        indice_dict[match_pattern].append(i)
    for i in range(cluster_num):
        pattern_distance_dict[i] = np.array(pattern_distance_dict[i])
        indice_dict[i] = np.array(indice_dict[i])
    return np.array(profile_value), np.array(profile_pattern), pattern_distance_dict, indice_dict


def main():
    # DATASET_LIST = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories',
    #                 'Cricket',
    #                 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'ERing', 'EthanolConcentration', 'FingerMovements',
    #                 'FaceDetection',
    #                 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'JapaneseVowels', 'LSST', 'MotorImagery',
    #                 'NATOPS', 'PenDigits', 'Phoneme', 'RacketSports', 'SelfRegulationSCP1', 'StandWalkJump',
    #                 'UWaveGestureLibrary']
    DATA_FILE = 'ArticularyWordRecognition'
    DATA_DIR = 'dataset/Multivariate2018_ts/Multivariate_ts'
    train_data, test_data, dimension, instance_length, n_classes = load_and_process(DATA_FILE, DATA_DIR)
    concatenated_data, _ = concatenate_data_samples(train_data)

    HIDDEN_SIZE = 64
    SEQ_LENGTH = 100
    WINDOW_STRIDE = SEQ_LENGTH // 2
    CLUSTER_NUM = 2
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    EMBEDDING_DIM = 15
    SERIES_NUM = 1
    TRAIN = False
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    criterion = torch.nn.MSELoss()
    demonstrate_data = concatenated_data
    candidates = generate_subsequence_candidates(demonstrate_data, SEQ_LENGTH, WINDOW_STRIDE)
    triple = generate_train_triple_by_frequency(candidates, hash_size=5, series_num=0)
    model = GCN_simple(input_series_length=SEQ_LENGTH, input_series_dim=dimension, hidden_channels=HIDDEN_SIZE,
                       output_dimension=EMBEDDING_DIM)
    model = model.to(device)
    triple_loss_train(model, triple, learning_rate=0.001, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                      device=device, name=DATA_FILE, train=TRAIN)
    cluster_centers = cluster(model, candidates, cluster_num=CLUSTER_NUM, device=device, class_num=0)
    my_profile, profile_pattern, pattern_distance_dict, indice_dict = generate_profile_value(model,
                                                                                             demonstrate_data[0][0],
                                                                                             SEQ_LENGTH,
                                                                                             cluster_centers,
                                                                                             CLUSTER_NUM,
                                                                                             device)
    plt.plot(my_profile, c="green")
    plt.show()


if __name__ == '__main__':
    main()
