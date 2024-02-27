import os
import random

import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from torch.utils.data import dataloader, random_split
from dataUtil.dataSet.MShapeDataset import MShapeDataset
from dataUtil.utils.data_preprocess import read_data
from frequency import multi_fft
from model.gcn import GCN_simple
from model.loss.triple_loss import TripleLoss
from sklearn.cluster import KMeans
from mylsh.lshash import LSHash


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


def generate_subsequence_candidates(train_data, length, window_stride, type="motif_plain"):
    if type == "motif_plain":
        candidates = [train_data[i:i + length].T for i in range(0, len(train_data) - length, window_stride)]
        return candidates
    elif type == "motif_classification":
        candidate_dict = {}
        for series_sample, series_class in train_data:
            current_class_candidates = [series_sample[i:i + length].T for i in
                                        range(0, len(series_sample) - length, window_stride)]
            if series_class in candidate_dict:
                candidate_dict[series_class] = candidate_dict[series_class] + current_class_candidates
            else:
                candidate_dict[series_class] = current_class_candidates
        return candidate_dict
    elif type == "classification":
        candidates = []
        for series_sample, series_class in train_data:
            current_instance_candidates = [series_sample[i:i + length].T for i in
                                           range(0, len(series_sample) - length, window_stride)]
            candidates = candidates + current_instance_candidates
        return candidates
    elif type == "anomaly":
        candidates = [train_data[i:i + length].T for i in range(0, len(train_data) - length, window_stride)]
        return candidates
    else:
        candidates = [train_data[i:i + length].T for i in range(0, min(len(train_data) - length, 10000), window_stride)]
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
        if not positive_sample or not negative_samples:
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
        if len(valid_iter) < 1:
            valid_loss = l / len(train_iter)
        else:
            valid_loss = validate(model, valid_iter, loss_fn, batch_size, device) / len(valid_iter)
        print(f"\n==> current epoch:{epoch} loss:{l / len(train_iter)} valid_loss:{valid_loss}")
        if 0 < valid_loss <= min_valid_loss:
            print("save current state")
            min_valid_loss = valid_loss
            os.makedirs(f"saved_state/{data_name}/{model.model_type}", exist_ok=True)
            torch.save(model.state_dict(),
                       f"saved_state/{data_name}/{model.model_type}/{state_name}.pth")


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


def cluster(model, candidates, cluster_num, device, radius=0.1, class_num=-1):
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
    cluster_centers = k_means.cluster_centers_
    distributions = {}
    for idx, cluster_center in enumerate(cluster_centers):
        distributions[idx] = [cluster_center]
    for embedding in embeddings:
        for idx, center in enumerate(cluster_centers):
            if np.sum((embedding - center) ** 2) <= radius:
                distributions[idx].append(embedding)
    for center_idx in distributions:
        kde = KernelDensity(bandwidth=0.5)
        kde.fit(np.array(distributions[center_idx]))
        distributions[center_idx] = kde
    return distributions


def embedding_distance_prev(embedding1, embedding2):
    return np.sum(np.square(embedding1 - embedding2), axis=1)


def embedding_distance(embedding, distributions):
    embedding_distances = []
    for idx in distributions:
        embedding_distances.append(1 - distributions[idx].score_samples(embedding))
    embedding_distances = np.array(embedding_distances)
    return np.min(embedding_distances), np.argmin(embedding_distances)


def cluster_distance(model, series_sample, cluster_centers, device):
    series = torch.tensor(series_sample.T).to(device)
    series_embedding = model(series, 1, device).detach().cpu().numpy()
    min_distance, cluster_num = embedding_distance(series_embedding, cluster_centers)
    return min_distance, cluster_num


def in_concat_region(i, seq_length, instance_length):
    if (i // instance_length + 1) * instance_length - i < seq_length:
        return True
    return False


def generate_profile_value(model, series, seq_length, cluster_centers, cluster_num, instance_length, device):
    profile_value = []
    profile_pattern = []
    pattern_distance_dict = {}
    indice_dict = {}
    for i in range(cluster_num):
        pattern_distance_dict[i] = []
        indice_dict[i] = []
    for i in tqdm(range(len(series) - seq_length)):
        distance, match_pattern = cluster_distance(model, series[i:i + seq_length, :], cluster_centers, device)
        if BORDER_JUDGE and in_concat_region(i, seq_length, instance_length):
            distance = float("inf")
        profile_value.append(distance)
        profile_pattern.append(match_pattern)
        pattern_distance_dict[match_pattern].append(distance)
        indice_dict[match_pattern].append(i)
    for i in range(cluster_num):
        pattern_distance_dict[i] = np.array(pattern_distance_dict[i])
        indice_dict[i] = np.array(indice_dict[i])
    return np.array(profile_value), np.array(profile_pattern), pattern_distance_dict, indice_dict


def generate_classification_profile(model, train_data, seq_length, cluster_centers, cluster_num, device):
    distances_of_all_classes = {}
    indices_of_all_classes = {}
    for series, class_num in tqdm(train_data):
        print(f"class:{class_num}")
        pattern_distance_dict = {}
        indice_dict = {}
        for i in range(cluster_num):
            pattern_distance_dict[i] = []
            indice_dict[i] = []
        for i in tqdm(range(len(series) - seq_length)):
            distance, match_pattern = cluster_distance(model, series[i:i + seq_length, :], cluster_centers, device)
            pattern_distance_dict[match_pattern].append(distance)
            indice_dict[match_pattern].append(i)
        distances_of_all_classes[class_num] = pattern_distance_dict
        indices_of_all_classes[class_num] = indice_dict
    return distances_of_all_classes, indices_of_all_classes


SERIES_NUM = -1
BORDER_JUDGE = False
torch.manual_seed(42)


def main():
    dataset_dir = "dataset"
    data_name = "xxx.pkl"
    data = read_data(dataset_dir, data_name)[:10000]
    dimension = data.shape[1]
    HIDDEN_SIZE = 64
    SEQ_LENGTH = 50
    WINDOW_STRIDE = SEQ_LENGTH // 2
    CLUSTER_NUM = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    EMBEDDING_DIM = 15
    TRAIN = True
    STATE_NAME = f"H{HIDDEN_SIZE}_S{SEQ_LENGTH}_E{EMBEDDING_DIM}_class{SERIES_NUM}"
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    candidates = generate_subsequence_candidates(data, SEQ_LENGTH, WINDOW_STRIDE)
    model = GCN_simple(input_series_length=SEQ_LENGTH, input_series_dim=dimension, hidden_channels=HIDDEN_SIZE,
                       output_dimension=EMBEDDING_DIM)
    model = model.to(device)
    if TRAIN:
        triple = generate_train_triple_by_frequency(candidates, hash_size=5, series_num=SERIES_NUM)
        triple_loss_train(model, triple, learning_rate=0.001, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                          device=device, data_name=data_name, state_name=STATE_NAME)
    else:
        model.load_state_dict(
            torch.load(os.path.join("saved_state", data_name, model.model_type, STATE_NAME + '.pth')))
    distributions = cluster(model, candidates, cluster_num=CLUSTER_NUM, device=device, class_num=SERIES_NUM)
    # generate profile value of each point and its corresponding distribution
    my_profile, profile_pattern, pattern_distance_dict, indice_dict = generate_profile_value(model,
                                                                                             data,
                                                                                             SEQ_LENGTH,
                                                                                             distributions,
                                                                                             CLUSTER_NUM,
                                                                                             10,
                                                                                             device)

if __name__ == '__main__':
    main()
