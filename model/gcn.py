import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, feature_dim, output_dimension):
        super(GCN, self).__init__()
        self.model_type = "GCN"
        torch.manual_seed(12345)
        self.conv1 = GCNConv(feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_dimension)

    def forward(self, x, edge_index, batch, layers=1):
        # s = self.dense_diff_lin(x)
        x = x.reshape(-1, x.shape[-1]).float()
        x = self.conv1(x, edge_index)
        # print("before relu")
        x = x.relu()
        # print("layer1:", x)
        if layers == 3:
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN_simple(torch.nn.Module):
    def __init__(self, input_series_length, input_series_dim, hidden_channels, output_dimension):
        super(GCN_simple, self).__init__()
        self.model_type = "GCN"
        self.num_nodes = input_series_dim
        self.gcn = GCN(hidden_channels, input_series_length, output_dimension)

    def forward(self, x, batch_size, device, layers=3):
        edge_index = self.create_edge_index()
        batch = self.current_graph_batch(batch_size)
        edge_index, batch = edge_index.to(device), batch.to(device)
        output = self.gcn(x, edge_index, batch, layers)
        return output

    def create_edge_index(self):
        # src with shape (batch_size, dimension, series_length)
        adjacency_matrix = torch.ones(self.num_nodes, self.num_nodes)
        edge_index, _ = dense_to_sparse(adjacency_matrix)
        return edge_index

    def current_graph_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            for j in range(self.num_nodes):
                batch.append(i)
        return torch.tensor(batch)
