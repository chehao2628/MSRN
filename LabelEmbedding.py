import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    # Graph Attention Network
    def __init__(self, in_features, out_features, dropout=0, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)  # shape [N, out_features]
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h) + h  # [N,N], [N, out_features] --> [N, out_features]
        return F.elu(h_prime)


class BatchedDiffPool(nn.Module):
    # Diffpool
    def __init__(self, nfeat, nnext, nhid, is_final=False):
        super(BatchedDiffPool, self).__init__()
        self.is_final = is_final
        self.embed = GraphAttentionLayer(nfeat, nhid)
        self.assign_mat = GraphAttentionLayer(nfeat, nnext)
        self.log = {}
        self.entropy_loss = 0

    def forward(self, x, adj, mask=None, log=False):
        z_l = self.embed(x, adj)
        assign_mat = self.assign_mat(x, adj)
        s_l = F.softmax(assign_mat, dim=-1)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        return xnext, anext, s_l


class LabelEmbed(nn.Module):
    # Generate label embeddings
    def __init__(self, input_size):
        super().__init__()
        self.label_dim = 512
        self.input_shape = input_size
        self.layers = nn.ModuleList([
            GraphAttentionLayer(input_size, 300),
            GraphAttentionLayer(300, self.label_dim),
        ])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class GroupEmbed(nn.Module):
    # Generate group embeddings
    def __init__(self, pool_size):
        super().__init__()
        self.label_dim = 512
        self.layers = nn.ModuleList([
            BatchedDiffPool(self.label_dim, pool_size, self.label_dim),
        ])

    def forward(self, x, adj):
        for layer in self.layers:
            if isinstance(layer, GraphAttentionLayer):
                x = layer(x, adj)
            elif isinstance(layer, BatchedDiffPool):
                x, adj, assign_mat = layer(x, adj)
        return x, assign_mat
