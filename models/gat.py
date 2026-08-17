import dgl
import torch
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        assert num_layers==3
        kwargs = dict(bias=False, allow_zero_in_degree=True, num_heads=1)
        conv_layer = dglnn.GATConv
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        self.convs.append(conv_layer(hidden_channels, hidden_channels, **kwargs))
        self.convs.append(conv_layer(hidden_channels, out_channels, **kwargs))

    def forward(self, blocks, h):
        h = h.float()
        for i, (layer, block) in enumerate(zip(self.convs, blocks)):
            h = layer(block, h)
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=0.5)
            if i == self.num_layers-1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return torch.log_softmax(h, dim=-1)

"""
class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_heads,
                 activation=F.elu,
                 dropout=0.5):
        assert len(n_heads) == n_layers
        assert n_heads[-1] == 1
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_heads = n_heads
        self.layers = nn.ModuleList()
        for i in range(0, n_layers):
            in_dim = in_feats if i == 0 else n_hidden * n_heads[i - 1]
            out_dim = n_classes if i == n_layers - 1 else n_hidden
            self.layers.append(
                dglnn.GATConv(in_dim,
                              out_dim,
                              n_heads[i],
                              allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == self.n_layers - 1:
                h = h.mean(1)
            else:
                h = self.activation(h)
                h = self.dropout(h)
                h = h.flatten(1)
        #return h
        return th.log_softmax(h, dim=-1)
"""
