import dgl
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=F.relu,
                 dropout=0):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x.float()
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return th.log_softmax(h, dim=-1)
