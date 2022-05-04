import torch.nn as nn
import torch.nn.functional as F
from Model.model_GCN.GCN import GraphConvolution


class GCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout=0.2
        self.ffn = nn.Linear(nfeat, nclass)

    def forward(self, x, adj):
        # x = F.relu(self.gc1(x, adj))
        # x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x, self.dropout, training=self.training)
        outputs = F.relu(self.gc2(x1, adj)) + x
        gcn_output = self.ffn(outputs)
        return gcn_output
