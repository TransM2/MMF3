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

class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.dropout=0.2

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj)) + x
        # return F.log_softmax(x, dim=1)
        return x

class GCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nclass, nfeat)

        # self.dropout=0.2

    def forward(self, x, adj, A2):

        # x = F.dropout(x, self.dropout, training=self.training)

        output1 = F.relu(self.gc1(x, adj))
        # print("输出x的形状为：",x.shape)
        # print("输出output1的形状为：",output1.shape)
        output1 = F.relu(self.gc2(output1, adj))
        output1 = F.relu(self.gc3(output1, adj))
        output2 = F.relu(self.gc1(output1, A2))
        output2 = F.relu(self.gc2(output2, A2))
        output = output2
        # return F.log_softmax(x, dim=1)
        return output

class GCN4(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN4, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nclass, nfeat)

        # self.dropout=0.2

    def forward(self, x, adj, A2):

        # x = F.dropout(x, self.dropout, training=self.training)

        # print("输出x的形状为：",x.shape)
        # print("输出output1的形状为：",output1.shape)
        output1 = F.relu(self.gc3(x, adj))
        output2 = F.relu(self.gc1(output1, A2))
        output2 = F.relu(self.gc2(output2, A2))
        output = output2
        # return F.log_softmax(x, dim=1)
        return output