import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=2,
#                     help='Number of the batch.')
# args = parser.parse_args()
batch_size = 32
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(batch_size,in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        outputs = torch.zeros_like(input)
        # print("outputs的维度为：", outputs.shape)
        for i in range(input.size(0)):
            support = torch.mm(input[i], self.weight[i])
            # spmm的意思是考虑加不加偏置
            # print("input的维度为：",input.shape)
            # print("input[i]的维度为：", input.shape)
            # print("support的类型为:",support.type())
            # support = support.int()
            # print("adj[i]的类型为:",adj[i].type)
            # print("adj的维度为:", adj.shape)
            # print("adj[i]的维度为:", adj[i].shape)
            output = torch.mm(adj[i], support)
            if self.bias is not None:
                output + self.bias
            outputs[i] = output
        # print(outputs)
        return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
