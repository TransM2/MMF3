import json
import scipy.sparse as sp
import networkx as nx
import numpy as np
from scipy import sparse
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def get_A(parse_file,max_node):
    pre_vex = []
    next_vex = []
    edgList = []
    vexNum = 0
    with open(parse_file, "r") as f:
        ID = 0
        data = f.readline()
        while data:
            if data.find("->") != -1:

                # matchObj = re.match(r'(\"*)(\d+)(\"*)', data)
                # if matchObj:
                #     edgDict[op][matchObj.group()] = ID
                #     ID += 1
                #     vexNum += 1
            # else:
                # if op == 0:
                #     sData = data.strip().split('->')
                #     edgList[op].append(sData)
                # else:
                #     edgeInfo = data.strip().split(':')[0]
                #     sData = edgeInfo.split('->')
                #     edgList[op].append(sData)
                edgeInfo = data.strip().split(':')[0]
                # print(edgeInfo)
                sData = edgeInfo.split('->')
                edgList.append((sData[0],sData[1]))
            data = f.readline()
        # 生成图
        # print(edgelist)
        G = nx.Graph()
        G.add_edges_from(edgList)

        # nx.draw(G, with_labels=True)
        A = np.array(nx.adjacency_matrix(G).todense())
        A1 = A + sp.eye(A.shape[0])
        A = np.array(A1, dtype=int)
        # print(A)
        print("初始邻接矩阵的维度为:",A.shape)
        if len(A[0]) > max_node:
            a = A[0:max_node, 0:max_node]
            # print(aa)
        else:
            a = np.zeros((max_node, max_node), dtype=int)
            # A = A + a
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    a[i][j] = A[i][j]
        print("裁剪或补充完之后邻接矩阵的维度为:", a.shape)
        a = np.array(a, dtype=float)
        adj = normalize(a)
        # print(adj)
        adj = sp.csr_matrix(adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        print("裁剪或补充完之后并转换的邻接矩阵的维度为:", adj.shape)
        # print(type(A))
        return adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
# get_A("ADG5.26.txt")
# get_A("E:/导师任务/AST与ADG融合/code1/datasets/HS_dataset/HS_AST/0.txt",185)