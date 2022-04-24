from bert_serving.client import BertClient
import numpy as np
import torch
import scipy.sparse as sp
import pandas as pd
# from get_node import ast_node
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



def get_embed(parse_file,max_node):

    val=[]
    df = pd.read_table(parse_file)
    # print(df)
    nodeNum = 0
    for i in range(0, len(df)):
        # print("嵌入的序号：",i)

        s = df.values[i]
        # print("对应的值：", s)
        # exit()
        if s[0].find('->') == -1 and not s[0].startswith('AST'):
            if s[0].find('[') != -1:
                n = s[0].find('[')
                sentence_str = s[0][n+1:len(s[0])]
                if sentence_str == '':
                    sentence_str = 'leaf'
                # print(sentence_str)

                val.append(sentence_str)

    # print(val[510])

    bc = BertClient()
    matrix = bc.encode(val)
    matrix = np.array(matrix)
    matrix = sp.csr_matrix(matrix, dtype=np.float32)

    feature = torch.FloatTensor(np.array(matrix.todense()))

    # print(features)
    print("结点嵌入向量的维度为:",feature.shape)
    if feature.size(0) > max_node:
        features = feature[0:max_node]
    else:
        features = torch.zeros(max_node, 768)
        for k in range(feature.size(0)):
            features[k] = feature[k]
    print("裁剪或补充完之后结点嵌入向量的维度为:", features.shape)
    return features

get_embed("E:/导师任务/AST与ADG融合/code1/datasets/PYB_dataset/PYB_AST/1238.txt",40)