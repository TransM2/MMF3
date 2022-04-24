# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import argparse

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, Sampler, random_split
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Model.model_GCN.GCN_model import  GCN1, GCN2
from Model.model_GCN.batch_embed2 import get_embed
from Model.model_GCN.adj_matrix1 import get_A
from Model.match.ADG_order import get_ADG_order
from Model.match.AST_order import get_AST_order
from Model.sentences_list.make_data import make_nl_vocab
from Model.sentences_list.make_data import make_code_vocab
from Model.sentences_list.make_data import generate_sentences
from Model.sentences_list.make_data import generate_code_datasets
from Model.sentences_list.make_data import generate_nl_datasets
from Model.match.match_JAH_tok_AST import match_Tok_AST2
from GCN_Encoder import GCNEncoder
# from train_eval import train, evaluate
# from metrics import nltk_sentence_bleu,meteor_score
# from rouge import Rouge


Device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Trans_HS.py")
parser.add_argument("-end_epoch", type=int, default=200,
                    help="Epoch to stop training.")
parser.add_argument("-save_model_dir", required=True,
                    help="Path to save the trained model")

parser.add_argument("-save_pred_dir", required=True,
                    help="Path to save prediction result file")

parser.add_argument("-tar_dir", required=True,
                    help="Path to save the target result file")

parser.add_argument("-tra_batch_size", type=int, default=32,
                    help="Batch size set during training")

parser.add_argument("-val_batch_size", type=int, default=32,
                    help="Batch size set during predicting")

parser.add_argument("-tes_batch_size", type=int, default=32,
                    help="Batch size set during predicting")

parser.add_argument("-AST_num", type=int, default=None,
                    help="Number of ASTs required for training")

parser.add_argument("-HS_AST_dir", default=None,
                    help="Path to AST on HS dataset")

opt = parser.parse_args()
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch():
    # input_batch, output_batch, target_batch = [],[],[]
    input_batch1, output_batch1, target_batch1 = [], [], []
    for i in range(len(sentences)):
        input_batch = [[src_vocab[n] for n in sentences[i][0].split()]]
        output_batch = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        target_batch = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        # print('00000')

        input_batch1.extend(input_batch)
        output_batch1.extend(output_batch)
        target_batch1.extend(target_batch)
    # print(output_batch)
    # exit()
    # print(input_batch1)
    # print(output_batch1[0])
    # exit()
    print(len(output_batch1[0]))
    print(len(target_batch1[0]))

    return torch.LongTensor(input_batch1), torch.LongTensor(output_batch1), torch.LongTensor(target_batch1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(MultiScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class GCNScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(GCNScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.linear = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class GCNMultiHeadAttention(nn.Module):
    def __init__(self):
        super(GCNMultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.linear = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = GCNScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


# class PoswiseFeedForwardNet(nn.Module):
#     def __init__(self):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.layer_norm = nn.LayerNorm(d_model)
#
#     def forward(self, inputs):
#         residual = inputs # inputs : [batch_size, len_q, d_model]
#         output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
#         output = self.conv2(output).transpose(1, 2)
#         return self.layer_norm(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(Device)(output + residual)  # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class MultiDecoderLayer(nn.Module):
    def __init__(self):
        super(MultiDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.dec_multi_attn = GCNMultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, multi_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs, dec_ast_attn = self.dec_multi_attn(dec_outputs, multi_outputs, multi_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn, dec_ast_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        src_embed = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns, src_embed

class Multi_model(nn.Module):
    def __init__(self, max_ast_node, src_max_length):
        super(Multi_model, self).__init__()
        # self.Linear = nn.Linear(768, d_model)
        self.conv1 = nn.Conv1d(max_ast_node, src_max_length, 1, stride=1)
        # self.conv2 = nn.Conv1d(8, 4, 1, stride=1)
        self.ffn = nn.Linear(2*d_model, d_model)
        # self.conv1 = nn.Conv1d(in_channels=max_ast_node, out_channels=max_ast_node, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=src_max_length, out_channels=max_ast_node, kernel_size=1)
        # self.pooling = nn.
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # self.pos_emb = PositionalEncoding(d_model)
        # 设定encoder的个数
        self.enc_self_attn = MultiScaledDotProductAttention()
        # self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_outputs, ast_outputs, src_embed, ast_embed, match_rela):

        # print("ast_outputs的形状为：", ast_outputs.shape)
        # print("ast_embed的形状为：", ast_embed.shape)
        ast_outputs1 = self.conv1(ast_outputs)
        ast_embed1 = self.conv1(ast_embed)
        # print("ast_embed1的形状为：", ast_embed1.shape)
        enc_outputs, enc_self_attn = self.enc_self_attn(ast_outputs1, enc_outputs, enc_outputs)

        cat_embed = torch.cat([src_embed, ast_embed1], -1)
        # print("cat_embed的形状为：",cat_embed.shape)
        cat_embed1 = self.ffn(cat_embed)
        # print("cat_embed1的形状为：", cat_embed1.shape)
        # cat_embed = cat_embed.permute(1,0,2)
        # print('gfhjh',cat_embed.shape)
        # cat_embed1 = self.conv2(cat_embed)
        # cat_embed1 = cat_embed1.permute(1, 0, 2)
        # print("cat_embed1的形状为：", cat_embed1.shape)
        # rela = match_rela
        # if ast_embed.shape[0] == 1:
        #     # print("输出测试时rela", rela)
        #     # print("输出测试时rela的维度", np.array(rela).shape)
        #     dec2 = ast_embed
        #     for k, v in rela[0].items():
        #         v = int(v)
        #         if k < 200 and v < max_ast_node:
        #             src_embed[0][k] = src_embed[0][k] + dec2[0][v]
        # # print(rela)
        # # print(rela[0][0])
        # else:
        #     batch_order = 0
        #     # print("输出训练时rela", rela)
        #     # print("输出训练时rela的维度", np.array(rela).shape)
        #     for AST_order in range(0, opt.tra_batch_size):
        #         dec2 = ast_embed[batch_order]
        #         for k, v in rela[AST_order][0].items():
        #             v = int(v)
        #             if k < 200 and v < max_ast_node:
        #                 src_embed[batch_order][k] = src_embed[batch_order][k] + dec2[v]
        #         batch_order = batch_order + 1

        enc_outputs = cat_embed1 + enc_outputs
        return enc_outputs


class MultiDecoder(nn.Module):
    def __init__(self, max_ast_node, src_max_length):
        super(MultiDecoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([MultiDecoderLayer() for _ in range(n_layers)])
        self.multi = Multi_model(max_ast_node, src_max_length)
        # self.AST_embed = AST_Model()

    def forward(self, dec_inputs, enc_inputs, enc_outputs, ast_outputs, src_embed, ast_embed, match_rela):   # 变动
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        multi_model_out = self.multi(enc_outputs, ast_outputs, src_embed, ast_embed, match_rela)   # 变动
        # print(multi_model_out)
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
            Device)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(Device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequent_mask(dec_inputs).to(
            Device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).to(Device)  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]
        # dec_ast_attn_mask = get_attn_pad_mask(dec_inputs, ast_inputs)

        dec_self_attns, dec_enc_attns = [], []
        dec_ast_attns = []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn, dec_ast_attn = layer(dec_outputs,  enc_outputs, multi_model_out, dec_self_attn_mask,  dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
            dec_ast_attns.append(dec_ast_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns, dec_ast_attns


class MyDataSet(Data.Dataset):
    def __init__(self, X, Adj, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.X = X
        self.Adj = Adj
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        # self.match_rela = match_rela


    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Adj[idx], self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

class MySampler(Sampler):
    def __init__(self, dataset, batchsize):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batchsize		# 每一批数据量
        self.indices = range(len(dataset))	 # 生成数据集的索引
        self.count = int(len(dataset) / self.batch_size)  # 一共有多少批

    def __iter__(self):
        for i in range(self.count):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return self.count

class Transformer(nn.Module):
    def __init__(self,max_ast_node, src_max_length):
        super(Transformer, self).__init__()
        self.encoder_tok = Encoder().to(Device)
        self.decoder1 = MultiDecoder(max_ast_node, src_max_length).to(Device)
        # self.decoder = Decoder(tgt_vocab_size).to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(Device)

    def forward(self,  enc_inputs, dec_inputs, ast_outputs, ast_embed, match_rela):    # 变动

        enc_outputs, enc_self_attns, src_embed = self.encoder_tok(enc_inputs)

        dec_outputs1, dec_self_attns, dec_enc_attns, dec_ast_attns = self.decoder1(dec_inputs, enc_inputs, enc_outputs,
                                                                                   ast_outputs, src_embed, ast_embed, match_rela)  # 变动
        dec_logits1 = self.projection(dec_outputs1)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits1.view(-1, dec_logits1.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns, dec_ast_attns


def greedy_decoder(model, enc_input, ast_outputs, ast_embed, match_rela, start_symbol):

    enc_outputs, enc_self_attns, src_embed = model.encoder_tok(enc_input)
    # print("greedy中enc_outputs的维度为：",enc_outputs.shape)
    # print("greedy中ast_outputs的维度为：", ast_outputs.shape)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _, _ = model.decoder1(dec_input, enc_input, enc_outputs, ast_outputs, src_embed, ast_embed, match_rela)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()



if __name__ == '__main__':


    c1,c2=generate_code_datasets()
    n1,n2,n3= generate_nl_datasets()
    sentences=generate_sentences()

    src_vocab=make_code_vocab()
    # print("11111",src_vocab)
    src_vocab_size = len(src_vocab)

    tgt_vocab =make_nl_vocab()
    # print(tgt_vocab)
    # print("22222",tgt_vocab)
    # print(len(sentences))
    # print("自然语言的最大长度为：",n2)
    # print(len(c1)+len(c2))
    # print("代码的最大长度为：",c3)
    number_dict = {i: w for w, i in tgt_vocab.items()}
    # number_dict2 = {i: w for w, i in enumerate(tgt_vocab)}
    # print(number_dict)
    # print(number_dict2)
    # exit()
    tgt_vocab_size = len(tgt_vocab)
    # print(tgt_vocab_size)
    # exit()

    src_len = c2 # length of source
    tgt_len = n3 # length of target
    max_ast_node = 100

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    # if opt.AST_num is not None:

    list_tok_AST = match_Tok_AST2()

    gcn_model = GCNEncoder().to(Device)
    trans_model = Transformer(max_ast_node,src_len).to(Device)
    # print(11111)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=0.0001, momentum=0.99)
    trans_optimizer = optim.SGD(trans_model.parameters(), lr=0.0001, momentum=0.99)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # print(len(sentences))
    enc_inputs, dec_inputs, target_batch = make_batch()


    X = []
    Adj = []
    for i1 in range(opt.AST_num):
        i1_str = str(i1)
        print("当前准备生成AST的序号为：",i1_str)
        file_name = i1_str + 'em.txt'

        features1 = get_embed(opt.HS_AST_dir + file_name,max_ast_node)
        adj1 = get_A(opt.HS_AST_dir + file_name,max_ast_node)
        X.append(features1)
        Adj.append(adj1)

    sp1 = 9000
    sp2 = 10000
    # sp3 = 20
    # list_tok_AST = np.expand_dims(list_tok_AST,axis=1)
    match_rela1 = list_tok_AST[:sp1]
    match_rela2 = list_tok_AST[sp1:sp2]
    match_rela3 = list_tok_AST[sp2:]
    # print("list_tok_AST的输出为：",list_tok_AST)
    # print("list_tok_AST的输出为：", np.array(list_tok_AST).shape)
    # print("match_rela1的输出为:",match_rela1)
    # print("match_rela1的形状为:", np.array(match_rela1).shape)
    # exit()
    X_1 = X[:sp1]
    Adj_1 = Adj[:sp1]
    X_2 = X[sp1:sp2]
    Adj_2 = Adj[sp1:sp2]
    X_3 = X[sp2:]
    Adj_3 = Adj[sp2:]
    enc_inputs1 = enc_inputs[:sp1]
    dec_inputs1 = dec_inputs[:sp1]
    target_batch1 = target_batch[:sp1]
    enc_inputs2 = enc_inputs[sp1:sp2]
    dec_inputs2 = dec_inputs[sp1:sp2]
    target_batch2 = target_batch[sp1:sp2]
    enc_inputs3 = enc_inputs[sp2:]
    dec_inputs3 = dec_inputs[sp2:]
    target_batch3 = target_batch[sp2:]

    # 对数据进行分批处理
    train_data = MyDataSet(X_1, Adj_1, enc_inputs1, dec_inputs1, target_batch1)
    valid_data = MyDataSet(X_2, Adj_2, enc_inputs2, dec_inputs2, target_batch2)
    test_data = MyDataSet(X_3, Adj_3,enc_inputs3, dec_inputs3, target_batch3)

    my_sampler1 = MySampler(train_data, opt.tra_batch_size)
    my_sampler2 = MySampler(valid_data, opt.val_batch_size)
    my_sampler3 = MySampler(test_data, opt.tes_batch_size)
    train_data_loader = Data.DataLoader(train_data, batch_sampler=my_sampler1)
    valid_data_loader = Data.DataLoader(valid_data, batch_sampler=my_sampler2)
    test_data_loader = Data.DataLoader(test_data, batch_sampler=my_sampler3)


    # gcn_model = GCNEncoder().to(Device)
    # gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=0.0001, momentum=0.99)

    best_test_loss = float("inf")

    #模型训练
    for epoch in range(opt.end_epoch):
        epoch_loss = 0
        gcn_model.train()
        trans_model.train()
        match_tra_order = 0
        for x, adj, enc_inputs, dec_inputs, dec_outputs in train_data_loader:
            x, adj = x.to(Device), adj.to(Device)
            ast_outputs, ast_embed = gcn_model(x, adj)

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(Device), dec_inputs.to(Device), dec_outputs.to(Device)
            # match_rela1 = match_rela1
            match_rela_1 = match_rela1[match_tra_order:match_tra_order + opt.tra_batch_size]
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns, dec_ast_attns = trans_model(enc_inputs, dec_inputs, ast_outputs, ast_embed, match_rela_1)  # 变动
            loss = criterion(outputs, dec_outputs.view(-1))

            trans_optimizer.zero_grad()
            gcn_optimizer.zero_grad()
            loss.backward()
            trans_optimizer.step()
            gcn_optimizer.step()

            epoch_loss += loss.item()
            # match_tra_order = match_tra_order + opt.tra_batch_size
        train_avg_loss = epoch_loss / len(train_data_loader)
        print('\ttrain loss: ', '{:.4f}'.format(train_avg_loss))

        #模型验证
        epoch_loss = 0
        gcn_model.eval()
        trans_model.eval()
        # trans2_model.eval()

        match_val_order = 0
        with torch.no_grad():
            for x, adj, enc_inputs, dec_inputs, dec_outputs in valid_data_loader:
                x, adj = x.to(Device), adj.to(Device)
                ast_outputs, ast_embed = gcn_model(x, adj)
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(Device), dec_inputs.to(Device), dec_outputs.to(
                    Device)
                # match_rela2 = match_rela2
                match_rela_2 = match_rela2[match_val_order : match_val_order + opt.val_batch_size]

                outputs1, enc_self_attns, dec_self_attns, dec_enc_attns, dec_ast_attns = trans_model(enc_inputs, dec_inputs, ast_outputs, ast_embed, match_rela_2)  # 变动
                loss1 = criterion(outputs1, dec_outputs.view(-1))
                # loss2 = criterion(outputs2, dec_outputs.view(-1))
                # loss = 0.8*loss1 + 0.2*loss2
                loss = loss1

                epoch_loss += loss.item()
                # match_val_order = match_val_order + opt.val_batch_size
        valid_avg_loss = epoch_loss / len(valid_data_loader)
        perplexity = math.exp(valid_avg_loss)
        perplexity = torch.tensor(perplexity).item()
        print('\t eval_loss: ', '{:.4f}'.format(valid_avg_loss))
        print('\tperplexity: ', '{:.4f}'.format(perplexity))
        if valid_avg_loss < best_test_loss:
            best_test_loss = valid_avg_loss
            torch.save(gcn_model.state_dict(), 'save_model/gcn_model.pt')
            torch.save(trans_model.state_dict(), 'save_model/trans_loss1.pt')


    # Test
    # enc_inputs, dec_inputs, target_batch = make_batch()
    # print(enc_inputs)
    # print(enc_inputs.shape)
    # print(enc_inputs[0].shape)
    # print(torch.unsqueeze(enc_inputs[0],0).shape)

    with open(opt.save_pred_dir+'.txt', 'w', encoding='utf-8') as f1:
        f1.write('')
    # with open(opt.save_tar_dir+'.txt', 'w', encoding='utf-8') as f2:
    #     f2.write('')
    print("sentence的长度：", len(sentences))

    gcn_model.load_state_dict(torch.load('save_model/gcn_model.pt'))
    trans_model.load_state_dict(torch.load('save_model/trans_loss1.pt'))
    # trans2_model.load_state_dict(torch.load('save_model/multi_loss2.pt'))
    gcn_model.eval()
    trans_model.eval()

    with open(opt.save_pred_dir+'.txt', 'w', encoding='utf-8') as f1:
        f1.write('')
    with open(opt.tar_dir+'.txt', 'w', encoding='utf-8') as f2:
        f2.write('')

    # x, adj, inputs, _, _ = next(iter(test_data_loader))
    # q = []
    match_tes_order = 0
    tar_order = sp2
    for x, adj, inputs, _, _ in test_data_loader:
        print("测试时x的维度为：",x.shape)
        print("测试时inputs的维度为：",inputs.shape)
        # match_rela3 = match_rela3
        match_rela_3 = match_rela3[match_tes_order : match_tes_order+opt.tes_batch_size]
        for j in range(len(inputs)):
            x, adj = x.to(Device), adj.to(Device)
            ast_outputs, ast_embed = gcn_model(x[j].unsqueeze(0), adj[j].unsqueeze(0))  # 变动
            # print(ast_outputs.shape)
            # exit()
            greedy_dec_input = greedy_decoder(trans_model, inputs[j].view(1, -1).to(Device), ast_outputs,
                                              ast_embed, match_rela_3[j], start_symbol=tgt_vocab['SOS'])  # 变动
            pred, _, _, _, _  = trans_model(inputs[j].view(1, -1).to(Device), greedy_dec_input,
                                          ast_outputs, ast_embed, match_rela_3[j])  # 变动
            pred = pred.data.max(1, keepdim=True)[1]

            print(sentences[tar_order][0], '->', [number_dict[n.item()] for n in pred.squeeze()])
            y1 = [number_dict[n.item()] for n in pred.squeeze()]
            # q.append(x1)

            # print(q)
            str1 = " ".join(y1)
            str2 = sentences[tar_order][2]
            with open(opt.save_pred_dir + '.txt', 'a', encoding='utf-8') as f1:
                f1.write(str1)
                f1.write('\n')
            with open(opt.tar_dir + '.txt', 'a', encoding='utf-8') as f2:
                f2.write(str2)
                f2.write('\n')
            tar_order = tar_order + 1
        match_tes_order = match_tes_order + opt.tes_batch_size
    # pred1 = []
    # for k in q:
    #     s = " ".join(k)
    #     pred1.append(s)
    # # print(pred1)
    # with open(opt.save_pred_dir+'.txt', 'w', encoding='utf-8') as ff:
    #     for z in pred1:
    #         ff.writelines(z + '\n')
    # ref = []
    # with open(opt.tar_dir, 'r', encoding='utf-8') as f:  # ref1
    #     lines = f.readlines()
    #
    #     for line in lines:
    #         line = line.strip('\n')
    #         # print(line)
    #         ref.append(line)
    # # print(ref)
    # avg_score = nltk_sentence_bleu(pred1, ref)
    # meteor = meteor_score(pred1, ref)
    # print('S_BLEU: %.4f' % avg_score)
    # # print('C-BLEU: %.4f' % corup_BLEU)
    # print('METEOR: %.4f' % meteor)
    # rouge = Rouge()
    # rough_score = rouge.get_scores(pred1, ref, avg=True)
    # print(' ROUGE: ', rough_score)