from __future__ import unicode_literals, print_function, division
from collections import Counter
from io import open
import nltk
# nltk.download('punkt')
import os
import math
import torch
import json
import torch.utils.data as Data
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SOS = 2
# EOS = 1
# pad = 0
# nl_max_len = 18
# seq_max_len = 35
def make_code_vocab(code_dir):
    w2i_code = {"pad": 0, "SOS": 1, "EOS": 2, "(":3, ")":4 , ".":5, ",":6}
    i = 7
    with open(code_dir, 'r', encoding='UTF-8') as f:

        lines = f.readlines()
        for line in lines:
            line = line.split('\n')
            line1 = line[0]
            line_tok = line1.split()
            for w in line_tok:
                if w not in w2i_code:
                    w2i_code[w] = i
                    i = i + 1
    # print(w2i_code)
    return w2i_code


def generate_code_datasets(code_dir):
    code = []
    code1_list=[]
    code2_list=[]
    code_maxlen = 200
    less_num = 0

    with open(code_dir, 'r', encoding='UTF-8') as f:
        # lines = f.readline()
        # print(lines[:10].split())
        #
        # lines_2 = lines.split('\n')
        #print(lines_2)
        #print(len(lines_2))
        lines = f.readlines()
        # print(len(lines))
        i = 1
        for line in lines:
            i=i+1
            line1 = line.split('\n')
            line2 = line1[0]
            code.append(line2.split())

            # exit()
    # print(len(code))
    # exit()
    # for code_line in code:
    #     # print(len(nl_line))
    #     # print(code_line)
    #     if code_maxlen > len(code_line):
    #         less_num = less_num + 1
    #
    # ratio = float(less_num / len(code))
    # print(len(code))
    # print("代码长度小于规定的长度的百分比：",ratio)
    # exit()
    for code_line in code:
        str1 = ""
        str2 = ""
        str3 = ""
        str4 = ""
        list1 = []
        list2 = []

        if code_maxlen > len(code_line):

            code1_line = code_line
            code2_line= code_line
            str1 = ' '.join(code1_line)
            str2 = ' '.join(code2_line)
            list1.append(str1)
            list2.append(str2)
            t = code_maxlen - len(code_line)
            # t = t-1
            for i in range(t):
                list1.append('pad')
                list2.append('pad')
            # list2 = list1
            # list1.insert(0,'SOS')
            # list2.append('EOS')
        else:
            code1_line = code_line[0:code_maxlen]
            code2_line = code_line[0:code_maxlen]
            str1 = ' '.join(code1_line)
            str2 = ' '.join(code2_line)
            list1.append(str1)
            list2.append(str2)
            # list1.insert(0, 'SOS')
            # list2.append('EOS')

        str3 = ' '.join(list1)
        str4 = ' '.join(list2)
        code1_list.append(str3)
        code2_list.append(str4)


    # print(code1_list)
    # print(code2_list)

    return code1_list,code_maxlen




def make_nl_vocab(nl_dir):
    w2i_nl = {"pad": 0, "SOS": 1, "EOS": 2, ".": 3}
    i = 4
    with open(nl_dir, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line= line.split('\n')
            line1 = line[0]
            line_tok = line1.split()
            for w in line_tok:
                if w not in w2i_nl:
                    w2i_nl[w] = i
                    i = i+1
    # print(w2i_nl)
    return w2i_nl



def generate_nl_datasets(nl_dir):
    nl_sentence = []
    nl_sentence2 = []
    nl=[]
    src_maxlen=35
    less_num=0

    with open(nl_dir, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            # nl.append(nltk.word_tokenize(line.lower()))
            line1 = line.split('\n')
            line2 = line1[0]
            nl.append(line2.split())
            # print(nl)
            # exit()
    #
    # for nl_line in nl:
    #     # print(len(nl_line))
    #     # print(nl_line)
    #     if src_maxlen > len(nl_line):
    #         less_num=less_num+1
    #
    # ratio = float(less_num / len(nl))
    # print(len(nl))
    #
    # print("自然语言描述长度小于规定的长度的百分比：",ratio)
    # exit()
    for nl_line in nl:
        str1=""
        # str2=[]
        l1 = src_maxlen-1
        if l1 > len(nl_line):
            t = l1-len(nl_line)
            for i in range(t):
                str1 = ' '.join([str1,'pad'])
            str = ' '.join(nl_line)
            str2 = 'SOS '+ str
            str3 = str2 + str1
            str4 = ' '.join([str,'EOS'])
            for j in range(t):
                str4 = ' '.join([str4,'pad'])
        else:
            list1 = nl_line[0:l1]
            # print("列表的长度为:",len(list1))
            str2 = ' '.join(list1)
            str3 = 'SOS '+str2
            str4 = ' '.join([str2,'EOS'])
        # print(str3)
        # print(len(str3.split()))

        # print(str4)
        # print(len(str4.split()))

        nl_sentence.append(str3)
        nl_sentence2.append(str4)
    # for j in range(2480, 2960):
    #     str2 = nl_sentence2[j]
    #     print(str2)
    #     with open('E:/导师任务/AST与ADG融合/code1/datasets/JAH_dataset/JAHoutput_tar.txt', 'a', encoding='utf-8') as f2:
    #         f2.write(str2)
    #         f2.write('\n')
    # exit()
    # print(nl_sentence)
    # print(nl_sentence2)
    # print(src_len)
    # print(nl_sentence)
    # print(type(nl_sentence))
    return nl_sentence,nl_sentence2,src_maxlen





def generate_sentences(code_dir,nl_dir):
    sentences = []
    c1, c2= generate_code_datasets(code_dir)
    n1,n2,n3=generate_nl_datasets(nl_dir)
    for i in range(len(c1)):
        sentences.append([])
        sentences[i].append(c1[i])
        sentences[i].append(n1[i])
        sentences[i].append(n2[i])
    # print(sentences)
    # print(sentences[0][0].split())
    # print(sentences[1][1])
    # print(len(sentences[1][1].split()))
    return sentences

# make_code_vocab()
# generate_code_datasets()
# generate_nl_datasets()
# generate_sentences()
# make_nl_vocab()
# make_code_vocab()
# print(n2[0])

# length=len(n2)-1
# print(length)
# print(c2)
# print(len(n2))
# print(len(c2))
#AST.txt文件中有多个nl对应的AST，想办法把它分开

# print(sentences[0])
# print(sentences)
# print(sentences)
# readast()
# generate_nl_datasets()