import json
import codecs
import numpy as np
import pandas as pd
from keras_bert import Tokenizer
import random
import os

random.seed(123)
file_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))  # 获取当前文件上三级的目录 pen add
data_path = os.path.join(file_path, 'data/raw/ccks_el_2019')


def csv_writer(data_list, path):
    with open(path, 'w') as writer:
        for i in data_list:
            writer.write(i)


def load_jsonl(path: str):
    data_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            data_list.append(json.loads(line))
    return data_list


def get_bio_dict():
    '''
    采用BIO标记形式，对于bert的CLS和SEP位置用TAG标记
    :return:
    '''
    bio_dict = {}
    bio_dict['O'] = 0
    bio_dict['B'] = 1
    bio_dict['I'] = 2
    bio_dict['TAG'] = 3
    return bio_dict


def get_bio(mention):
    '''

    :param mention: mention
    :return: mention的对应标记索引
    '''
    bio_d = get_bio_dict()
    bio = []
    for i in range(len(mention)):
        if i == 0:
            bio.append(bio_d['B'])
        else:
            bio.append(bio_d['I'])
    return bio


def get_bio_list(text, mention_data):
    '''

    :param text: 一段文本
    :param mention_data: 文本中mention列表
    :return: 标记列表
    '''
    bio = [0] * len(text)
    for mention in mention_data:
        men = mention['mention']
        offset = int(mention['offset'])
        bio[offset:offset + len(men)] = get_bio(men)
    return bio


def seq_padding(seq, max_len, value=0):
    '''

    :param seq: 序列列表
    :param max_len:最大长度
    :param value: 填充的值
    :return:
    '''
    x = [value] * max_len
    x[:len(seq)] = seq[:max_len]
    return x


def get_token_dict():
    '''
    bert vocab
    :return:
    '''
    bert_path = 'bert_model/'
    dict_path = bert_path + 'vocab.txt'
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def get_ids_seg(text, tokenizer, max_len=52):
    '''
    得到bert的输入
    :param text:
    :param tokenizer:
    :param max_len:
    :return:
    '''
    indices, segments = [], []
    for ch in text[:max_len - 2]:
        indice, segment = tokenizer.encode(first=ch)
        if len(indice) != 3:
            indices += [100]
            segments += [0]
        else:
            indices += indice[1:-1]
            segments += segment[1:-1]
    segments = [0] + segments + [0]
    indices = [101] + indices + [102]
    return indices, segments


def get_input_bert(input_file, out_file, max_len=52):
    '''
    序列标注的训练数据构建
    :param input_file: 训练文件
    :param out_file: 输出文件
    :param max_len: 最大长度
    :return:
    '''
    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    bio_dict = get_bio_dict()
    inputs = {'ids': [], 'seg': [], 'labels': []}
    dataset = load_jsonl(input_file)
    random.seed(123)
    random.shuffle(dataset)
    dataset = dataset[:900]
    review_samples = []
    # with open(input_file, 'r') as f:
    for temDict in dataset:
        # temDict = json.loads(line)
        text = temDict['text']
        review_samples.append(text + '\n')
        print(text)
        mention_data = temDict['mention_data']
        label = get_bio_list(text, mention_data)
        label = [bio_dict['TAG']] + label + [bio_dict['TAG']]
        indices, segments = get_ids_seg(text, tokenizer)
        assert len(indices) == len(label)
        inputs['ids'].append(seq_padding(indices, max_len))
        inputs['seg'].append(seq_padding(segments, max_len))
        inputs['labels'].append(seq_padding(label, max_len))

    for key in inputs:
        inputs[key] = np.array(inputs[key])
        print(key, inputs[key].shape)
        print(inputs[key][0])
    pd.to_pickle(inputs, out_file)
    csv_writer(review_samples, 'data/train_samples.csv')


def get_input_bert_test(input_file, out_file, max_len=52):
    '''
    测试数据
    :param input_file:
    :param out_file:
    :param max_len:
    :return:
    '''
    token_dict = get_token_dict()
    tokenizer = Tokenizer(token_dict)
    inputs = {'ids': [], 'seg': [], 'labels': [0]}
    dataset = load_jsonl(input_file)
    random.seed(123)
    random.shuffle(dataset)
    test_dataset = dataset[900:1000]
    review_samples = []
    # with open(input_file, 'r') as f:
    for temDict in test_dataset:
        # temDict = json.loads(line)
        text = temDict['text']
        print(text)
        review_samples.append(text + '\n')
        indices, segments = get_ids_seg(text, tokenizer)
        inputs['ids'].append(seq_padding(indices, max_len))
        inputs['seg'].append(seq_padding(segments, max_len))
    for key in inputs:
        inputs[key] = np.array(inputs[key])
        print(key, inputs[key].shape)
        print(inputs[key][0])
    pd.to_pickle(inputs, out_file)
    csv_writer(review_samples, 'data/test_samples.csv')


if __name__ == '__main__':
    # 训练集
    get_input_bert(os.path.join(data_path, 'train.json'), 'data/input_train_ner.pkl')
    train_data = pd.read_pickle('data/input_train_ner.pkl')
    print(train_data)
    # 测试集
    get_input_bert_test(os.path.join(data_path, 'train.json'), 'data/input_eval_ner.pkl')
    test_data = pd.read_pickle('data/input_eval_ner.pkl')  # 测试集
    print(test_data)

    pass
