#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : create_md_data.py
# Date    : 2022-08-16
from utils import load_jsonl, csv_reader, csv_reader_static, dump_jsonl, clear_format
from typing import List, Optional, Set
import random
from loguru import logger
from knowledge import Knowledge
import json
import re

random.seed(123)


def create_md_dataset(load_data_path: str,
                      dump_data_path: List,
                      ds_name: str = 'nlpcc',
                      ds_type: str = 'train'):
    """
    生成提及识别数据集
    :param load_data_path: 处理好的json数据输入目录
    :param dump_data_path: 输出目录
    :param ds_name: 数据集名字
    :param ds_type: train or dev or test
    :return:
    """
    if ds_name == 'nlpcc':
        dataset = load_jsonl(load_data_path)
        # train + dev
        if len(dump_data_path) == 2:
            train_data_list = []
            dev_data_list = []
            rowid = 0
            random.shuffle(dataset)
            for example in dataset:
                question = example['question']
                mention = example['mention']
                beg = question.find(mention)
                label = list('O' * len(question))
                label[beg] = 'B-entity'
                for i in range(len(mention) - 1):
                    label[beg + i + 1] = 'I-entity'
                # 取10分之一训练数据做验证
                rowid += 1
                if rowid % 10 == 1:
                    dev_data_list.append(' '.join(question) + '\t' + ' '.join(label) + '\n')
                else:
                    train_data_list.append(' '.join(question) + '\t' + ' '.join(label) + '\n')
            with open(dump_data_path[0], 'w') as f_obj:
                for i in train_data_list:
                    f_obj.write(i)
            with open(dump_data_path[1], 'w') as f_obj:
                for i in dev_data_list:
                    f_obj.write(i)
        # test
        elif len(dump_data_path) == 1:
            test_data_list = []
            for example in dataset:
                question = example['question']
                mention = example['mention']
                beg = question.find(mention)
                label = list('O' * len(question))
                label[beg] = 'B-entity'
                for i in range(len(mention) - 1):
                    label[beg + i + 1] = 'I-entity'
                test_data_list.append(' '.join(question) + '\t' + ' '.join(label) + '\n')
            with open(dump_data_path[0], 'w') as f_obj:
                for i in test_data_list:
                    f_obj.write(i)
        else:
            logger.error("检查输入参数")
    elif ds_name == 'ccksel19':
        dataset = load_jsonl(load_data_path)
        random.shuffle(dataset)
        data_list = []
        if ds_type == 'train':
            dataset = dataset[:8000]
        elif ds_type == 'dev':
            dataset = dataset[8000:9000]
        elif ds_type == 'test':
            dataset = dataset[9000:10000]
        else:
            logger.error('参数错误')
        for example in dataset:
            question = example['text']
            print(question)
            mention_data = example['mention_data']
            print(mention_data)
            label = list('O' * len(question))
            for mention_dict in mention_data:
                # mention = json.load(mention)
                beg = int(mention_dict['offset'])
                mention = mention_dict['mention']
                print(beg)
                label[beg] = 'B-entity'
                for i in range(len(mention) - 1):
                    label[beg + i + 1] = 'I-entity'
            assert (len(question) == len(label))
            print(question)
            print(label)
            # 找到空格位置，替换question和label
            blank_pos = [substr.start() for substr in re.finditer(' ', question)]
            print(blank_pos)
            for i in reversed(blank_pos):
                del label[i]
            question = question.replace(' ', '')
            assert (len(question) == len(label))
            print('删除后的question{}和label{}'.format(question,label))
            data_list.append(' '.join(question) + '\t' + ' '.join(label) + '\n')  # 只要把question中的空格都删除才能用空格作为分隔符给到后面使用
        with open(dump_data_path[0], 'w') as f_obj:
            for i in data_list:
                f_obj.write(i)


# def create_md_dataset_w2ner(load_data_path: str, dump_data_path: List):
#     """
#     生成w2ner提及识别数据集 todo
#     :param load_data_path: 处理好的json数据输入目录
#     :param dump_data_path: 输出目录
#     :return:
#     """
#     dataset = load_jsonl(load_data_path)
#     # train + dev
#     if len(dump_data_path) == 2:
#         train_data_list = []
#         dev_data_list = []
#         rowid = 0
#         random.shuffle(dataset)
#         for example in dataset:
#             question = example['question']
#             mention = example['mention']
#             beg = question.find(mention)
#             label = list('O' * len(question))
#             label[beg] = 'B-entity'
#             for i in range(len(mention) - 1):
#                 label[beg + i + 1] = 'I-entity'
#             # 取10分之一训练数据做验证
#             rowid += 1
#             if rowid % 10 == 1:
#                 dev_data_list.append(' '.join(question) + '\t' + ' '.join(label) + '\n')
#             else:
#                 train_data_list.append(' '.join(question) + '\t' + ' '.join(label) + '\n')
#         with open(dump_data_path[0], 'w') as f_obj:
#             for i in train_data_list:
#                 f_obj.write(i)
#         with open(dump_data_path[1], 'w') as f_obj:
#             for i in dev_data_list:
#                 f_obj.write(i)
#     # test
#     elif len(dump_data_path) == 1:
#         test_data_list = []
#         for example in dataset:
#             question = example['question']
#             mention = example['mention']
#             beg = question.find(mention)
#             label = list('O' * len(question))
#             label[beg] = 'B-entity'
#             for i in range(len(mention) - 1):
#                 label[beg + i + 1] = 'I-entity'
#             test_data_list.append(' '.join(question) + '\t' + ' '.join(label) + '\n')
#         with open(dump_data_path[0], 'w') as f_obj:
#             for i in test_data_list:
#                 f_obj.write(i)
#     else:
#         logger.error("检查输入参数")


if __name__ == "__main__":
    # nlpcc
    # 训练集+验证集
    # load_train_data_path = cfg.preprocessing["step1"]["load_train_data_path"]
    # dump_train_data_path = cfg.preprocessing["step1"]["dump_train_data_path"]
    # dump_dev_data_path = cfg.preprocessing["step1"]["dump_dev_data_path"]
    # create_md_dataset(load_train_data_path, [dump_train_data_path, dump_dev_data_path])
    # # 测试集
    # load_test_data_path = cfg.preprocessing["step1"]["load_test_data_path"]
    # dump_test_data_path = cfg.preprocessing["step1"]["dump_test_data_path"]
    # create_md_dataset(load_test_data_path, [dump_test_data_path])

    # # ccksel2019
    # load_data_path = cfg.preprocessing["step1"]["load_train_data_path"]
    # dump_data_path = cfg.preprocessing["step1"]["dump_train_data_path"]
    # create_md_dataset(load_data_path, dump_data_path, ds_name='cckse19', type='dev')

    # a_str = '你 好'
    # print(a_str, len(a_str))
    # a_str = ' '.join(a_str)
    # print(a_str, len(a_str))
    # print(a_str.split(' '))
    # a_str = ''.join(a_str.split(' '))
    # print(a_str, len(a_str))
    # a_str = '你  好'
    # pos = [substr.start() for substr in re.finditer(' ', a_str)]
    # print(pos)
    # a_str = a_str.replace(' ', '')
    # print(a_str)
    pass
