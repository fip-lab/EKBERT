#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : infuse_entity_token_and_subgraph.py
# Date    : 2022-08-21
from typing import List, Dict
from utils import csv_reader_static, clear_format, load_jsonl
from collections import OrderedDict
import json
import random
from loguru import logger

random.seed(123)

"""
对实体消歧数据集进行信息融合，在实体、提及前后加入特殊token,在实体后加子图上下文
"""


def infuse_mention_token(question: str, mention: str) -> str:
    """
    给问题的提及前后加入特殊token
    :return:
    """
    # question_infused = ""
    mention_pos_beg = question.find(mention)
    mention_pos_end = mention_pos_beg + len(mention)
    question_infused = question[:mention_pos_beg] + '<e1>' + mention + '</e1>' + question[mention_pos_end:]
    return question_infused


def infuse_entity_token_and_subgraph(entity: str, entity_subgraph_dict: Dict) -> str:
    """
    给实体加入特殊token和子图信息
    :param entity:
    :param entity_subgraph_dict:
    :return:
    """
    # infused_candidate = ""
    subgraph = entity_subgraph_dict[entity]
    infused_candidate = '<e2>' + entity + '</e2>' + subgraph
    return infused_candidate


def prepare_entity_subgraph_dict(filter_spo_path: str) -> Dict:
    """
    返回实体和对应子图的映射
    :param filter_spo_path:
    :return:
    """
    entity_subgraph_dict = OrderedDict()
    filter_kb = csv_reader_static(filter_spo_path)
    for i in filter_kb:
        s, p, o = i.split('\t')
        s = clear_format(s)
        p = clear_format(p)
        o = clear_format(o)
        if s in entity_subgraph_dict.keys():
            desc.append(p + ':' + o)
            entity_subgraph_dict[s] = desc
        else:
            desc = [p + ':' + o]
            entity_subgraph_dict[s] = desc
    for k, v in entity_subgraph_dict.items():
        entity_subgraph_dict[k] = ','.join(v)
    return entity_subgraph_dict


# entity_subgraph_dict = prepare_entity_subgraph_dict("../data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_test.spo")
# print(entity_subgraph_dict['计算机应用基础(2011年中国铁道出版社出版图书)'], len(entity_subgraph_dict))
# print('infused_entity',infuse_entity_token_and_subgraph('计算机应用基础(2011年中国铁道出版社出版图书)',entity_subgraph_dict))

def create_ed_dataset_and_infuse_knowledge(load_data_path: str,
                                           dump_data_path: List,
                                           filter_spo_path: str):
    """
    生成提及识别数据集
    :param load_data_path: 处理好的json数据输入目录
    :param dump_data_path: 输出路径
    :param filter_spo_path: spo路径
    :return:
    """
    dataset = load_jsonl(load_data_path)
    entity_subgraph_dict = prepare_entity_subgraph_dict(filter_spo_path)
    data_list = []
    example_acc = 0
    positive_acc = 0
    if len(dump_data_path) == 2:
        train_data_list = []
        dev_data_list = []
        rowid = 0
        random.shuffle(dataset)  # 打乱数据
        for example in dataset:
            example_acc += 1  # 累计
            question = example['question']
            gold_entity = example['gold_entity']
            example_id = example['example_id']
            mention = example['mention']
            question_infused = infuse_mention_token(question, mention)
            candidates_dict = json.loads(example['candidates'])
            rowid += 1
            if rowid % 10 == 1:  # 每10条抽1条作为验证集
                for c in candidates_dict.keys():
                    if c == gold_entity:
                        label = 1
                        positive_acc += 1
                    else:
                        label = 0
                    candidate_infused = infuse_entity_token_and_subgraph(c, entity_subgraph_dict)
                    candidate_id = str(candidates_dict[c])
                    # dev_data_list.append(example_id + '_' + candidate_id + '\t' + question_infused + '\t' + candidate_infused + '\t' + str(label) + '\n')
                    dev_data_list.append(question_infused + '\t' + candidate_infused + '\t' + str(label) + '\n')
            else:
                for c in candidates_dict.keys():
                    if c == gold_entity:
                        label = 1
                        positive_acc += 1
                    else:
                        label = 0
                    candidate_infused = infuse_entity_token_and_subgraph(c, entity_subgraph_dict)
                    candidate_id = str(candidates_dict[c])
                    # train_data_list.append(example_id + '_' + candidate_id + '\t' + question_infused + '\t' + candidate_infused + '\t' + str(label) + '\n')
                    train_data_list.append(question_infused + '\t' + candidate_infused + '\t' + str(label) + '\n')
        if example_acc == positive_acc:
            with open(dump_data_path[0], 'w') as f_obj:
                for i in train_data_list:
                    f_obj.write(i)
            with open(dump_data_path[1], 'w') as f_obj:
                for i in dev_data_list:
                    f_obj.write(i)
        else:
            logger.error('检查json数据')
    elif len(dump_data_path) == 1:
        test_data_list = []
        for example in dataset:
            example_acc += 1
            question = example['question']
            gold_entity = example['gold_entity']
            example_id = example['example_id']
            mention = example['mention']
            question_infused = infuse_mention_token(question, mention)
            candidates_dict = json.loads(example['candidates'])
            for c in candidates_dict.keys():
                if c == gold_entity:
                    label = 1
                    positive_acc += 1
                else:
                    label = 0
                candidate_infused = infuse_entity_token_and_subgraph(c, entity_subgraph_dict)
                candidate_id = str(candidates_dict[c])
                # test_data_list.append(example_id + '_' + candidate_id + '\t' + question_infused + '\t' + candidate_infused + '\t' + str(label) + '\n')
                test_data_list.append(question_infused + '\t' + candidate_infused + '\t' + str(label) + '\n')
        if example_acc == positive_acc:
            with open(dump_data_path[0], 'w') as f_obj:
                for i in test_data_list:
                    f_obj.write(i)
        else:
            logger.error('检查json数据')
    else:
        logger.error("检查输入参数,第二个参数为输出路径列表（长度为1或者2）")


if __name__ == "__main__":
    # train + dev
    load_data_path = "../data/processed/nlpcc_kbqa/train.json"
    dump_train_data_path = "../data/processed/nlpcc_kbqa/ed_ekbert/train.tsv"
    dump_dev_data_path = "../data/processed/nlpcc_kbqa/ed_ekbert/dev.tsv"
    filter_spo_path = "../data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_train.spo"
    create_ed_dataset_and_infuse_knowledge(load_data_path, [dump_train_data_path, dump_dev_data_path], filter_spo_path)
    # test
    load_data_path = "../data/processed/nlpcc_kbqa/test.json"
    dump_data_path = "../data/processed/nlpcc_kbqa/ed_ekbert/test.tsv"
    filter_spo_path = "../data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_test.spo"
    create_ed_dataset_and_infuse_knowledge(load_data_path, [dump_data_path], filter_spo_path)
