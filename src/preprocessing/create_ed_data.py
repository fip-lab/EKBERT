#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : create_ed_data.py
# Date    : 2022-08-16
from utils import load_jsonl
import json
from typing import List
import random
from loguru import logger

random.seed(123)


def create_ed_dataset(load_data_path: str,
                      dump_data_path: List,
                      ds_name: str = 'nlpcc',
                      ds_type: str = 'train',
                      mention2id_path: str = ''):
    """
    生成ed数据集
    :param load_data_path: 处理好的json数据输入目录
    :param dump_data_path: 输出目录
    :param ds_name: 数据集名字
    :param ds_type: train or dev or test
    :param mention2id_path: mention2id输入目录
    :return:
    """
    if ds_name == 'nlpcc':
        dataset = load_jsonl(load_data_path)
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
                candidates_dict = json.loads(example['candidates'])
                rowid += 1
                if rowid % 10 == 1:  # 每10条抽1条作为验证集
                    for c in candidates_dict.keys():
                        if c == gold_entity:
                            label = 1
                            positive_acc += 1
                        else:
                            label = 0
                        dev_data_list.append(question + '\t' + c + '\t' + str(label) + '\n')
                else:
                    for c in candidates_dict.keys():
                        if c == gold_entity:
                            label = 1
                            positive_acc += 1
                        else:
                            label = 0
                        train_data_list.append(question + '\t' + c + '\t' + str(label) + '\n')
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
                candidates_dict = json.loads(example['candidates'])
                for c in candidates_dict.keys():
                    if c == gold_entity:
                        label = 1
                        positive_acc += 1
                    else:
                        label = 0
                    test_data_list.append(question + '\t' + c + '\t' + str(label) + '\n')
            if example_acc == positive_acc:
                with open(dump_data_path[0], 'w') as f_obj:
                    for i in test_data_list:
                        f_obj.write(i)
            else:
                logger.error('检查json数据')
        else:
            logger.error("检查输入参数,第二个参数为输出路径列表（长度为1或者2）")
    elif ds_name == 'ccksel19':
        if mention2id_path == '':
            logger.error('请指定mention2id_path参数')
        mention2id = {i['mention']: i['candidates'] for i in load_jsonl(mention2id_path)}
        dataset = load_jsonl(load_data_path)
        ed_data = []
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
        for eid, example in enumerate(dataset):
            # if eid > 100:
            #     break
            text_id = example['text_id']
            question = example['text']
            print(question)
            mention_data = example['mention_data']
            print(mention_data)
            for mid, mention_dict in enumerate(mention_data):
                mention = mention_dict['mention']
                gold_entity_id = mention_dict['kb_id']
                # line_id = str(text_id) + '_' + str(mid)
                # 找到mention的候选实体
                if mention in mention2id and gold_entity_id != "NIL":
                    candidates_dict = mention2id[mention]
                    for cid, (k, v) in enumerate(candidates_dict.items()):
                        line_id = str(text_id) + '_' + str(mid) + '_' + str(k)  # 案例id + 提及序号 + 知识库实体id
                        if k == gold_entity_id:
                            label = 1
                        else:
                            label = 0
                        ed_data.append(line_id + '\t' + question + '\t' + v + '\t' + str(label) + '\n')
                elif gold_entity_id == "NIL":
                    line_id = str(text_id) + '_' + str(0) + '_' + str(0)  # 案例id + 提及序号 + 知识库实体id
                    ed_data.append(line_id + '\t' + question + '\t' + "NIL" + '\t' + str(1) + '\n')
                else:
                    logger.info('案例中【{}】的提及【{}】在mention2id找不到，请检查。'.format(question, mention))
        with open(dump_data_path[0], 'w') as f_obj:
            for i in ed_data:
                f_obj.write(i)


def create_ed_dataset_kbert(load_data_path: str,
                            dump_data_path: List,
                            ds_name: str = 'nlpcc',
                            ds_type: str = 'train',
                            mention2id_path: str = ''):
    """
    生成ed数据集
    :param load_data_path: 处理好的json数据输入目录
    :param dump_data_path: 输出目录
    :param ds_name: 数据集名字
    :param ds_type: train or dev or test
    :param mention2id_path: mention2id输入目录
    :return:
    """
    if ds_name == 'nlpcc':
        dataset = load_jsonl(load_data_path)
        example_acc = 0
        positive_acc = 0
        if len(dump_data_path) == 2:
            train_data_list = ['text_a' + '\t' + 'text_b' + '\t' + 'label' + '\n']
            dev_data_list = ['text_a' + '\t' + 'text_b' + '\t' + 'label' + '\n']
            rowid = 0
            random.shuffle(dataset)  # 打乱数据
            for example in dataset:
                example_acc += 1  # 累计
                question = example['question']
                qid = example['example_id']
                gold_entity = example['gold_entity']
                candidates_dict = json.loads(example['candidates'])
                rowid += 1
                if rowid % 10 == 1:  # 每10条抽1条作为验证集
                    for c in candidates_dict.keys():
                        if c == gold_entity:
                            label = 1
                            positive_acc += 1
                        else:
                            label = 0
                        dev_data_list.append(question + '\t' + c + '\t' + str(label) + '\n')
                else:
                    for c in candidates_dict.keys():
                        if c == gold_entity:
                            label = 1
                            positive_acc += 1
                        else:
                            label = 0
                        train_data_list.append(question + '\t' + c + '\t' + str(label) + '\n')
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
            test_data_list = ['text_a' + '\t' + 'text_b' + '\t' + 'label' + '\n']
            for example in dataset:
                example_acc += 1
                question = example['question']
                qid = example['example_id']
                gold_entity = example['gold_entity']
                candidates_dict = json.loads(example['candidates'])
                for c in candidates_dict.keys():
                    if c == gold_entity:
                        label = 1
                        positive_acc += 1
                    else:
                        label = 0
                    test_data_list.append(question + '\t' + c + '\t' + str(label) + '\n')
            if example_acc == positive_acc:
                with open(dump_data_path[0], 'w') as f_obj:
                    for i in test_data_list:
                        f_obj.write(i)
            else:
                logger.error('检查json数据')
        else:
            logger.error("检查输入参数,第二个参数为输出路径列表（长度为1或者2）")
    elif ds_name == 'ccksel19':
        if mention2id_path == '':
            logger.error('请指定mention2id_path参数')
        mention2id = {i['mention']: i['candidates'] for i in load_jsonl(mention2id_path)}
        dataset = load_jsonl(load_data_path)
        ed_data = ['example_id' + '\t' + 'text_a' + '\t' + 'text_b' + '\t' + 'label' + '\n']
        random.shuffle(dataset)
        if ds_type == 'train':
            dataset = dataset[:8000]
        elif ds_type == 'dev':
            dataset = dataset[8000:9000]
        elif ds_type == 'test':
            dataset = dataset[9000:10000]
        else:
            logger.error('参数错误')
        for eid, example in enumerate(dataset):
            # if eid > 100:
            #     break
            text_id = example['text_id']
            question = example['text']
            print(question)
            mention_data = example['mention_data']
            print(mention_data)
            for mid, mention_dict in enumerate(mention_data):
                mention = mention_dict['mention']
                gold_entity_id = mention_dict['kb_id']
                # line_id = str(text_id) + '_' + str(mid)
                # 找到mention的候选实体
                if mention in mention2id and gold_entity_id != "NIL":
                    candidates_dict = mention2id[mention]
                    for cid, (k, v) in enumerate(candidates_dict.items()):
                        line_id = str(text_id) + '_' + str(mid) + '_' + str(k)  # 案例id + 提及序号 + 知识库实体id
                        if k == gold_entity_id:
                            label = 1
                        else:
                            label = 0
                        ed_data.append(line_id + '\t' + question + '\t' + v + '\t' + str(label) + '\n')
                elif gold_entity_id == "NIL":
                    line_id = str(text_id) + '_' + str(0) + '_' + str(0)  # 案例id + 提及序号 + 知识库实体id
                    ed_data.append(line_id + '\t' + question + '\t' + "NIL" + '\t' + str(1) + '\n')
                else:
                    logger.info('案例中【{}】的提及【{}】在mention2id找不到，请检查。'.format(question, mention))
        with open(dump_data_path[0], 'w') as f_obj:
            for i in ed_data:
                f_obj.write(i)


if __name__ == "__main__":
    # train
    # load_data_path = "../data/processed/nlpcc_kbqa/train.json"
    # dump_train_data_path = "../data/processed/nlpcc_kbqa/ed/train.tsv"
    # dump_dev_data_path = "../data/processed/nlpcc_kbqa/ed/dev.tsv"
    # create_ed_dataset(load_data_path, [dump_train_data_path, dump_dev_data_path])
    # # test
    # load_data_path = "../data/processed/nlpcc_kbqa/test.json"
    # dump_data_path = "../data/processed/nlpcc_kbqa/ed/test.tsv"
    # create_ed_dataset(load_data_path, [dump_data_path])

    # 判断一下通过最长公共子串找到的mention,是不是都等于gold_entity
    # is_gold_entity_equal_mention("../data/processed/nlpcc_kbqa/test.json")

    # 对kbert 生成对应格式的数据集
    load_data_path = "../data/processed/nlpcc_kbqa/train.json"
    dump_train_data_path = "../data/processed/nlpcc_kbqa/ed_kbert/train.tsv"
    dump_dev_data_path = "../data/processed/nlpcc_kbqa/ed_kbert/dev.tsv"
    create_ed_dataset_kbert(load_data_path, [dump_train_data_path, dump_dev_data_path])
    # # test
    load_data_path = "../data/processed/nlpcc_kbqa/test.json"
    dump_data_path = "../data/processed/nlpcc_kbqa/ed_kbert/test.tsv"
    create_ed_dataset_kbert(load_data_path, [dump_data_path])
