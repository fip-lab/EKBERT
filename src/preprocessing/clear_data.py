#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : entity_disambiguation.py
# Date    : 2022-08-16
from typing import Set, List, Dict, Tuple, Any, Generator
from collections import defaultdict, OrderedDict
import re
import json
from utils import clear_format, load_jsonl, dump_jsonl, find_lcsubstr, csv_reader, csv_reader_static
import traceback
import os
from tqdm import tqdm
import sys
from loguru import logger


def new_alias(load_org_kb_path, load_dataset_path):
    """
    从 https://github.com/panchunguang/ccks_baidu_entity_link 拷贝
    统计训练数据中不能链接到实体库的mentioin, 统计出现次数，将其添加到对应实体的别名中
    :return: 字典形式 key 为实体名字 value 为添加的新的别名字典
             如：'bilibili': {'b站', '哔哩哔哩', '哔哩哔哩弹幕视频网'}
    :param load_org_kb_path:
    :param load_dataset_path:
    :return:entity_alias_num 判断一个实体是否需要添加一个别名
    """
    id_alias = {}
    entity_id = {}
    id_entity = {}
    with open(load_org_kb_path, 'r') as f:
        for line in f:
            temDict = json.loads(line)
            subject = temDict['subject']
            subject_id = temDict['subject_id']
            alias = set()
            for a in temDict['alias']:
                alias.add(clear_format(a))
                # alias.add(a.lower())
            # alias.add(subject.lower())
            # alias.add(entity_clear(subject))
            id_alias[subject_id] = alias
            subject_id = temDict['subject_id']
            entity_name = set(alias)
            entity_name.add(clear_format(subject))
            # entity_name.add(subject.lower())
            for a in alias:
                entity_name.add(clear_format(a))
            id_entity[subject_id] = subject
            for n in entity_name:
                if n in entity_id:
                    entity_id[n].add(subject_id)
                else:
                    entity_id[n] = set()
                    entity_id[n].add(subject_id)
    with open(load_dataset_path) as f:
        entity_alias_num = {}
        for line in f:
            temDict = json.loads(line)
            mention_data = temDict['mention_data']
            for men in mention_data:
                mention = men['mention']
                kb_id = men['kb_id']
                if kb_id != 'NIL':
                    if id_entity[kb_id] != mention:
                        if mention not in id_alias[kb_id]:
                            if id_entity[kb_id] in entity_alias_num:
                                entity_alias_num[id_entity[kb_id]]['count'] += 1
                                if mention in entity_alias_num[id_entity[kb_id]]:
                                    entity_alias_num[id_entity[kb_id]][mention] += 1
                                else:
                                    entity_alias_num[id_entity[kb_id]][mention] = 1
                            else:
                                entity_alias_num[id_entity[kb_id]] = {}
                                entity_alias_num[id_entity[kb_id]]['count'] = 1
                                entity_alias_num[id_entity[kb_id]][mention] = 1
    entity_alias = {}
    for en in entity_alias_num:
        total_num = entity_alias_num[en]['count']
        if total_num > 4:
            entity_alias[en] = set()
            for alias in entity_alias_num[en]:
                if alias == 'count':
                    continue
                a_num = entity_alias_num[en][alias]
                if a_num > 3:
                    entity_alias[en].add(alias)
            if len(entity_alias[en]) == 0:
                entity_alias.pop(en)
    return entity_alias, entity_alias_num


def get_cantitidate_entity(dataset_mentions: Set, mention2id_path) -> Dict:
    """
    输入数据集涉及的所有提及词，通过提供的mention2id文件找到提及词对应的候选实体，
    返回{mention,set(candidate1,candidate2,...)
    :param dataset_mentions: 提及
    :param mention2id_path: 官方提供的mention2id文件路径
    :return:
    """
    mention_candidates_dict = OrderedDict()
    gen = csv_reader(mention2id_path)

    for i in gen:
        try:
            mention, candidates = i.split('|||')
            mention = clear_format(mention)
            if mention != '' and mention in dataset_mentions:  # 列表通过 infile 得到
                candidates = [clear_format(j) for j in candidates.split('\t')]
                if mention in mention_candidates_dict.keys():
                    for c in candidates:
                        mention_candidates_dict[mention].add(c)
                else:
                    mention_candidates_dict[mention] = set()
                    for c in candidates:
                        mention_candidates_dict[mention].add(c)
                logger.info("提及:{}对应的候选实体:{}".format(mention, mention_candidates_dict[mention]))
        except ValueError:
            logger.error('mention2id文件的行格式错误:{}'.format(i))
            continue
    return mention_candidates_dict


def get_a_mention_canditidate_entity(example_mention: str, mention2id_path: str) -> Set:
    """
    给定一个提及词，通过mention2id文件得到对应的候选集合
    :param example_mention: 提及
    :param mention2id_path: 官方提供的mention2id文件
    :return:
    """
    gen = csv_reader(mention2id_path)
    candidates_set = set()
    for i in gen:
        try:
            mention, candidates = i.split('|||')
            mention = clear_format(mention)
            if mention != '' and mention == example_mention:  # 列表通过 infile 得到
                candidates = [clear_format(j) for j in candidates.split('\t')]
                for c in candidates:
                    candidates_set.add(c)
        except ValueError:
            logger.error('mention2id文件的行格式错误:{}'.format(i))
            continue
    logger.info("{}:{}".format(example_mention, candidates_set))
    return candidates_set


def find_gold_entity(kb_gen: Generator or List,
                     gold_relation: str,
                     gold_answer: str,
                     candidates_set: Set) -> str:
    """
    通过标签三元组中的关系和答案 倒找到 目标实体 gold_entity
    :param kb_gen:
    :param gold_relation:
    :param gold_answer:
    :param candidates_set:
    :return:
    """
    gold_entity = ""
    for i in kb_gen:
        s, p, o = i.split('\t')
        s = clear_format(s)
        if s in candidates_set:
            p = clear_format(p)
            o = clear_format(o)
            if p == gold_relation and o == gold_answer:
                gold_entity = s
                break
    return gold_entity


def constract_example_data(raw_train_path) -> Dict:
    """
    读取原始的数据集，转成案例字典
    :param raw_train_path:
    :return:
    """
    gen = csv_reader(raw_train_path)
    # kb_gen = csv_reader(kb_path)
    examples_dict = OrderedDict()
    rowid = 0
    try:
        for i in gen:
            if rowid % 4 == 0:
                question = ""
                mention = ""
                gold_entity = ""
                gold_relation = ""
                example_id = ""
            i = clear_format(i)
            rowid += 1
            if rowid % 4 == 1:
                example_id = re.findall(r'(\d+)', i)[0]
                question = clear_format(i.split('\t')[1])
            if rowid % 4 == 2:
                _, triple = i.split('\t')
                mention, gold_relation, gold_answer = triple.split('|||')  # 官方提供的标签三元组 第一个是mention
                mention = clear_format(mention)
                gold_relation = clear_format(gold_relation)
                gold_answer = clear_format(gold_answer)
                examples_dict[example_id] = {
                    "question": question,
                    "mention": mention,
                    "gold_answer": gold_answer,
                    "gold_relation": gold_relation,
                }
    except:
        logger.error('{}出现问题{}'.format(i, traceback.format_exc()))
    return examples_dict


def is_gold_entity_equal_mention(load_data_path: str):
    dataset = load_jsonl(load_data_path)
    example_count = 0
    equal_count = 0
    for example in dataset:
        example_count += 1
        gold_entity = example['gold_entity']
        mention = example['mention']
        if gold_entity == mention:
            equal_count += 1
        else:
            logger.info('不等于的mention{}和gold_entity{}'.format(mention, gold_entity))
    logger.info('数据集中共{}条数据，有{}条mention等于gold_entity'.format(example_count, equal_count))


def filter_kb_by_candidates(mention_candidates_path, kb_path, dump_data_path):
    """
    从kb所有spo中筛选出数据集涉及的候选实体的spo
    :param mention_candidates_path: 读取所有mention对应候选的文件路径
    :param kb_path: # 原始kb路径
    :param dump_data_path: # 写出路径
    :return:
    """
    mention_candidates_dict = {j['mention']: json.loads(j['candidates']).keys() for j in load_jsonl(mention_candidates_path)}
    # print('aa', mention_candidates_dict.keys())
    # print('bb', mention_candidates_dict.values())
    filter_candidates = {clear_format(i) for k, v in mention_candidates_dict.items() for i in v}
    logger.info("数据集涉及的候选实体数量:{}个,前10个候选实体为:{}".format(len(filter_candidates), list(filter_candidates)[:10]))
    kb_gen = csv_reader(kb_path)
    data_list = []
    for i in kb_gen:
        s, p, o = i.split('\t')
        s = clear_format(s)
        if s in filter_candidates:
            p = clear_format(p)
            o = clear_format(o)
            data_list.append(s + '\t' + p + '\t' + o + '\n')
    logger.info("筛选出spo:{}条".format(len(data_list)))
    with open(dump_data_path, 'w') as f_obj:
        for i in data_list:
            f_obj.write(i)


def clear_nlpcc_kbqa_data(load_data_path: str,
                          mention2id_path: str,
                          kb_path: str,
                          mention_candidates_path: str,
                          filter_kb_path: str,
                          dump_data_path: str,
                          example_id=None):
    """
    清洗nlpcc kbqa数据
    :param load_data_path: 官方提供原始数据集
    :param mention2id_path: 官方提供mention2id
    :param kb_path: 官方提供kb
    :param mention_candidates_path: 提及和候选映射输出路径
    :param filter_kb_path: 筛选所有候选实体的spo的输出路径
    :param dump_data_path: 清洗之后的数据集的输出路径
    :param example_id: 指定一个案例id
    :return:
    ---------------------
    给的训练数据集三元组spo中，s代表gold_entity错误，处理逻辑：通过mention找到所有candidates，通过标签spo中的p关系和o答案匹配candidates对应的spo，反向找到匹配的s作为gold_entity
    eg.
    <question id=1>	《机械设计基础》这本书的作者是谁？
    <triple id=1>	机械设计基础 ||| 作者 ||| 杨可桢，程光蕴，李仲生
    <answer id=1>	杨可桢，程光蕴，李仲生
    正确的triple是：机械设计基础(2010年高等教育出版社出版作者杨可桢)	作者	杨可桢，程光蕴，李仲生
    """
    examples_dict = constract_example_data(load_data_path)
    mention_list = []
    for i in examples_dict.keys():
        mention_list.append(examples_dict[i]["mention"])
    # print(len(mention_list))
    dataset_mentions = set(mention_list)
    # print(len(dataset_mentions))
    # 得到提及词和候选实体的映射
    if os.path.exists(mention_candidates_path):
        mention_candidates_dict = {j['mention']: json.loads(j['candidates']).keys() for j in load_jsonl(mention_candidates_path)}
    else:
        # 对于 在数据集 且 在mention2id 的提及
        mention_candidates_dict = get_cantitidate_entity(dataset_mentions, mention2id_path)
        # 对于 在数据集 但 不在mention2id 的提及
        for m in dataset_mentions:
            if m not in mention_candidates_dict.keys():
                mention_candidates_dict[m] = {m}
        data_list = []
        for i in mention_candidates_dict.keys():
            # 对 candidates 排序
            sorted_candidates = sorted(mention_candidates_dict[i])
            data_list.append({
                "mention": i,
                "candidates": json.dumps({c: cid for cid, c in enumerate(sorted_candidates)}, ensure_ascii=False)
            })
        dump_jsonl(data_list, mention_candidates_path)

    # 从所有的spo中筛选出所有candidates的spo
    if not os.path.exists(filter_kb_path):
        filter_kb_by_candidates(mention_candidates_path, kb_path, filter_kb_path)
    # 两种读取kb的方式
    filter_kb_gen = csv_reader_static(filter_kb_path)  # 静态 耗内存 快
    # filter_kb_gen = csv_reader(filter_kb_path) # 动态 不耗内存 慢

    # 找到gold_entity
    data_list = []
    if example_id:
        examples_dict = {key: value for key, value in examples_dict.items() if key == example_id}
        print('examples_dict', examples_dict)
    for key, value in examples_dict.items():
        try:
            question = value["question"]
            mention = value["mention"]
            gold_answer = value["gold_answer"]
            gold_relation = value["gold_relation"]
            candidates = mention_candidates_dict[mention]
            gold_entity = find_gold_entity(filter_kb_gen, gold_relation, gold_answer, candidates)
            if gold_entity == "":
                logger.error('问题:{}，通过提及:{}找到的候选:{},在条件关系:{}+答案:{}找不到金实体，请检查'.format(key, mention, candidates, gold_relation, gold_answer))
            else:
                logger.info('问题:{}，提及:{}，金实体:{}'.format(key, mention, gold_entity))
            if gold_entity != "":
                candidates_json_obj = json.dumps({c: cid for cid, c in enumerate(list(candidates))}, ensure_ascii=False)
                data_list.append({"example_id": key,
                                  "question": question,
                                  "mention": mention,
                                  "candidates": candidates_json_obj,
                                  "gold_entity": gold_entity,
                                  "gold_relation": gold_relation,
                                  "gold_answer": gold_answer,
                                  })
        except:
            continue
    # print('mention_candidates_dict', mention_candidates_dict)
    if not example_id:
        with open(dump_data_path, "w") as f:
            for json_obj in data_list:
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


def clear_ccks_el_kb_data(load_kb_data_path, dump_kb_data_path, load_dataset_path, dump_mention2id_path):
    """
    处理原始的ccks kb数据，导出spo三元组格式的kb文件
    :param load_kb_data_path:
    :param dump_kb_data_path:
    :param load_dataset_path:
    :param dump_mention2id_path:
    :return:
    """
    # clear spo file
    if not os.path.exists(dump_kb_data_path):
        entities = load_jsonl(load_kb_data_path)
        spo_list = []
        rowcnt = 0
        for entity in tqdm(entities):
            # rowcnt += 1
            # if rowcnt > 10:
            #     break
            subject = entity['subject']
            subject_id = entity['subject_id']
            spo_list.append((clear_format(subject), '实体id', clear_format(subject_id)))
            data = entity['data']
            for d in data:
                predicate = d['predicate']
                obj = d['object']
                spo_list.append((clear_format(subject), clear_format(predicate), clear_format(obj)))
        with open(dump_kb_data_path, 'w') as f_obj:
            for s, p, o in spo_list:
                f_obj.write(s + '\t' + p + '\t' + o + '\n')
    # find mention candidates
    # 通过mention 去匹配spo文件中 的 实体别名 得到mention to candidates文件
    if not os.path.exists(dump_mention2id_path):
        entity_id = {}
        id_entity = {}
        new_entity_alias, _ = new_alias(load_kb_data_path, load_dataset_path)
        entities = load_jsonl(load_kb_data_path)
        for entity in tqdm(entities):
            subject = entity['subject']
            subject_id = entity['subject_id']
            id_entity[subject_id] = subject
            alias = set()
            for a in entity['alias']:
                alias.add(clear_format(a))
            alias.add(clear_format(subject))
            if subject in new_entity_alias:
                alias = alias | new_entity_alias[subject]  # 并集
            entity_name = set(alias)
            for n in entity_name:
                if n in entity_id:
                    entity_id[n].append(subject_id)
                else:
                    entity_id[n] = []
                    entity_id[n].append(subject_id)
        data_list = []
        for k, v in tqdm(entity_id.items()):
            candidates_dict = {i: id_entity[i] for i in v}
            data_list.append({'mention': k,
                              'candidates': candidates_dict})
        dump_jsonl(data_list, dump_mention2id_path)


if __name__ == "__main__":
    new_entity_alias, entity_alias_num = new_alias(
        '/Users/penho/Documents/GDUT/PythonProject/EKBERT/src/data/raw/ccks_el_2019/kb_data',
        '/Users/penho/Documents/GDUT/PythonProject/EKBERT/src/data/raw/ccks_el_2019/train.json')
    print(new_entity_alias, entity_alias_num)
