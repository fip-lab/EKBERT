#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : knowledge.py
# Date    : 2022-08-29
import os
from typing import Dict, Tuple
import ahocorasick
from utils import csv_reader
from loguru import logger

FILE_DIR_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))

KGS = {
    "nlpcc_train": os.path.join(FILE_DIR_PATH, 'data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_train.spo'),
    "nlpcc_test": os.path.join(FILE_DIR_PATH, 'data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_test.spo'),
    "nlpcc": os.path.join(FILE_DIR_PATH, 'data/raw/nlpcc_kbqa/nlpcc-iccpol-2016-new.spo'),
    "ccks_el_2019": os.path.join(FILE_DIR_PATH, 'data/processed/ccks_el_2019/ccks_2019_kb.spo'),
    "ccks_el_2020": os.path.join(FILE_DIR_PATH, 'data/processed/ccks_el_2020/ccks_2020_kb.spo'),
    "ccks_el_2020_m2id": os.path.join(FILE_DIR_PATH, 'data/processed/ccks_el_2020/mention2id.txt'),
    "ccks_kbqa_2020": os.path.join(FILE_DIR_PATH, 'data/processed/ccks_kbqa_2020/ccks_kbqa_2020_filter.spo'),
    "ccks_kbqa_2020_m2id": os.path.join(FILE_DIR_PATH, 'data/processed/ccks_kbqa_2020/mention2id.txt'),
}


class Knowledge(object):
    """
    注入实体边界和子图信息
    """

    def __init__(self, spo_files, min_span: int = 3):
        self.spo_file_paths = [KGS.get(f) for f in spo_files]
        self.min_span = min_span
        self.lookup_table = self._create_lookup_table()
        self.a = self._create_automaton()

    def _create_lookup_table(self) -> Dict:  # 以单个下划线开头的变量或方法仅供内部使用
        lookup_table = {}
        for spo_path in self.spo_file_paths:
            print("[KnowledgeGraph] Loading spo from {}".format(spo_path))
            with open(spo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        subj, pred, obje = line.strip().split("\t")
                        subj = subj.replace(' ', '')
                        pred = pred.replace(' ', '')
                        obje = obje.replace(' ', '')
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if len(subj) >= self.min_span:
                        value = (pred, obje)
                        if subj in lookup_table.keys():
                            lookup_table[subj].add(value)
                        else:
                            lookup_table[subj] = set([value])
        return lookup_table

    def _create_automaton(self):
        # 准备好所有span搜索字典
        a = ahocorasick.Automaton()
        for idx, word in enumerate(self.lookup_table.keys()):
            a.add_word(word, (idx, word))
        a.make_automaton()
        return a

    def infuse_entities_border(self, sentence: str, label: str) -> Tuple[str, str]:
        """
        为md数据集的句子和标签，注入实体边界，并返回新的标签
        :param sentence:
        :param label:
        :return:
        eg.
        sentence:"你 知 道 金 山 旅 游 区 南 北 有 多 长 吗 ?"
        label:"O O O B-entity I-entity I-entity I-entity I-entity O O O O O O O"
        return:
        ('| 你 | 知 道 | 金 | 山 旅 游 区 | 南 北 有 多 长 吗 ?','O O O B-entity I-entity I-entity I-entity I-entity O O O O O O O')
        """
        # 记录sentence匹配到的span位置
        entities_pos = []
        # text = sentence.split(' ')
        try:
            assert (len(sentence.split(' ')) == len(label.split(' ')))
        except AssertionError:
            logger.error('ner任务的文本长度和标签长度不相同。请检查数据。text:{},label:{}'.format(sentence, label))
            return sentence, label
        sentence = ''.join(sentence.split(' '))
        label = label.split(' ')
        for k in self.a.iter(sentence):
            # print(k)
            entities_pos.append(k)
        insert_time = 0
        insert_idx = set()  # 得到插入边界'|'的位置集合
        for end, (idx, span) in entities_pos:
            beg = end - len(span)
            new_beg = beg + 1
            new_end = end + 1
            insert_idx.add(new_beg)
            insert_idx.add(new_end)
        insert_idx = list(insert_idx)
        insert_idx.sort()
        # print(insert_idx)
        for s in insert_idx:
            dynam_insert_pos = s + insert_time
            new_sent = sentence[:dynam_insert_pos] + '|' + sentence[dynam_insert_pos:]  # 新的问句
            new_labels = label[:dynam_insert_pos] + ['O'] + label[dynam_insert_pos:]  # 新的标签
            insert_time += 1
            sentence = new_sent
            label = new_labels
        # 新句子、新标签后处理
        sentence = ' '.join(sentence)
        label = ' '.join(label)
        label = label.replace('O I-entity', 'I-entity I-entity')  # 最终的标签
        # print(sentence)
        # print(label)
        return sentence, label


if __name__ == "__main__":
    # nlpcc 数据集测试 输入
    # examples = [
    #     ("你 知 道 金 山 旅 游 区 南 北 有 多 长 吗 ?", "O O O B-entity I-entity I-entity I-entity I-entity O O O O O O O"),
    #     ("兰 州 大 学 大 气 科 学 学 院 学 生 会 的 机 构 是 什 么 ?", "B-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity O O O O O O O")
    # ]
    #
    # # 准备kg数据
    # spo_files = "nlpcc_train+nlpcc_test".split('+')
    # kg = Knowledge(spo_files, min_span=3)
    # for text_a, label in examples:
    #     label = label.split(" ")
    #     logger.info('注入前,{}-{}'.format(text_a, label))
    #     text_a, label = kg.infuse_entities_border(text_a, ' '.join(label))
    #     text_a = text_a
    #     label = label.split(' ')
    #     logger.info('注入后,{}-{}'.format(text_a, label))

    # |你|知道|金|山旅游区|南北有多长吗?
    # O O O B-entity I-entity I-entity I-entity I-entity O O O O O O O

    # ccks 19 el 数据集测试 输入
    examples = [
        ("d a v i d g a r r e t t 《 野 蜂 飞 舞 》 小 提 琴 独 奏 : 小 提 琴 演 奏 . . .",
         "B-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity I-entity O B-entity I-entity I-entity I-entity O B-entity I-entity I-entity O O O B-entity I-entity I-entity O O O O O")
    ]

    # 准备kg数据
    spo_files = "ccks_el_2019".split('+')
    kg = Knowledge(spo_files, min_span=3)
    for text_a, label in examples:
        label = label.split(" ")
    logger.info('注入前,{}-{}'.format(text_a, label))
    text_a, label = kg.infuse_entities_border(text_a, ' '.join(label))
    text_a = text_a
    label = label.split(' ')
    print(len(text_a.split(' ')))
    print(len(label))
    # assert (len(text_a.split(' ')) == len(label))
    logger.info('注入后,{}-{}'.format(text_a, label))
