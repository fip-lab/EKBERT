#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : utils.py
# Date    : 2022-08-16
import json
from typing import Dict, List, Any, Tuple


def clear_format(org_str: str) -> str:
    """
    统一标点为英文符号
    :param org_str:
    :return:
    """
    pun = {'，': ',',
           '。': '.',
           '：': ':',
           '！': '!',
           '（': '(',
           '）': ')',
           '；': ';',
           '？': '?',
           ' ': '',
           '\n': '',
           }
    for p in pun:
        if p in org_str:
            org_str = org_str.replace(p, pun[p])
    org_str = org_str.lower()
    return org_str


def load_jsonl(path: str):
    data_list = []
    with open(path, "r") as f:
        for line in f.readlines():
            data_list.append(json.loads(line))
    return data_list


def dump_jsonl(data_list: List[Any], path: str):
    with open(path, "w") as f:
        for json_obj in data_list:
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


def find_lcsubstr(s1: str, s2: str) -> Tuple[str, int]:
    """
    找到两个字符串的最长公共子串
    :param s1:
    :param s2:
    :return:
    """
    # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长子串及其长度


# print(find_lcsubstr('《高等数学一（微积分）》是哪一门课的通用教材？', '高等数学'))


def csv_reader(file_name):
    for row in open(file_name, "r"):
        yield row


def csv_reader_static(file_name):
    f_list = []
    for row in open(file_name, "r"):
        f_list.append(row)
    return f_list


def csv_writer(data_list, path):
    with open(path, 'w') as writer:
        for i in data_list:
            writer.write(i)
            
            
def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            # print('tag',tag)
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    """
    :param seq:
    :param id2label:
    :param markup:
    :return:
    """
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)



if __name__ == "__main__":
    # print(find_lcsubstr('《高等数学一（微积分）》是哪一门课的通用教材？', '高等数学'))
    # alist = [1,2,3,2,3]
    # alist.remove(2)
    # print(alist)
    get_entities()

