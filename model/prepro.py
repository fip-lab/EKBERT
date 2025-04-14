# coding=utf-8

# 对处理好符合格式的数据文件（.tsv  .txt）转换成bert定义的输入数据格式。
# 包括分词、转换成数字下标，添加特殊字符，截断成固定最大长度，定义attention_mask区分哪些是真正的字符，定义token_type_ids区分不同的句子

# import logging
import os
from enum import Enum
from typing import List, Optional, Union
# python注解 List[类型]表示是类型的列表，List[int or float] = [2, 3.5],
# Union[X, Y] 代表要么是 X 类型，要么是 Y 类型。
# Optional，意思是说这个参数可以为空或已经声明的类型，即 Optional[X] 等价于 Union[X, None]。
import dataclasses
import numpy as np
import json
from dataclasses import dataclass
import torch
import codecs
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers.data.processors import DataProcessor
from transformers.file_utils import is_tf_available
from transformers.tokenization_utils import PreTrainedTokenizer
# from knowledge import Knowledge
from loguru import logger
from utils import dump_jsonl
import traceback

# logger = logging.getLogger(__name__)

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>", "<kg>", "</kg>"]


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None  # text_b可以是str类型也可以是None
    label: Optional[str] = None  # label可以是str类型也可以是是None
    pos: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class NERInputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    tokens1: List[str]
    input_ids1: List[int]
    attention_mask1: Optional[List[int]]
    token_type_ids1: Optional[List[int]]
    label_id: Optional[Union[int, float]]
    tokens2: Optional[List[str]] = None
    input_ids2: Optional[List[int]] = None
    attention_mask2: Optional[List[int]] = None
    token_type_ids2: Optional[List[int]] = None
    e1_mask: Optional[List[int]] = None
    e2_mask: Optional[List[int]] = None
    k_mask: Optional[List[int]] = None
    e_begin_mask: Optional[List[int]] = None
    e_end_mask: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class NERInputFeatures:
    tokens1: List[str]
    input_ids1: List[int]
    attention_mask1: Optional[List[int]]
    token_type_ids1: Optional[List[int]]
    label_id: Optional[List[Union[int, float]]]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


'''
分类任务
'''


class ClassificationProcessor(DataProcessor):  # 继承DataProcessor
    # 得到数据集中的所有sample的文本以及label
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train_part.tsv"))  # 读取 data_preprocess 处理好的训练数据文件.tsv
        examples = []  # example列表
        for (i, line) in enumerate(lines):  # 下标，数据集中的行  enumerate是键值结构
            guid = "%s-%s" % ("train", i)
            text_a = line[0]  # 行的第一列 是 训练文本
            label = line[1]  # 行的第二列 是 训练标签   看train.tsv的结构
            # 把 guid,text_a,label三个字段构造为InputExample对象，添加到example列表中
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))

            # print(examples)
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


'''
文本匹配任务
'''


class MatchingProcessor(DataProcessor):  # 继承DataProcessor

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))  # Reads a tab separated value file.
        # lines = lines[:50]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            # guid = line[0]
            # text_a = line[1].replace(' ', '')
            # text_b = line[2].replace(' ', '')
            # label = line[3]
            guid = "%s-%s" % ("train", i)  # test-1
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))  # file_name 通过参数传入 默认 test.tsv 可指定为dev.tsv
        # lines = lines[:50]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            # guid = line[0]  # test-1
            # text_a = line[1].replace(' ', '')
            # text_b = line[2].replace(' ', '')
            # label = line[3]
            guid = "%s-%s" % ("test", i)  # test-1
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


class CcksEkbertProcessor(DataProcessor):  # 继承DataProcessor

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))  # Reads a tab separated value file.
        # lines = lines[:50]  # 读前100条数据 调通程序
        examples = []
        try:
            for (i, line) in enumerate(lines):
                # guid = line[0]
                # text_a = line[1].replace(' ', '')
                # text_b = line[2].replace(' ', '')
                # label = line[3]
                guid = "%s-%s" % ("train", i)  # test-1
                text_a = line[1]
                text_b = line[2]
                label = line[3]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        except IndexError:
            print(line)
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))  # file_name 通过参数传入 默认 test.tsv 可指定为dev.tsv
        # lines = lines[:50]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            # guid = line[0]  # test-1
            # text_a = line[1].replace(' ', '')
            # text_b = line[2].replace(' ', '')
            # label = line[3]
            guid = "%s-%s" % ("test", i)  # test-1
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


class EMatchingProcessor(DataProcessor):  # 继承DataProcessor
    """
    带实体token<e1></e1> <e2></e2>的实体消歧数据处理器
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))  # Reads a tab separated value file.
        # lines = lines[:10]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)  # test-1
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))  # file_name 通过参数传入 默认 test.tsv 可指定为dev.tsv
        # lines = lines[:10]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)  # test-1
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


class CcksNo1MatchingProcessor(DataProcessor):  # 继承DataProcessor
    """
    需要记录mention的开始、结束位置
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train_baseline.tsv"))  # Reads a tab separated value file.
        # lines = lines[:10]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)  # test-1
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            pos = line[4]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))  # file_name 通过参数传入 默认 test.tsv 可指定为dev.tsv
        # lines = lines[:10]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)  # test-1
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            pos = line[4]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, pos=pos))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


class EKGMatchingProcessor(DataProcessor):  # 继承DataProcessor
    """
    带实体token<e1></e1> <e2></e2> <kg></kg>的实体消歧数据处理器
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train_e1_e2_kg.tsv"))  # Reads a tab separated value file.
        lines = lines[:10]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)  # test-1
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))  # file_name 通过参数传入 默认 test.tsv 可指定为dev.tsv
        lines = lines[:10]  # 读前100条数据 调通程序
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)  # test-1
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


'''
序列标注ner任务
'''


class NerProcessor(DataProcessor):  # 继承DataProcessor
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))  # 读出 处理好符合格式的数据   即\t（tab制表符）作为句子和标签的分隔符
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]  # 取出文本句子
            label = line[1].split(" ")  # 空格分割每一个字符子串,即取出每一个token标签
            examples.append(NERInputExample(guid=guid, text_a=text_a, label=label))  # 初始化一个NERInputExample，序列标注任务的输入案例
        return examples

    def get_test_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("test", i)  # test-1,test-2 ...
            text_a = line[0]
            label = line[1].split(" ")
            examples.append(NERInputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_labels(self, data_dir):
        """See base class."""
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels


# 字典 对不同任务的原始输入数据 进行处理
processors = {
    "afqmc": MatchingProcessor,
    "lcqmc": MatchingProcessor,
    "nlpccpm": MatchingProcessor,
    "chnsenti": ClassificationProcessor,
    "tnews": ClassificationProcessor,
    "weiboner": NerProcessor,
    "msraner": NerProcessor,
    "nlpccmd": NerProcessor,
    "nlpcced": MatchingProcessor,
    "sfarerank": MatchingProcessor,
    "nlpccedpoetoken": EMatchingProcessor,
    "nlpccedpoekgtoken": EKGMatchingProcessor,
    "cckselbaseline": CcksNo1MatchingProcessor,
    "cckseledpoetoken": CcksEkbertProcessor,
    "ekbertmd": NerProcessor,
    "cckselbaselinemd": NerProcessor,
    "cckskbqaedpoetoken": CcksEkbertProcessor,
}

# 字典 对不同任务的原始标签数据 进行处理
output_modes = {
    "chnsenti": "classification",
    "tnews": "classification",
    "weiboner": "ner",
    "msraner": "ner",
    "nlpccmd": "ner",
    "lcqmc": "matching",
    "afqmc": "matching",
    "nlpccpm": "matching",
    "nlpcced": "matching",
    "sfarerank": "matching",
    "nlpccedpoetoken": "matching",
    "nlpccedpoekgtoken": "matching",
    "cckselbaseline": "matching",
    "ekbertmd": "ner",
    "cckseledpoetoken": "matching",
    "cckselbaselinemd": "ner",
    "cckskbqaedpoetoken": "matching",
}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.
    当输入的文本超过指定的max_length，则截断多出的token部分，函数从两段文本中平衡长度
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:  # 跳出条件
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# 获得所有tsv文件中的文本数据，转成特征
def convert_examples_to_features(
        examples: Union[List[InputExample]],  # Union
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int],
        label_list: List[str],
        output_mode: str,
        kg_enhance: bool,
        kg_enhance_2: bool,
        kg_enhance_3: bool,
        ccks_no_1: bool,
):
    def convert_text_to_ids(text):
        """
        得到每个sample的文本增加特殊字符并转为字典的ID的list
        :param text:
        :return:
        """
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        tokens = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]  # 只要前面 最大长度-2的tokens
        text_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(
            tokens + ["[PAD]"] * (max_length - text_len))  # 调用库工具将分词转为分词所在词表上的id
        attention_mask = [1] * text_len + [0] * (max_length - text_len)  # attention_mask文本长度的token都要参加注意力的计算
        token_type_ids = [0] * max_length  # token_type_ids表示文本1和文本2是不同的句子段落

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        return tokens, input_ids, attention_mask, token_type_ids
        # 针对句间关系任务，添加特殊token如:[CLS]句子1[SEP]句子2[SEP][PAD][PAD]...[PAD]

    def convert_text_to_ids_for_matching(text_a, text_b):
        tokens_a = tokenizer.tokenize(text_a)  # 对文本进行分词
        tokens_b = tokenizer.tokenize(text_b)  # 对文本进行分词
        if len(tokens_a) + len(tokens_b) > (max_length - 3):  # 如果两文本长度 比 最大长度-3([CLS]1[SEP]2[SEP])，就要截断输入文本。
            _truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        text_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(
            tokens + ["[PAD]"] * (max_length - text_len))  # 调用库工具将分词转为分词所在词表上的id
        attention_mask = [1] * text_len + [0] * (max_length - text_len)  # attention_mask文本长度的token都要参加注意力的计算
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) + [0] * (
                max_length - text_len)  # token_type_ids表示文本1和文本2是不同的句子段落

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        return tokens, input_ids, attention_mask, token_type_ids

    def convert_text_to_ids_for_matching_ccks_no_1(text_a, text_b, pos):
        """
        :param text_a:
        :param text_b:
        :param pos:
        :return:
        """
        # tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        tokens_a = tokenizer.tokenize(text_a)  # 对文本进行分词
        tokens_b = tokenizer.tokenize(text_b)  # 对文本进行分词

        if len(tokens_a) + len(tokens_b) > (max_length - 3):  # 如果两个文本的长度 比 最大长度-3(即[CLS]句子1[SEP]句子2[SEP]三个特殊标签)，就要截断输入文本。
            _truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        text_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens + ["[PAD]"] * (max_length - text_len))  # 调用库工具将分词转为分词所在词表上的id
        attention_mask = [1] * text_len + [0] * (max_length - text_len)  # attention_mask文本长度的token都要参加注意力的计算
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) + [0] * (max_length - text_len)  # token_type_ids表示文本1和文本2是不同的句子段落
        pos_list = pos.split(',')
        e_begin_index = int(pos_list[0])
        e_end_index = int(pos_list[1])
        # e_begin_mask, e_end_mask 分别表示实体开始、结束位置的0，1列表
        e_begin_mask = [0] * len(attention_mask)  # 是一个列表，长度就是max_seq_len,对应实体片段的几个token值为1，其他位置为0
        e_end_mask = [0] * len(attention_mask)
        e_begin_mask[e_begin_index] = 1
        e_end_mask[e_end_index] = 1
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_length, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_length, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        return tokens, input_ids, attention_mask, token_type_ids, e_begin_mask, e_end_mask

    def convert_text_to_ids_for_matching_kg_enhance(text_a, text_b):
        """
        对于使用了实体词条<e1></e1><e2></e2>的数据集，先把这些词条转为$和#，然后转成id的列表，并记录特殊token的位置
        :param text_a:
        :param text_b:
        :return:
        """
        try:
            tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
            tokens_a = tokenizer.tokenize(text_a)  # 对文本进行分词
            tokens_b = tokenizer.tokenize(text_b)  # 对文本进行分词

            e11_p = tokens_a.index("<e1>")  # the start position of entity1  得到文本中<e1>这个特殊token的位置，比如13
            e12_p = tokens_a.index("</e1>")  # the end position of entity1   如15
            e21_p = tokens_b.index("<e2>")  # the start position of entity2  如18
            e22_p = tokens_b.index("</e2>")  # the end position of entity2   如20

            # Replace the token
            tokens_a[e11_p] = "$"  # 将文本中的 <e1> 换成 $ 对应下标[109]
            tokens_a[e12_p] = "$"
            tokens_b[e21_p] = "#"  # 将 <e2> 换成 # 对应下标[108]
            tokens_b[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            # Add 2 because of the [CLS] and [SEP]
            e21_p += 2 + len(tokens_a)
            e22_p += 2 + len(tokens_a)

            if len(tokens_a) + len(tokens_b) > (max_length - 3):  # 如果两个文本的长度 比 最大长度-3(即[CLS]句子1[SEP]句子2[SEP]三个特殊标签)，就要截断输入文本。
                _truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            text_len = len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens + ["[PAD]"] * (max_length - text_len))  # 调用库工具将分词转为分词所在词表上的id
            attention_mask = [1] * text_len + [0] * (max_length - text_len)  # attention_mask文本长度的token都要参加注意力的计算
            token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) + [0] * (max_length - text_len)  # token_type_ids表示文本1和文本2是不同的句子段落

            # e1 mask, e2 mask
            e1_mask = [0] * len(attention_mask)  # 是一个列表，长度就是max_seq_len,对应实体片段的几个token值为1，其他位置为0
            e2_mask = [0] * len(attention_mask)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_length, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_length, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        except:
            logger.error('错误案例，text_a:{},text_b:{}，详细错误：{}'.format(text_a, text_b, traceback.format_exc()))
        return tokens, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask

    def convert_text_to_ids_for_matching_kg_enhance_3(text_a, text_b):
        """
        20211006 add
        对于使用了实体词条<e1></e1><e2></e2>的数据集，先把这些词条转为$和#，然后转成id的列表，并记录特殊token的位置
        *同时在</e2>之后引入的子图信息，把attention_mask都关掉。
        :param text_a:
        :param text_b:
        :return:
        """
        tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        tokens_a = tokenizer.tokenize(text_a)  # 对文本进行分词
        tokens_b = tokenizer.tokenize(text_b)  # 对文本进行分词

        e11_p = tokens_a.index("<e1>")  # the start position of entity1  得到文本中<e1>这个特殊token的位置，比如13
        e12_p = tokens_a.index("</e1>")  # the end position of entity1   如15
        e21_p = tokens_b.index("<e2>")  # the start position of entity2  如18
        e22_p = tokens_b.index("</e2>")  # the end position of entity2   如20

        # Replace the token
        tokens_a[e11_p] = "$"  # 将文本中的 <e1> 换成 $ 对应下标[109]
        tokens_a[e12_p] = "$"
        tokens_b[e21_p] = "#"  # 将 <e2> 换成 # 对应下标[108]
        tokens_b[e22_p] = "#"

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1
        # Add 2 because of the [CLS] and [SEP]
        e21_p += 2 + len(tokens_a)
        e22_p += 2 + len(tokens_a)

        if len(tokens_a) + len(tokens_b) > (max_length - 3):  # 如果两个文本的长度 比 最大长度-3(即[CLS]句子1[SEP]句子2[SEP]三个特殊标签)，就要截断输入文本。
            _truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        text_len = len(tokens)
        # list.index(x[, start[, end]])
        attention_tokens = tokens[:int(tokens.index('#', int(tokens.index('#') + 1))) + 1]  # 需要子注意力的文本长度 为 到"引入kg子图的位置</e2>"(第二个'#'的位置)结束
        # print('掩盖子图前的文本tokens', tokens)
        # print('/n')
        # print('掩盖子图后的文本tokens', attention_tokens)
        attention_text_len = len(attention_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens + ["[PAD]"] * (max_length - text_len))  # 调用库工具将分词转为分词所在词表上的id
        attention_mask = [1] * attention_text_len + [0] * (max_length - attention_text_len)  # attention_mask文本长度的token都要参加注意力的计算
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) + [0] * (max_length - text_len)  # token_type_ids表示文本1和文本2是不同的句子段落

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)  # 是一个列表，长度就是max_seq_len,对应实体片段的几个token值为1，其他位置为0
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_length, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_length, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        return tokens, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask

    def convert_text_to_ids_for_matching_kg_enhance_2(text_a, text_b):
        """
        对于使用了实体词条<e1></e1> <e2></e2> <kg></kg>的数据集，先把这些词条转为$和#和¥，然后转成id的列表，并记录特殊token的位置
        如果有多个<kg>对的怎么池化处理 先把每个kg对平均池化，然后每个kg对的一维kg表示再平均池化？
        如果没有<kg></kg>对的怎么处理 他的kg_hidden表示？给一个全0表示
        如果有多个<kg></kg>对的怎么表示 kg对内的n_gram，每个token平均
        :param text_a:
        :param text_b:
        :return:
        """
        tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        tokens_a = tokenizer.tokenize(text_a)  # 对文本进行分词
        tokens_b = tokenizer.tokenize(text_b)  # 对文本进行分词

        e11_p = tokens_a.index("<e1>")  # the start position of entity1  得到文本中<e1>这个特殊token的位置，比如13
        e12_p = tokens_a.index("</e1>")  # the end position of entity1   如15
        e21_p = tokens_b.index("<e2>")  # the start position of entity2  如18
        e22_p = tokens_b.index("</e2>")  # the end position of entity2   如20

        # Replace the token
        tokens_a[e11_p] = "$"  # 将文本中的 <e1> 换成 $ 对应下标[109]
        tokens_a[e12_p] = "$"
        tokens_b[e21_p] = "#"  # 将 <e2> 换成 # 对应下标[108]
        tokens_b[e22_p] = "#"

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1

        # Add 2 because of the [CLS] and [SEP]
        e21_p += 2 + len(tokens_a)
        e22_p += 2 + len(tokens_a)

        # 对句子1与知识库的重叠词的处理
        k1_p_list = []  # 记录<kg>的位置
        k2_p_list = []  # 记录</kg>的位置

        for idx, i in enumerate(tokens_a):
            if i == '<kg>':
                k1_p_list.append(idx)
        print(k1_p_list)

        for idx, i in enumerate(tokens_a):
            if i == '</kg>':
                k2_p_list.append(idx)
        print(k2_p_list)

        p_list = k1_p_list + k2_p_list
        print(tokens_a)
        for iii in p_list:
            tokens_a[iii] = "¥"
        print(tokens_a)

        # 7-10 和 11-14 全部变为1
        assert len(k1_p_list) == len(k2_p_list)

        if len(tokens_a) + len(tokens_b) > (max_length - 3):  # 如果两个文本的长度 比 最大长度-3(即[CLS]句子1[SEP]句子2[SEP]三个特殊标签)，就要截断输入文本。
            _truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        text_len = len(tokens)
        # 调用分词工具转为词表id
        input_ids = tokenizer.convert_tokens_to_ids(tokens + ["[PAD]"] * (max_length - text_len))
        attention_mask = [1] * text_len + [0] * (max_length - text_len)  # attention_mask文本长度的token都要参加注意力的计算
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) + [0] * (max_length - text_len)  # token_type_ids表示文本1和文本2是不同的句子段落

        # e1 mask, e2 mask
        # 全初始化为0
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        # 是一个列表，长度就是max_seq_len,对应实体片段的几个token值为1，其他位置为0
        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        # k_mask
        # 初始化为0
        k_mask = [0] * len(attention_mask)
        # k1_p   the start position of kg pairs
        # k2_p   the end position of kg pairs
        for k1_p, k2_p in zip(k1_p_list, k2_p_list):  # 每一对<kg></kg>对的位置
            for ii in range(k1_p + 1, k2_p + 2):  # 这里是因为k_mask是在token_a前面加了[CLS]
                k_mask[ii] = 1

        # print('input_ids', input_ids)
        # print('attention_mask', attention_mask)
        # print('token_type_ids', token_type_ids)
        # print('e1_mask', e1_mask)
        # print('e2_mask', e2_mask)
        # print('k_mask', k_mask)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_length, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_length, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        return tokens, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, k_mask

    label_map = {label: i for i, label in enumerate(label_list)}  # 字典，键是标签，值是索引
    features = []

    # 对每一个sample进行如下的操作
    for i in range(len(examples)):
        # for i in range(20):
        # 传入的文本数据，有无有句子2 text_b
        # 加入边界的 句子1 句子2 的分类
        if examples[i].text_b and ccks_no_1:  # 使用了mention的开始和结束位置
            try:
                tokens1, input_ids1, attention_mask1, token_type_ids1, e_begin_mask, e_end_mask \
                    = convert_text_to_ids_for_matching_ccks_no_1(examples[i].text_a, examples[i].text_b, examples[i].pos)
            except ValueError:
                print(examples[i].text_a)
        elif examples[i].text_b and kg_enhance_2:  # 使用了特殊token<e1><e2><kg>的情况
            try:
                tokens1, input_ids1, attention_mask1, token_type_ids1, e1_mask, e2_mask, k_mask \
                    = convert_text_to_ids_for_matching_kg_enhance_2(examples[i].text_a, examples[i].text_b)
            except ValueError:
                print(examples[i].text_a)
        elif examples[i].text_b and kg_enhance_3:  # 使用了特殊token<e1><e2>且关闭子图自注意力的情况，防止kn
            try:
                tokens1, input_ids1, attention_mask1, token_type_ids1, e1_mask, e2_mask \
                    = convert_text_to_ids_for_matching_kg_enhance_3(examples[i].text_a, examples[i].text_b)
            except ValueError:
                print(examples[i].text_a)
        elif examples[i].text_b and kg_enhance:  # 使用了特殊token<e1><e2>的情况
            try:
                tokens1, input_ids1, attention_mask1, token_type_ids1, e1_mask, e2_mask \
                    = convert_text_to_ids_for_matching_kg_enhance(examples[i].text_a, examples[i].text_b)
            except ValueError:
                print(examples[i].text_a)
        elif examples[i].text_b:  # 普通情况 句子1 句子2 的分类
            tokens1, input_ids1, attention_mask1, token_type_ids1 \
                = convert_text_to_ids_for_matching(examples[i].text_a, examples[i].text_b)
        else:  # 普通情况 句子1 的分类
            tokens1, input_ids1, attention_mask1, token_type_ids1 = convert_text_to_ids(examples[i].text_a)
            # print(tokens1,input_ids1,attention_mask1,token_type_ids1)

        # label_map得到每个任务的标签下标
        if output_mode == "ner":
            label_id = [label_map["O"]]  # 一头插入0，特殊符号['O']
            for j in range(len(tokens1) - 2):
                label_id.append(label_map[examples[i].label[j]])  # 取出其余token标签的id
            label_id.append(label_map["O"])  # 一尾插入0，特殊符号['O']
            if len(label_id) < max_length:
                label_id = label_id + [label_map["O"]] * (max_length - len(label_id))  # 剩余的位置都是插入0，表示[PAD]
        else:
            # print('examples[i]', examples[i])
            label_id = label_map[examples[i].label]  # 其余的直接读出预处理后的tsv文件获得案例的标签，从label_map中取出标签的id

        # 通过InputFeatures类将每一个sample的输出输入组装到特征列表，输入：tokens1，input_ids1，attention_mask1，token_type_ids1，输出：label_id
        if ccks_no_1:
            feature = InputFeatures(
                tokens1=tokens1,
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                token_type_ids1=token_type_ids1,
                label_id=label_id,
                e_begin_mask=e_begin_mask,
                e_end_mask=e_end_mask,
            )
        elif kg_enhance_2:
            feature = InputFeatures(
                tokens1=tokens1,
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                token_type_ids1=token_type_ids1,
                label_id=label_id,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
                k_mask=k_mask
            )
        elif kg_enhance_3:
            feature = InputFeatures(
                tokens1=tokens1,
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                token_type_ids1=token_type_ids1,
                label_id=label_id,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
            )
        elif kg_enhance:
            feature = InputFeatures(
                tokens1=tokens1,
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                token_type_ids1=token_type_ids1,
                label_id=label_id,
                e1_mask=e1_mask,
                e2_mask=e2_mask,
            )
        else:
            feature = InputFeatures(
                tokens1=tokens1,
                input_ids1=input_ids1,
                attention_mask1=attention_mask1,
                token_type_ids1=token_type_ids1,
                label_id=label_id,
            )
        # 将每一个sample转成的feature对象，添加到对象features列表中
        features.append(feature)

        # 在日志中输出第一个例子
        if i < 1 and ccks_no_1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % examples[i].guid)
            logger.info("label_id: %s" % label_id)
            logger.info("tokens1: %s" % " ".join(tokens1))
            logger.info("input_ids1: %s" % " ".join([str(x) for x in input_ids1]))  # 将ids变为字符串然后 每个元素用空格隔开
            logger.info("attention_mask1: %s" % " ".join([str(x) for x in attention_mask1]))
            logger.info("token_type_ids1: %s" % " ".join([str(x) for x in token_type_ids1]))
            logger.info("e_begin_mask: %s" % e_begin_mask)
            logger.info("e_end_mask: %s" % e_end_mask)
        elif i < 1 and kg_enhance_2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % examples[i].guid)
            logger.info("label_id: %s" % label_id)
            logger.info("tokens1: %s" % " ".join(tokens1))
            logger.info("input_ids1: %s" % " ".join([str(x) for x in input_ids1]))  # 将ids变为字符串然后 每个元素用空格隔开
            logger.info("attention_mask1: %s" % " ".join([str(x) for x in attention_mask1]))
            logger.info("token_type_ids1: %s" % " ".join([str(x) for x in token_type_ids1]))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
            logger.info("k_mask :%s" % " ".join([str(x) for x in k_mask]))
        elif i < 1 and kg_enhance_3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % examples[i].guid)
            logger.info("label_id: %s" % label_id)
            logger.info("tokens1: %s" % " ".join(tokens1))
            logger.info("input_ids1: %s" % " ".join([str(x) for x in input_ids1]))  # 将ids变为字符串然后 每个元素用空格隔开
            logger.info("attention_mask1: %s" % " ".join([str(x) for x in attention_mask1]))
            logger.info("token_type_ids1: %s" % " ".join([str(x) for x in token_type_ids1]))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
        elif i < 1 and kg_enhance:
            logger.info("*** Example ***")
            logger.info("guid: %s" % examples[i].guid)
            logger.info("label_id: %s" % label_id)
            logger.info("tokens1: %s" % " ".join(tokens1))
            logger.info("input_ids1: %s" % " ".join([str(x) for x in input_ids1]))  # 将ids变为字符串然后 每个元素用空格隔开
            logger.info("attention_mask1: %s" % " ".join([str(x) for x in attention_mask1]))
            logger.info("token_type_ids1: %s" % " ".join([str(x) for x in token_type_ids1]))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
        elif i < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % examples[i].guid)
            logger.info("label_id: %s" % label_id)
            logger.info("tokens1: %s" % " ".join(tokens1))
            logger.info("input_ids1: %s" % " ".join([str(x) for x in input_ids1]))  # 将ids变为字符串然后 每个元素用空格隔开
            logger.info("attention_mask1: %s" % " ".join([str(x) for x in attention_mask1]))
            logger.info("token_type_ids1: %s" % " ".join([str(x) for x in token_type_ids1]))

    return features


# 从指定目录拿到转为张量的样本数据
def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = processors[task]()  # processor获得一个类名，因此这个类名值的类。比如 入参tnews→task，processors[task]→TNewsProcessor，这时processor指向 TNewsProcessor()
    output_mode = output_modes[task]  # 同上，但获得是一个值，如传入tnews，得到为classification
    logger.info("Creating features from dataset file at %{}".format(args.data_dir))
    label_list = processor.get_labels(args.data_dir)  # 从processor指向的类，如TNewsProcessor()中调用get_labels函数
    if evaluate:
        examples = (
            processor.get_test_examples(args.data_dir, args.input_test_name)
        )
    else:
        examples = (
            processor.get_train_examples(args.data_dir)
        )
    if args.is_debug:
        examples = examples[:100]
    if task == "ekbertmd":
        # 指定kb
        # spo_files = "nlpcc_test+nlpcc_train".split('+')
        spo_files = args.kg_name.split('+')
        kg = Knowledge(spo_files=spo_files, min_span=args.min_span_length)
        # print('注入边界前，前两条案例{}'.format(examples[:2]))
        for e in examples:
            # loggerer.info('注入前,{}-{}'.format(e.text_a, e.label))
            text_a, label = kg.infuse_entities_border(e.text_a, ' '.join(e.label))
            e.text_a = text_a
            e.label = label.split(' ')
            # logger.info('注入后,{}-{}'.format(e.text_a, e.label))
        logger.info('注入边界后，前5条案例{}'.format(examples[:5]))

    # 拿到所有原始的文本，用分词器，转成索引
    features = convert_examples_to_features(
        examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode, kg_enhance=args.kg_enhance, kg_enhance_2=args.kg_enhance_2, kg_enhance_3=args.kg_enhance_3,
        ccks_no_1=args.ccks_no_1)
    # print(features)  # add by hzp

    # Convert ids to Tensors and build dataset

    # 将token id的数值列表转成torch中的tensor类型
    if args.ccks_no_1:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
        all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_e_begin_mask = torch.tensor([f.e_begin_mask for f in features], dtype=torch.long)
        all_e_end_mask = torch.tensor([f.e_end_mask for f in features], dtype=torch.long)
        # 将各个tensor特征包裹到数据迭代器中
        dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels, all_e_begin_mask, all_e_end_mask)
    elif args.kg_enhance_2:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
        all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
        all_k_mask = torch.tensor([f.k_mask for f in features], dtype=torch.long)  # add kg mask
        # 将各个tensor特征包裹到数据迭代器中
        dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels, all_e1_mask, all_e2_mask, all_k_mask)
    elif args.kg_enhance_3:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
        all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
        # 将各个tensor特征包裹到数据迭代器中
        dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels, all_e1_mask, all_e2_mask)
    elif args.kg_enhance:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
        all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
        # 将各个tensor特征包裹到数据迭代器中
        dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels, all_e1_mask, all_e2_mask)
    else:
        all_input_ids1 = torch.tensor([f.input_ids1 for f in features], dtype=torch.long)
        all_attention_mask1 = torch.tensor([f.attention_mask1 for f in features], dtype=torch.long)
        all_token_type_ids1 = torch.tensor([f.token_type_ids1 for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)
        # 将各个tensor特征包裹到数据迭代器中
        dataset = TensorDataset(all_input_ids1, all_attention_mask1, all_token_type_ids1, all_labels)

    return dataset, examples


try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


# 计算acc,返回字典，评估指标名：评估分值
def compute_metrics(preds, labels):
    return {"acc": (preds == labels).mean()}


# 对于ner 使用token级别的精度acc，即句子中中50个token,正确预测了40个，该句子的精度是0.8
# 返回字典，评估指标名：评估分值
def ner_compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    # print(len(preds))  # test
    accs = []  # [1.0, 0.8, 1.0, 1.0, 0.98, 1.0 ....]
    for i in range(len(preds)):  # 句子维度的精确率,比如100个句子
        sent_acc = (preds[i] == labels[i]).mean()  # =，==, .fun的优先级 ==优先于=，然后到函数，再到赋值
        accs.append(sent_acc)  # 相等+1,否则+0

    return {"acc": np.array(accs).mean()}


# 不适用于ner任务，仅适用于序列批注任务。需要重写
def ner_F1(preds, labels, mask_indicators):
    assert len(preds) == len(labels) == len(mask_indicators)  # 预测标签数 = 标签数 = 注意力的token数
    # print(preds.shape)
    # print(labels.shape)
    # print(preds[0])
    # print(labels[0])
    total_preds = []
    total_ground = []
    test_l = len(preds)  # test
    test_r = range(len(preds))  # test
    for i in range(len(preds)):
        num = sum(mask_indicators[i]) - 2  # sum():mask=1的token数,mask=0的未被何求和，然后减去一头一尾两个标签
        total_preds.extend(preds[i][1: 1 + num])  # 减去一头一尾的mask=1的token 预测值列表
        total_ground.extend(labels[i][1: 1 + num])  # 减去一头一尾的mask=1的token 真值列表

    refer_label = total_ground  # token标签列表
    pred_label = total_preds  # token预测列表
    fn = dict()  # 正类预测成负类
    tp = dict()  # 正类预测成正类
    fp = dict()  # 负类预测成正类
    for i in range(len(refer_label)):
        if refer_label[i] == pred_label[i]:  # token预测正确
            if refer_label[i] not in tp:  # 字典没有该预测值的键，则新建键值对，值用于统计该预测值被预测正确的次数
                tp[refer_label[i]] = 0
            tp[refer_label[i]] += 1  # 比如 token的label==pred==7,{7:n(n为预测正确次数)
        else:  # 预测值 不等于 标签值
            if pred_label[i] not in fp:
                fp[pred_label[i]] = 0  # 预测值 不在fp字典，则创建一项键值对
            fp[pred_label[i]] += 1  # 累计该预测值的预测错误次数，比如预测为7，但实际标签不是7，{7:n(错误预测为7的次数)}
            if refer_label[i] not in fn:
                fn[refer_label[i]] = 0  # 标签纸 不在fp字典，则创建一项键值对
            fn[refer_label[i]] += 1  # 累计该标签纸 未被召回的 次数，比如标签为8，但未预测为8，{8:n(标签为8但预测错的次数)}
    tp_total = sum(tp.values())  # 对字典所有键值对的值求和
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())
    # 精确率（查准率） = 正类预测为正类/正类预测为正类+负类预测为正类（针对预测结果）
    p_total = float(tp_total) / (tp_total + fp_total)
    # 召回率（查全率） = 正类预测为正类/正类预测为正类+正类预测为负类（针对原来样本）
    r_total = float(tp_total) / (tp_total + fn_total)
    # 查准和查全的调和平均
    f_micro = 2 * p_total * r_total / (p_total + r_total)  # f1值的公式 等于 精确率p和召回率r的调和平均
    return {"f1_score": f_micro, "precision": p_total, "recall": r_total}


def ner_f1_new(preds, labels, masks, begin_ids: List, notentity_ids: List):
    """
    :param preds:  所有案例的序列预测值 ndarray:(examples_size x seq_len)
    :param labels: 所有案例的序列标签值 ndarray:(examples_size x seq_len)
    :param masks:一个批次的masktoken ndarray:(batch_size x seq_len)
    :param begin_ids: 表示实体开始的token标签的下标。比如标签'B'或者'B-entity'或者'S'的下标
    :param notentity_ids:表示非实体的token标签的下标。标签'O'的下标
    :return:
    """
    assert len(preds) == len(labels) == len(masks)  # 预测标签数 = 标签数 = 注意力的token数
    # print("预测值形状" + str(preds.shape))
    # print("真值形状" + str(labels.shape))
    # print("遮盖形状" + str(masks.shape))
    # print("第一个样本的预测值" + str(preds[0]))
    # print("第一个样本的真值" + str(labels[0]))
    total_preds = []
    total_ground = []
    # print("预测样本数" + str(len(preds)))
    test_r = range(len(preds))  # test
    # 将所有样本需要关注的token取出来，每个样本除去特殊标签[CLS][SEP][PAD] 得到列表total_preds，total_ground
    for i in range(len(preds)):
        num = sum(masks[i]) - 2  # sum():需要关注的token数，对mask=1的求和，减去一头一尾两个token
        total_preds.extend(preds[i][1: 1 + num])  # 减去一头一尾的mask=1的token 预测值列表
        total_ground.extend(labels[i][1: 1 + num])  # 减去一头一尾的mask=1的token 真值列表

    refer_label = total_ground  # token标签列表
    pred_label = total_preds  # token预测列表
    gold_entities_num = 0
    pred_entities_num = 0
    correct = 0
    # 统计真实的实体数量
    for i in range(len(refer_label)):
        if refer_label[i] in begin_ids:
            gold_entities_num += 1
    # 统计预测的实体数量
    for j in range(len(pred_label)):
        if pred_label[j] in begin_ids:
            pred_entities_num += 1

    pred_entities_pos = []
    gold_entities_pos = []
    start, end = 0, 0

    for j in range(len(refer_label)):
        if refer_label[j] in begin_ids:
            start = j
            for k in range(j + 1, len(refer_label)):
                # 如果遇到'O'或者'B''S',则有一个新的实体，k-1表示当前实体的结束位置
                if refer_label[k] in notentity_ids or refer_label[k] in begin_ids:
                    end = k - 1
                    break
                else:
                    end = len(refer_label) - 1
            gold_entities_pos.append((start, end))  # 通过始末位置表示 金实体

    for j in range(len(pred_label)):
        if pred_label[j] in begin_ids:
            start = j
            for k in range(j + 1, len(pred_label)):
                if pred_label[k] in notentity_ids or pred_label[k] in begin_ids:
                    end = k - 1
                    break
                else:
                    end = len(pred_label) - 1
            pred_entities_pos.append((start, end))  # 通过始末位置表示 金实体

    for entity in pred_entities_pos:
        if entity not in gold_entities_pos:
            continue
        else:
            correct += 1

    # 精确率（查准率） = 正类预测为正类/正类预测为正类+负类预测为正类（针对预测结果）
    if pred_entities_num != 0:
        p_total = correct / pred_entities_num
    else:
        p_total = 0
    # 召回率（查全率） = 正类预测为正类/正类预测为正类+正类预测为负类（针对原来样本）
    r_total = correct / gold_entities_num
    # 查准和查全的调和平均
    if p_total == 0 and r_total == 0:
        f_micro = 0
    else:
        f_micro = 2 * p_total * r_total / (p_total + r_total)  # f1值的公式 等于 精确率p和召回率r的调和平均
    return {"f1_score": f_micro, "precision": p_total, "recall": r_total}


if _has_sklearn:
    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
