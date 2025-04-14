# -*- encoding:utf-8 -*-
"""
  This script provides an K-BERT example for NER.
"""
import random
import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import BertAdam
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
import numpy as np
from copy import deepcopy
from typing import Dict, List, Any
import os
import time
from loguru import logger
from brain import KnowledgeGraph
from args import args
import json
import sys

# 当前文件的上一级
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')  # 文件所有目录的上一级 即kbert的上一级src
from utils import get_entities, get_entity_bio


class BertTagger(nn.Module):
    def __init__(self, args, model):
        super(BertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)  # 全链接层
        self.softmax = nn.LogSoftmax(dim=-1)  # 最后一维度 每个元组的概率值累加=1 logsoftmax即在进行softmax运算后取log的值

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        output = self.encoder(emb, mask, vm)
        # Target.
        output = self.output_layer(output)  # output_layer 全连接层

        output = output.contiguous().view(-1, self.labels_num)  # 调用view方法前要确保tensor是连续的
        output = self.softmax(output)

        label = label.contiguous().view(-1, 1)  # tensor torch.Size([3072, 1])
        label_mask = (label > 0).float().to(torch.device(label.device))
        one_hot = torch.zeros(label_mask.size(0), self.labels_num).to(torch.device(label.device)).scatter_(1, label, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        predict = output.argmax(dim=-1)  # tensor
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )

        return loss, correct, predict, label


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
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


def dump_jsonl(data_list: List[Any], path: str):
    with open(path, "w") as f:
        for json_obj in data_list:
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


def outputs_eval_results(results: Dict, args, output_eval_file: str):
    """
    把评估结果写到文件
    :param results: 评估结果
    :param args:
    :param output_eval_file:
    :return:
    """
    with open(output_eval_file, "a+") as writer:  # 追加 ，不存在则创建
        logger.info("***** Eval results *****")
        writer.write("评估任务(task_name) = %s\n" % "nlpccmd")
        if args.is_test:
            writer.write("评估数据集 = %s\n" % "test")
            writer.write("数据集路径(data_dir) = %s\n" % args.test_path)
        else:
            writer.write("评估数据集 = %s\n" % "dev")
            writer.write("数据集路径(data_dir) = %s\n" % args.dev_path)
        for key in sorted(results.keys()):
            logger.info("{}={}".format(key, str(results[key])))
            writer.write("%s = %s\n" % (key, str(results[key])))
        writer.write("eval_time = %s\n" % time.asctime(time.localtime(time.time())))  # 执行时间 写入文档
    writer.close()


def main(args):
    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    labels_map = {"[PAD]": 0, "[ENT]": 1}
    # 作为实体开始标签的下标，比如labals_map = {'[PAD]': 0, '[ENT]': 1, 'O': 2, 'B-entity': 3, 'I-entity': 4},这里的 begin_ids = [3]
    begin_ids = []

    # Find tagging labels
    # 从训练集数据中获取标签映射
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            labels = line.strip().split("\t")[1].split()  # split() 无参数默认以空格作为分隔得到一个标签列表
            for l in labels:  # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O']
                if l not in labels_map:
                    if l.startswith("B") or l.startswith("S"):
                        begin_ids.append(len(labels_map))  # B或者S开头都是实体的开始标签 [3, 5, 7]
                    labels_map[l] = len(labels_map)  # 将 标签:下标 添加到字典

    # print("Labels: ", labels_map)  # {'[PAD]': 0, '[ENT]': 1, 'O': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-PER': 5, 'I-PER': 6, 'B-ORG': 7, 'I-ORG': 8}
    args.labels_num = len(labels_map)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        # pen
        # spo_files = [args.kg_name]
        spo_files = list(args.kg_name.split('+'))
    # 初始化kg，不带关系
    kg = KnowledgeGraph(spo_files=spo_files, predicate=False, min_span=args.min_span)

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build sequence labeling model.
    model = BertTagger(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size, :]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vm_ids_batch = vm_ids[i * batch_size: (i + 1) * batch_size, :, :]
            tag_ids_batch = tag_ids[i * batch_size: (i + 1) * batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:, :]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
            vm_ids_batch = vm_ids[instances_num // batch_size * batch_size:, :, :]
            tag_ids_batch = tag_ids[instances_num // batch_size * batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch

    # Read dataset.
    def read_dataset(path):
        dataset = []
        match_entities_dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                # 调试部分
                if args.is_debug and line_id == 100:
                    break
                tokens, labels = line.strip().split("\t")
                text = ''.join(tokens.split(" "))  # '海钓比赛地点在厦门与金门之间的海域。'
                tokens, pos, vm, tag, match_entities = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length, model_type=args.model_type)

                # test
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")
                tag = tag[0]
                tokens = [vocab.get(t) for t in tokens]
                labels = [labels_map[l] for l in labels.split(" ")]
                mask = [1] * len(tokens)

                # 对dataset中的label做了处理，比如加入的知识token用的是ENT标签代替
                new_labels = []
                j = 0
                for i in range(len(tokens)):
                    if tag[i] == 0 and tokens[i] != PAD_ID:
                        new_labels.append(labels[j])
                        j += 1
                    elif tag[i] == 1 and tokens[i] != PAD_ID:  # 是添加的实体
                        new_labels.append(labels_map['[ENT]'])  # ENT标识为实体的特殊token
                    else:
                        new_labels.append(labels_map[PAD_TOKEN])
                # print('labels', labels)
                # print('new_labels', new_labels)
                # print('tokens_txt', tokens_txt)
                # print('tokens', tokens)
                dataset.append([tokens, new_labels, mask, pos, vm, tag])
                match_entities_dataset.append(match_entities)
        # print('一共匹配' + str(match_count) + '个知识头实体')
        # print('前5条样本的tokens和labels为:', [{"tokens": d[0], "labels": d[1], "match_entities": m} for d, m in zip(dataset[:5], match_entities_dataset[:5])])
        return dataset

    # Evaluation function.
    def evaluate(args):
        if args.is_test:
            dataset = read_dataset(args.test_path)  # 测试集
        else:
            dataset = read_dataset(args.dev_path)  # 验证集
        input_ids = torch.LongTensor(np.array([sample[0] for sample in dataset]))
        label_ids = torch.LongTensor(np.array([sample[1] for sample in dataset]))
        mask_ids = torch.LongTensor(np.array([sample[2] for sample in dataset]))
        pos_ids = torch.LongTensor(np.array([sample[3] for sample in dataset]))
        vm_ids = torch.BoolTensor(np.array([sample[4] for sample in dataset]))
        tag_ids = torch.LongTensor(np.array([sample[5] for sample in dataset]))
        instances_num = input_ids.size(0)
        batch_size = args.batch_size
        print("Batch size: ", batch_size)
        print("The number of test instances:", instances_num)
        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0
        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)  # 创建元素为0的标签长度的矩阵
        model.eval()  # model 进入评估状态
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(
                batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)
            loss, _, pred, gold = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)  # model给出预测的结果
            # 得到金实体总数
            for j in range(gold.size()[0]):  # 金标签数，gold.size()[0]:batch_size x seq_len ,eg：12 x 256 = 3072个 token
                if gold[j].item() in begin_ids:  # 有多少个开始标签，就有多少个实体
                    gold_entities_num += 1  # 统计句子中的实体数量
            # 得到预测实体总数
            for j in range(pred.size()[0]):  # 预测标签数，pred.size()[0]:batch_size x seq_len ,eg：12 x 256 = 3072个 token
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"]:  # 预测的标签值 等于 B标签，并且不是[pad]标签，才计入预测实体数
                    pred_entities_num += 1  # 统计模型预测的句子的实体数量
            # 得到表示金实体位置 和 预测实体位置 的列表
            pred_entities_pos = []
            gold_entities_pos = []
            start, end = 0, 0
            for j in range(gold.size()[0]):  # 金标签数，gold.size()[0]:batch_size x seq_len ,eg：12 x 256 = 3072个 token
                if gold[j].item() in begin_ids:
                    start = j  # 实体起点位置
                    for k in range(j + 1, gold.size()[0]):
                        if gold[k].item() == labels_map['[ENT]']:
                            continue
                        if gold[k].item() == labels_map["[PAD]"] or gold[k].item() == labels_map["O"] or gold[k].item() in begin_ids:
                            end = k - 1  # 实体终点位置（1）：遇到pad标签或者遇到O标签或者遇到下一个B标签的位置-1
                            break
                    else:
                        end = gold.size()[0] - 1  # 实体终点位置（2）：(1)都不符合，就到该seq_len的最后一个
                    gold_entities_pos.append((start, end))  # 通过始末位置表示 金实体
                    # print(gold_entities_pos)
            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"] and gold[j].item() != labels_map["[ENT]"]:
                    start = j
                    for k in range(j + 1, pred.size()[0]):
                        if gold[k].item() == labels_map['[ENT]']:
                            continue
                        if pred[k].item() == labels_map["[PAD]"] or pred[k].item() == labels_map["O"] or pred[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = pred.size()[0] - 1
                    pred_entities_pos.append((start, end))  # 通过始末位置表示 预测实体
            # print('pred_entities_pos:', pred_entities_pos)
            # print('gold_entities_pos: ', gold_entities_pos)
            for entity in pred_entities_pos:  # 模型预测的实体的位置
                if entity not in gold_entities_pos:  # 如果模型预测的实体的位置和标签的实体位置不一，就下一个
                    continue
                else:
                    correct += 1  # 模型预测的实体位置 在 金标签的实体位置 列表中
        print("Report precision, recall, and f1:")
        try:
            p = correct / pred_entities_num
        except ZeroDivisionError:
            p = 0
        try:
            r = correct / gold_entities_num
        except ZeroDivisionError:
            r = 0
        try:
            f1 = 2 * p * r / (p + r)
        except ZeroDivisionError:
            f1 = 0
        print("{:.3f}, {:.3f}, {:.3f}".format(p, r, f1))
        # 执行结果 写入文档
        file_path_2 = os.path.abspath(os.path.join(args.output_model_path, ".."))
        output_eval_file = os.path.join(file_path_2, "eval_results.txt")  # 输出评估结果到目录
        results = {"f1_score": f1, "precision": p, "recall": r}
        outputs_eval_results(results, args, output_eval_file)
        return f1

    # Preict function.
    def predict(args):
        if args.is_test:
            dataset = read_dataset(args.test_path)  # 测试集
        else:
            dataset = read_dataset(args.dev_path)  # 验证集
        input_ids = torch.LongTensor(np.array([sample[0] for sample in dataset]))
        label_ids = torch.LongTensor(np.array([sample[1] for sample in dataset]))
        mask_ids = torch.LongTensor(np.array([sample[2] for sample in dataset]))
        pos_ids = torch.LongTensor(np.array([sample[3] for sample in dataset]))
        vm_ids = torch.BoolTensor(np.array([sample[4] for sample in dataset]))
        tag_ids = torch.LongTensor(np.array([sample[5] for sample in dataset]))
        instances_num = input_ids.size(0)
        batch_size = args.batch_size
        print("Batch size: ", batch_size)
        print("The number of test instances:", instances_num)
        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0
        model.eval()  # model 进入评估状态
        gold_token_total = []
        pred_token_total = []
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(
                batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)
            loss, _, pred, gold = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)  # model给出预测的结果
            # print(gold.size(), '\n', gold)
            # print(pred.size(), '\n', pred)
            # 将预测和金标签转结果输出json
            # tensor to list 并加到所有的列表
            gold_token_total += gold.cpu().numpy().tolist()
            pred_token_total += pred.cpu().numpy().tolist()
        example_gold = [gold_token_total[i:i + args.seq_length] for i in range(0, len(gold_token_total), args.seq_length)]
        example_pred = [pred_token_total[i:i + args.seq_length] for i in range(0, len(pred_token_total), args.seq_length)]
        labels_map_r = {idx: label for label, idx in labels_map.items()}
        print(labels_map)
        print(labels_map_r)
        predict_result = []
        # 得到表示金实体位置 和 预测实体位置 的列表
        for idx, (pred, gold) in enumerate(zip(example_pred, example_gold)):
            gold_without_pad = []
            pred_without_pad = []
            for gold_token, pred_token in zip(gold, pred):
                # 当金标签的token是PAD或者ENT，则跳过这些位置的token。
                if gold_token == labels_map['[PAD]'] or gold_token == labels_map['[ENT]']:
                    continue
                else:
                    gold_without_pad.append(gold_token)
                    pred_without_pad.append(pred_token)  # 加入同样位置的预测token
            assert len(pred_without_pad) == len(gold_without_pad)
            pred_tags = [labels_map_r[t] for t in pred_without_pad]
            gold_tags = [labels_map_r[t] for t in gold_without_pad]
            pred_entities = get_entities(pred_without_pad, labels_map_r, 'bio')
            gold_entities = get_entities(gold_without_pad, labels_map_r, 'bio')
            # 预测正确标志
            pred_poss = [(i[1], i[2]) for i in pred_entities]
            gold_poss = [(i[1], i[2]) for i in gold_entities]
            hit_flag = []  # 长度等于金实体数量 0925
            for i in gold_poss:
                if i in pred_poss:
                    hit_flag.append(True)
                else:
                    hit_flag.append(False)
            json_d = {'id': idx,
                      'pred_entities': pred_entities,
                      'gold_entities': gold_entities,
                      'pred_tags': " ".join(pred_tags),
                      'gold_tags': " ".join(gold_tags),
                      'hit_flag': hit_flag
                      }
            predict_result.append(json_d)
        # 写出预测结果，json格式
        dump_path = args.output_predict_result_path
        print(dump_path)
        dump_jsonl(predict_result, dump_path)

    if args.do_train:
        # Training phase. 训练截断
        print("Start training.")
        instances = read_dataset(args.train_path)  # 读出处理好的dataset instance:[tokens, new_labels, mask, pos, vm, tag]
        # instances = instances[:50]  # 只读取50个例子训练 调试程序
        input_ids = torch.LongTensor(np.array([ins[0] for ins in instances]))  # [example x seq_length]
        label_ids = torch.LongTensor(np.array([ins[1] for ins in instances]))  # 对于引入的知识实体token，记特殊标签[ENT]
        mask_ids = torch.LongTensor(np.array([ins[2] for ins in instances]))
        pos_ids = torch.LongTensor(np.array([ins[3] for ins in instances]))
        vm_ids = torch.BoolTensor(np.array([ins[4] for ins in instances]))
        tag_ids = torch.LongTensor(np.array([ins[5] for ins in instances]))
        instances_num = input_ids.size(0)
        batch_size = args.batch_size
        train_steps = int(instances_num * args.epochs_num / batch_size) + 1
        print("Batch size: ", batch_size)
        print("The number of training instances:", instances_num)
        # 设置不同的优化器 和 权重衰减
        param_optimizer = list(model.named_parameters())  # 有209个参数名 比如word_embedding.weight,embedding.layer_norm.gamma,embedding.layer_norm.beta,
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)
        total_loss = 0.
        f1 = 0.0
        best_f1 = 0.0
        for epoch in range(1, args.epochs_num + 1):
            model.train()  # 模型进入训练模式
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(
                    batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):
                model.zero_grad()  # 清空网络中所有学习参数的梯度，因为学习参数的tensor中的属性grad是累积的
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                tag_ids_batch = tag_ids_batch.to(device)
                vm_ids_batch = vm_ids_batch.long().to(device)
                loss, _, _, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)
                if torch.cuda.device_count() > 1:
                    loss = torch.mean(loss)  # 如果多台机器跑，求平均值
                total_loss += loss.item()  # 每一批次的数据反向计算一次梯度，得出一次loss,对所有批次的loss值求和,直到report_steps步，平均求得每一步的loss
                if (i + 1) % args.report_steps == 0:  # 每 report_steps 步报告一次loss
                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                    total_loss = 0.  # 清空这 report_steps 步的loss,开始下一轮report_steps步
                loss.backward()  # 反向计算 网络中所有学习参数的梯度
                optimizer.step()  # 按设定的优化器，根据lr以及梯度，执行一次 网络所有学习参数的更新
            # Evaluation phase. 评估阶段
            print("Start evaluate on dev dataset.")
            f1 = evaluate(args)  # False 表示在dev集上训练  每一训练轮次在dev上计算一起f1值，最终保存在dev上表现最好的模型
            if f1 >= best_f1:  # 如果f1值最高 则保存该模型
                best_f1 = f1
                save_model(model, args.output_model_path)
            else:
                continue

    # Evaluation phase.
    if args.do_eval:
        print("Final evaluation on the test dataset.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        elif torch.cuda.device_count() == 1:
            model.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path, map_location=torch.device('cpu')))
        evaluate(args)

    # Predicting phase
    if args.do_predict:
        print("Final prediction on the test dataset.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        elif torch.cuda.device_count() == 1:
            model.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path, map_location=torch.device('cpu')))
        predict(args)


if __name__ == "__main__":
    main(args)
