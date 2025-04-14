#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : run_kbert_cls.py
# Date    : 2022-08-22
# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
import sys
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import *
from uer.model_builder import build_model
from uer.utils.optimizers import BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from brain import KnowledgeGraph
from multiprocessing import Process, Pool
import numpy as np
import traceback
import pandas as pd
from pandas import DataFrame
import os
from typing import Dict
from loguru import logger
import time
from args import args
import re
import csv

# 当前文件的上一级
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')  # 文件所有目录的上一级 即kbert的上一级src
from utils import csv_reader_static


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


def add_knowledge_worker(params):
    p_id, sentences, columns, kg, vocab, args = params

    sentences_num = len(sentences)
    dataset = []
    match_entities_dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 10000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        try:
            if len(line) == 2:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]]

                tokens, pos, vm, _, match_entities = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length, model_type=args.model_type)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                dataset.append((token_ids, label, mask, pos, vm))
                match_entities_dataset.append(match_entities)

            elif len(line) == 3:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _, match_entities = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length, model_type=args.model_type)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm))
                match_entities_dataset.append(match_entities)
            elif len(line) == 4:  # for ccks kbqa 2020
                # example_id	text_a	text_b	label
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _, match_entities = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length, model_type=args.model_type)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1
                dataset.append((token_ids, label, mask, pos, vm))
                match_entities_dataset.append(match_entities)
            elif len(line) == 5:  # for ccksel19 ccksel20
                # example_id	text_a	text_b	label	mention_pos
                mention_pos = line[columns["mention_pos"]]
                kid = mention_pos
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _, match_entities = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length, model_type=args.model_type, kid=kid)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm))
                match_entities_dataset.append(match_entities)
            else:
                pass
        except:
            print("Error line: ", line, traceback.format_exc())
    # print('前5条样本的tokens和labels为:', [{"tokens": d[0], "labels": d[1], "match_entities": m} for d, m in zip(dataset[:5], match_entities_dataset[:5])])
    return dataset


def outputs_el_results(results: Dict, args, output_eval_file: str):
    """
    把评估结果写到文件
    :param results: 评估结果
    :param args:
    :param output_eval_file:
    :return:
    """
    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results *****")
        writer.write("评估任务(task_name) = 实体链接\n")
        writer.write("评估数据集(input_test_name) = %s\n" % args.test_path)
        writer.write("数据集路径(data_dir) = %s\n" % args.test_path)
        for key in sorted(results.keys()):
            logger.info("%s = %s\n" % (key, str(results[key])))
            writer.write("%s = %s\n" % (key, str(results[key])))
        writer.write("eval_time = %s\n" % time.asctime(time.localtime(time.time())))  # 执行时间 写入文档
    writer.close()


def compute_ranking_acc(df: DataFrame, top_n: int, file_path_2: str) -> Dict:
    """
    返回前几的精确率 Accuracy@N
    :param df: 所有包含二分类softmax得分的案例df
    :param top_n: 计算前几的得分
    :param file_path_2: 评估结果输出目录
    :return { "acc@{}".format(top_n) : 88%}
    """
    # data = {'score': [0.92, 0.67, 0.8, 0.2, 0.4, 0.12, 0.34, 0.8],
    #         'label': [1, 0, 0, 1, 0, 0, 0, 1],
    #         'query': ['玫瑰主要生长在我国那些地区', '玫瑰主要生长在我国那些地区', '玫瑰主要生长在我国那些地区', '傅博是什么刊物的主编', '傅博是什么刊物的主编', 'text_a', 'text_a', 'text_a']}
    # df = pd.DataFrame(data)
    # print(df)
    # 输出同一query下top_n个分值
    grouped = df.groupby('questions').apply(top, n=top_n)  # 用top()方法对query列进行groupby
    # print(grouped)
    # top_n个候选中是否包含gold eg:l_hit->[1,0,1]  表示query1 和query3 包含了gold
    l_hit = grouped['gold'].groupby(grouped['questions']).max().tolist()  # 前top_n个候选包含了标签1的案例，表示命中
    # print('acc@{}'.format(top_n) + ' hit/not-hit(1/0)', l_hit, '索引对应每个query')
    count = 0
    for v in l_hit:
        count += v
    acc = count / len(l_hit)
    print('acc@{},{}/{}={}'.format(top_n, count, len(l_hit), acc))
    # 将df写入tsv
    df = grouped[['pred', 'gold', 'score', 'questions', 'candidates']]
    output_dir = os.path.join(file_path_2, "ranking_error_als_{}.tsv".format(top_n))
    df.to_csv(output_dir, sep='\t', index=False)
    return {"acc@{}".format(top_n): acc}


def top(df, n=2, column='score'):
    """
    返回表格按某一列排序的前n行
    :param df:
    :param n:
    :param column:
    :return:
    # data = {'score': [0.92, 0.67, 0.8, 0.2, 0.4, 0.12, 0.34, 0.8],
    #         'label': [1, 0, 0, 1, 0, 0, 0, 1],
    #         'query': ['玫瑰主要生长在我国那些地区', '玫瑰主要生长在我国那些地区', '玫瑰主要生长在我国那些地区', '傅博是什么刊物的主编', '傅博是什么刊物的主编', 'text_a', 'text_a', 'text_a']}
    # df = pd.DataFrame(data)
    # top(df)
    """
    return df.sort_values(by=column, ascending=False)[:n]


def outputs_eval_results(results: Dict, args, output_eval_file: str):
    """
    把评估结果写到文件
    :param results: 评估结果
    :param args:
    :param output_eval_file:
    :return:
    """
    with open(output_eval_file, "a+") as writer:  # 追加
        logger.info("***** Eval results *****")
        writer.write("评估任务(task_name) = %s\n" % "nlpcced")
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
    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        if torch.cuda.device_count() != 0:
            model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        else:
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location=torch.device('cpu')), strict=False)  # cpu
        # model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  # gpu
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build classification model.
    model = BertClassifier(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vms_batch = vms[i * batch_size: (i + 1) * batch_size]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
            vms_batch = vms[instances_num // batch_size * batch_size:]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        # pen
        # spo_files = [args.kg_name]
        spo_files = list(args.kg_name.split('+'))
    # 初始化kg，带关系
    kg = KnowledgeGraph(spo_files=spo_files, predicate=True)
    logger.info('kg已完成初始化')

    def read_dataset(path, workers_num=1):
        """

        :param path:
        :param workers_num:
        :return:
        """
        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                # 调试部分
                if args.is_debug and line_id == 101:
                    break
                sentences.append(line)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append((i, sentences[i * sentence_per_block: (i + 1) * sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset

    # Evaluation function.
    def evaluate(args, metrics='Acc'):
        if args.is_test:
            dataset = read_dataset(args.test_path, workers_num=args.workers_num)
        else:
            dataset = read_dataset(args.dev_path, workers_num=args.workers_num)

        input_ids = torch.LongTensor(np.array([sample[0] for sample in dataset]))
        label_ids = torch.LongTensor(np.array([sample[1] for sample in dataset]))
        mask_ids = torch.LongTensor(np.array([sample[2] for sample in dataset]))
        pos_ids = torch.LongTensor(np.array([sample[3] for sample in dataset]))
        vms = [example[4] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()

        if not args.mean_reciprocal_rank:
            total_pred_list = []
            total_gold_list = []
            total_score_list = []
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
                # vms_batch = vms_batch.long()
                vms_batch = torch.LongTensor(np.array(vms_batch))
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)
                with torch.no_grad():
                    try:
                        loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                    except:
                        print(input_ids_batch)
                        print(input_ids_batch.size())
                        print(vms_batch)
                        print(vms_batch.size())
                logits = nn.Softmax(dim=1)(logits)
                # print(logits[:, 1])  # 标签为1的可能性
                score = logits[:, 1]
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                # print('input_ids_batch', input_ids_batch)
                # tensor 转 list
                # .cpu() 在gpu环境时要使用，cpu环境去掉.cpu()
                batch_pred_list = pred.cpu().numpy().tolist()
                batch_gold_list = gold.cpu().numpy().tolist()
                batch_score_list = score.cpu().numpy().tolist()
                total_pred_list += batch_pred_list
                total_gold_list += batch_gold_list
                total_score_list += batch_score_list
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                correct += torch.sum(pred == gold).item()

            file_path = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 文件所有目录的上一级 即kbert的上一级src
            # 读取数据标签
            if args.is_test:
                refer_path = args.test_path
            else:
                refer_path = args.dev_path
            print('refer_path', refer_path)
            refer_df = pd.read_csv(refer_path, error_bad_lines=False, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, header=None, skiprows=1)
            print('refer')
            if 'ccks_el_2019' in args.kg_name or 'ccks_el_2020' in args.kg_name:
                if args.is_debug:
                    # 测试部分案例
                    total_mention_list = refer_df[4].values.tolist()[:100]
                    total_question_list = refer_df[1].values.tolist()[:100]
                    total_candidate_list = refer_df[2].values.tolist()[:100]
                    total_question_list = [q + ':' + m for m, q in zip(total_mention_list, total_question_list)]
                else:
                    # 测试所有案例
                    total_mention_list = refer_df[4].values.tolist()
                    total_question_list = refer_df[1].values.tolist()
                    total_candidate_list = refer_df[2].values.tolist()
                    total_question_list = [q + ':' + m for m, q in zip(total_mention_list, total_question_list)]
            elif 'ccks_kbqa_2020' in args.kg_name:
                if args.is_debug:
                    # 测试部分案例
                    total_question_list = refer_df[1].values.tolist()[:100]
                    total_candidate_list = refer_df[2].values.tolist()[:100]
                else:
                    # 测试所有案例
                    total_question_list = refer_df[1].values.tolist()
                    total_candidate_list = refer_df[2].values.tolist()
            else:
                if args.is_debug:
                    # 测试部分案例
                    total_question_list = refer_df[0].values.tolist()[:100]
                    total_candidate_list = refer_df[1].values.tolist()[:100]
                else:
                    # 测试所有案例
                    total_question_list = refer_df[0].values.tolist()
                    total_candidate_list = refer_df[1].values.tolist()
            print('len(total_question_list)', len(total_question_list), total_question_list[:2])
            print('len(total_gold_list)', len(total_gold_list))
            assert len(total_question_list) == len(total_gold_list)
            assert len(total_pred_list) == len(total_gold_list)
            assert len(total_gold_list) == len(total_score_list)
            assert len(total_score_list) == len(total_candidate_list)
            # 'pred''gold''score'是模型推理得到，question,candidates是数据集得到，两者对齐
            data = {'questions': total_question_list, 'candidates': total_candidate_list, 'pred': total_pred_list, 'gold': total_gold_list, 'score': total_score_list}
            df = DataFrame(data)
            file_path_2 = os.path.abspath(os.path.join(args.output_model_path, ".."))
            dump_path = os.path.join(file_path_2, 'kbert_eval_result.csv')
            # print('dump_path', dump_path)
            df.to_csv(dump_path, columns=['questions', 'candidates', 'pred', 'gold', 'score'], index=False, sep="\t")
            # hzp add 导出预测结果 分析top1,2,3的acc@1,acc@2,acc@3
            if 'ccks_kbqa_2020' in args.kg_name:
                results = {}
                result = compute_ranking_acc(df, 1, file_path_2)
                results.update(result)
                result = compute_ranking_acc(df, 3, file_path_2)
                results.update(result)
                result = compute_ranking_acc(df, 5, file_path_2)
                results.update(result)
            else:
                results = {}
                result = compute_ranking_acc(df, 1, file_path_2)
                results.update(result)
                result = compute_ranking_acc(df, 2, file_path_2)
                results.update(result)
                result = compute_ranking_acc(df, 3, file_path_2)
                results.update(result)
            print(results)

            print("Confusion matrix:(is_test:{})".format(args.is_test))
            print(confusion)
            print("Report precision, recall, and f1:")

            for i in range(confusion.size()[0]):
                try:
                    p = confusion[i, i].item() / confusion[i, :].sum().item()
                except ZeroDivisionError:
                    p = 0
                try:
                    r = confusion[i, i].item() / confusion[:, i].sum().item()
                except ZeroDivisionError:
                    r = 0
                try:
                    f1 = 2 * p * r / (p + r)
                except ZeroDivisionError:
                    f1 = 0
                if i == 1:
                    label_1_f1 = f1
                print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
            results.update({"Acc. (Correct/Total)": "{:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset))})
            print('results', results)
            # 执行结果 写入文档
            output_eval_file = os.path.join(file_path_2, "eval_results.txt")  # 输出评估结果到目录
            outputs_eval_results(results, args, output_eval_file)
            if metrics == 'Acc':
                return correct / len(dataset)
            elif metrics == 'f1':
                return label_1_f1
            else:
                return correct / len(dataset)
        else:
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

                vms_batch = torch.LongTensor(np.array(vms_batch))

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all = logits
                if i >= 1:
                    logits_all = torch.cat((logits_all, logits), 0)

            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][-1]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid, j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid, j))

            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order = gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid = int(dataset[i][-1])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            print(len(score_list))
            print(len(label_order))
            for i in range(len(score_list)):
                if len(label_order[i]) == 1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i], reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print("MRR", MRR)
            return MRR

    # Training phase.
    if args.do_train:
        print("Start training.")
        trainset = read_dataset(args.train_path, workers_num=args.workers_num)
        print("Shuffling dataset")
        random.shuffle(trainset)
        instances_num = len(trainset)
        batch_size = args.batch_size
        print("Trans data to tensor.")
        # print("input_ids")
        input_ids = torch.LongTensor(np.array([example[0] for example in trainset]))
        # print("label_ids")
        label_ids = torch.LongTensor(np.array([example[1] for example in trainset]))
        # print("mask_ids")
        mask_ids = torch.LongTensor(np.array([example[2] for example in trainset]))
        # print("pos_ids")
        pos_ids = torch.LongTensor(np.array([example[3] for example in trainset]))
        # print("vms")
        vms = [example[4] for example in trainset]
        train_steps = int(instances_num * args.epochs_num / batch_size) + 1
        print("Batch size: ", batch_size)
        print("The number of training instances:", instances_num)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)
        total_loss = 0.
        result = 0.0
        best_result = 0.0
        for epoch in range(1, args.epochs_num + 1):
            model.train()
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
                model.zero_grad()
                vms_batch = torch.LongTensor(np.array(vms_batch))
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)
                loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)
                if torch.cuda.device_count() > 1:
                    loss = torch.mean(loss)
                total_loss += loss.item()
                if (i + 1) % args.report_steps == 0:
                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                    sys.stdout.flush()
                    total_loss = 0.
                loss.backward()
                optimizer.step()
            print("Start evaluation on dev dataset.")
            result = evaluate(args)
            if result > best_result:
                best_result = result
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

    # 最后，计算实体链接的得分
    if args.compute_pipeline_el_metric:
        # 分别从两个任务的模型输出路径读取评估结果，然后计算得到el的得分
        md_eval_path = os.path.join(args.pipeline_el_md_dir, "eval_results.txt")
        logger.info('md评估结果路径:{}'.format(md_eval_path))
        md_metric = csv_reader_static(md_eval_path)
        # print('md_metric', md_metric, os.path.join(args.pipeline_el_md_dir, "eval_results.txt"))

        for i in md_metric:
            if 'f1_score' in i:
                t = i.replace('f1_score', '')
                term = re.findall(r'\d+\.?\d*', t)
                f1 = term[0]
            if 'precision' in i:
                t = i.replace('precision', '')
                term = re.findall(r'\d+\.?\d*', t)
                p = term[0]
            if 'recall' in i:
                t = i.replace('recall', '')
                term = re.findall(r'\d+\.?\d*', t)
                r = term[0]
        logger.info('最近一次的md表现，f1:{},p:{},r:{}'.format(f1, p, r))
        ed_eval_path = os.path.join(args.pipeline_el_ed_dir, "eval_results.txt")
        logger.info('ed评估结果路径:{}'.format(ed_eval_path))
        ed_metric = csv_reader_static(ed_eval_path)
        if 'ccks_kbqa_2020' in args.kg_name:
            for i in ed_metric:
                if 'acc@1' in i:
                    t = i.replace('acc@1', '')
                    term = re.findall(r'\d+\.?\d*', t)
                    acc1 = term[0]
                if 'acc@3' in i:
                    t = i.replace('acc@3', '')
                    term = re.findall(r'\d+\.?\d*', t)
                    acc3 = term[0]
                if 'acc@5' in i:
                    t = i.replace('acc@5', '')
                    term = re.findall(r'\d+\.?\d*', t)
                    acc5 = term[0]
            logger.info('最近一次的ed表现，acc1:{},acc3:{},acc5:{}'.format(acc1, acc3, acc5))
            el_r1 = float(r) * float(acc1)
            el_r3 = float(r) * float(acc3)
            el_r5 = float(r) * float(acc5)
            el_f1 = float(f1) * float(acc1)
            el_r = float(r) * float(acc1)
            el_p = float(p) * float(acc1)
            results = {
                "el_recall1": el_r1,
                "el_recall3": el_r3,
                "el_recall5": el_r5,
                "el_f1": el_f1,
                "el_p": el_p,
                "el_r": el_r,
            }
        else:
            # print('ed_metric', ed_metric, os.path.join(args.pipeline_el_ed_dir, "eval_results.txt"))
            for i in ed_metric:
                if 'acc@1' in i:
                    t = i.replace('acc@1', '')
                    term = re.findall(r'\d+\.?\d*', t)
                    acc1 = term[0]
                if 'acc@2' in i:
                    t = i.replace('acc@2', '')
                    term = re.findall(r'\d+\.?\d*', t)
                    acc2 = term[0]
                if 'acc@3' in i:
                    t = i.replace('acc@3', '')
                    term = re.findall(r'\d+\.?\d*', t)
                    acc3 = term[0]
            logger.info('最近一次的ed表现，acc1:{},acc2:{},acc3:{}'.format(acc1, acc2, acc3))
            el_acc1 = float(p) * float(acc1)
            el_acc2 = float(p) * float(acc2)
            el_acc3 = float(p) * float(acc3)
            el_f1 = float(f1) * float(acc1)
            el_r = float(r) * float(acc1)
            el_p = float(p) * float(acc1)
            results = {
                "el_acc1": el_acc1,
                "el_acc2": el_acc2,
                "el_acc3": el_acc3,
                "el_f1": el_f1,
                "el_p": el_p,
                "el_r": el_r,
            }
        # 执行结果 写入文档
        outputs_el_results(results, args, os.path.join(os.path.abspath(os.path.join(args.output_model_path, "..")), "eval_results.txt"))


if __name__ == "__main__":
    main(args)
