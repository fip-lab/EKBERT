#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : run_entity_linking.py
# Date    : 2022-08-30
from utils import load_jsonl, dump_jsonl, csv_reader, csv_reader_static, csv_writer
import os
from config import cfg
import argparse
from loguru import logger
from typing import Dict


def filter_ed_data_by_md_hitted_text_a(args, md_hitted_text_and_mid: Dict, ed_org_data_path: str, ed_filter_data_path: str):
    """
    根据md的预测数据，筛选ed数据，生成新的ed数据，进行后续预测
    :param args:
    :param md_hitted_text_and_mid:
    :param ed_org_data_path:
    :param ed_filter_data_path:
    :return:
    """
    ed_org_data = csv_reader_static(ed_org_data_path)
    data_list = []
    if args.dataset == 'nlpcc':
        if args.model_name in ['ekbert']:
            for line in ed_org_data:
                text_a, text_b, label = line.split('\t')
                raw_text_a = text_a.replace('<e1>', '').replace('</e1>', '')  # 恢复原始文本
                if raw_text_a in md_hitted_text_and_mid:
                    data_list.append(line)
        elif args.model_name in ['bbkbqa']:
            for line in ed_org_data:
                text_a, text_b, label = line.split('\t')
                raw_text_a = text_a
                if raw_text_a in md_hitted_text_and_mid:
                    data_list.append(line)
        elif args.model_name in ['kbert']:
            data_list.append(ed_org_data[1])  # 写表头
            ed_org_data = ed_org_data[1:]  # 跳过第一行表头
            for line in ed_org_data:
                text_a, text_b, label = line.split('\t')
                raw_text_a = text_a
                if raw_text_a in md_hitted_text_and_mid:
                    data_list.append(line)
    elif args.dataset == 'cckskbqa20':
        if args.model_name in ['ekbert']:
            for line in ed_org_data:
                tid, text_a, text_b, label = line.split('\t')
                raw_text_a = text_a.replace('<e1>', '').replace('</e1>', '')  # 恢复原始文本
                if raw_text_a in md_hitted_text_and_mid:
                    data_list.append(line)
        elif args.model_name in ['kbert']:
            data_list.append(ed_org_data[1])  # 写表头
            ed_org_data = ed_org_data[1:]  # 跳过第一行表头
            for line in ed_org_data:
                tid, text_a, text_b, label = line.split('\t')
                raw_text_a = text_a
                if raw_text_a in md_hitted_text_and_mid:
                    data_list.append(line)
    elif args.dataset == 'ccksel19':
        if args.model_name in ['ekbert']:
            examples_mention_dict = dict()  # eg. {example1:{mention1:0,mention2:1,mention3:2 ...}}
            for line in ed_org_data:
                text_id, text_a, text_b, label = line.split('\t')
                mention_pos = text_a
                raw_text_a = text_a.replace('<e1>', '').replace('</e1>', '').replace(' ', '')  # 恢复原始文本
                if raw_text_a in examples_mention_dict:
                    if mention_pos in examples_mention_dict[raw_text_a]:
                        continue
                    else:
                        examples_mention_dict[raw_text_a][mention_pos] = len(examples_mention_dict[raw_text_a].keys())
                else:
                    examples_mention_dict[raw_text_a] = {mention_pos: 0}
            # print('examples_mention_dict', examples_mention_dict)
            for line in ed_org_data:
                text_id, text_a, text_b, label = line.split('\t')
                mention_pos = text_a
                raw_text_a = text_a.replace('<e1>', '').replace('</e1>', '').replace(' ', '')  # 恢复原始文本
                if raw_text_a in md_hitted_text_and_mid:
                    hitted_mention = md_hitted_text_and_mid[raw_text_a]  # 取命中mention序列列表 [0,2,4]
                    example_mention = examples_mention_dict[raw_text_a]
                    mid = example_mention[mention_pos]
                    if mid in hitted_mention:
                        data_list.append(line)
        elif args.model_name in ['kbert']:
            data_list.append(ed_org_data[1])  # 写表头
            ed_org_data = ed_org_data[1:]  # 跳过第一行表头
            examples_mention_dict = dict()  # eg. {example1:{mention1:0,mention2:1,mention3:2 ...}}
            for line in ed_org_data:
                text_id, text_a, text_b, label, mention_pos = line.split('\t')
                raw_text_a = text_a.replace(' ', '')  # 恢复原始文本
                if raw_text_a in examples_mention_dict:
                    if mention_pos in examples_mention_dict[raw_text_a]:
                        continue
                    else:
                        examples_mention_dict[raw_text_a][mention_pos] = len(examples_mention_dict[raw_text_a].keys())
                else:
                    examples_mention_dict[raw_text_a] = {mention_pos: 0}
            # print('examples_mention_dict', examples_mention_dict)
            for line in ed_org_data:
                text_id, text_a, text_b, label, mention_pos = line.split('\t')
                raw_text_a = text_a.replace(' ', '')  # 恢复原始文本
                if raw_text_a in md_hitted_text_and_mid:
                    hitted_mention = md_hitted_text_and_mid[raw_text_a]  # 取命中mention序列列表 [0,2,4]
                    example_mention = examples_mention_dict[raw_text_a]
                    mid = example_mention[mention_pos]
                    if mid in hitted_mention:
                        data_list.append(line)
        elif args.model_name in ['sota']:
            examples_mention_dict = dict()  # eg. {example1:{mention1:0,mention2:1,mention3:2 ...}}
            for line in ed_org_data:
                text_id, text_a, text_b, label, mention_pos = line.split('\t')
                raw_text_a = text_a.replace('<e1>', '').replace('</e1>', '').replace(' ', '')  # 恢复原始文本
                if raw_text_a in examples_mention_dict:
                    if mention_pos in examples_mention_dict[raw_text_a]:
                        continue
                    else:
                        examples_mention_dict[raw_text_a][mention_pos] = len(examples_mention_dict[raw_text_a].keys())
                else:
                    examples_mention_dict[raw_text_a] = {mention_pos: 0}
            # print('examples_mention_dict', examples_mention_dict)
            for line in ed_org_data:
                text_id, text_a, text_b, label, mention_pos = line.split('\t')
                raw_text_a = text_a.replace(' ', '')  # 恢复原始文本
                if raw_text_a in md_hitted_text_and_mid:
                    hitted_mention = md_hitted_text_and_mid[raw_text_a]  # 取命中mention序列列表 [0,2,4]
                    example_mention = examples_mention_dict[raw_text_a]
                    mid = example_mention[mention_pos]
                    if mid in hitted_mention:
                        data_list.append(line)
    elif args.dataset == 'ccksel20':
        if args.model_name in ['ekbert']:
            examples_mention_dict = dict()  # eg. {example1:{mention1:0,mention2:1,mention3:2 ...}}
            for line in ed_org_data:
                text_id, text_a, text_b, label = line.split('\t')
                mention_pos = text_a
                raw_text_a = text_a.replace('<e1>', '').replace('</e1>', '').replace(' ', '')  # 恢复原始文本
                if raw_text_a in examples_mention_dict:
                    if mention_pos in examples_mention_dict[raw_text_a]:
                        continue
                    else:
                        examples_mention_dict[raw_text_a][mention_pos] = len(examples_mention_dict[raw_text_a].keys())
                else:
                    examples_mention_dict[raw_text_a] = {mention_pos: 0}
            # print('examples_mention_dict', examples_mention_dict)
            for line in ed_org_data:
                text_id, text_a, text_b, label = line.split('\t')
                mention_pos = text_a
                raw_text_a = text_a.replace('<e1>', '').replace('</e1>', '').replace(' ', '')  # 恢复原始文本
                if raw_text_a in md_hitted_text_and_mid:
                    hitted_mention = md_hitted_text_and_mid[raw_text_a]  # 取命中mention序列列表 [0,2,4]
                    example_mention = examples_mention_dict[raw_text_a]
                    mid = example_mention[mention_pos]
                    if mid in hitted_mention:
                        data_list.append(line)
        elif args.model_name in ['kbert']:
            data_list.append(ed_org_data[1])  # 写表头
            ed_org_data = ed_org_data[1:]  # 跳过第一行表头
            examples_mention_dict = dict()  # eg. {example1:{mention1:0,mention2:1,mention3:2 ...}}
            for line in ed_org_data:
                text_id, text_a, text_b, label, mention_pos = line.split('\t')
                raw_text_a = text_a.replace(' ', '')  # 恢复原始文本
                if raw_text_a in examples_mention_dict:
                    if mention_pos in examples_mention_dict[raw_text_a]:
                        continue
                    else:
                        examples_mention_dict[raw_text_a][mention_pos] = len(examples_mention_dict[raw_text_a].keys())
                else:
                    examples_mention_dict[raw_text_a] = {mention_pos: 0}
            # print('examples_mention_dict', examples_mention_dict)
            for line in ed_org_data:
                text_id, text_a, text_b, label, mention_pos = line.split('\t')
                raw_text_a = text_a.replace(' ', '')  # 恢复原始文本
                if raw_text_a in md_hitted_text_and_mid:
                    hitted_mention = md_hitted_text_and_mid[raw_text_a]  # 取命中mention序列列表 [0,2,4]
                    example_mention = examples_mention_dict[raw_text_a]
                    mid = example_mention[mention_pos]
                    if mid in hitted_mention:
                        data_list.append(line)
    csv_writer(data_list, ed_filter_data_path)
    logger.info('从md命中的所有text_a中得到等待预测的ed数据：{}条'.format(len(data_list)))


def main():
    """
    用训练好的md,ed模型测试执行整个实体链接的表现
    :return:
    """
    # 读取md任务predict的json结果，取hit_flag为true的输出，
    # 输入到ed任务，ed任务输入hit_flag的true的json结果。
    # 以及el任务的评估结果。
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ekbert", type=str, required=True, help="指定实体链接模型")
    parser.add_argument("--load_md_predict_data_path", type=str, required=True, help="读取指定模型的md预测数据路径")
    parser.add_argument("--dump_ed_predict_data_path", type=str, required=True, help="写出指定模型的md预测数据路径")
    parser.add_argument("--dataset", type=str, default="nlpcc", required=True, help="指定数据集，对不同数据集的预测文件处理有所不同")
    args = parser.parse_args()
    if args.dataset == 'nlpcc':
        if args.model_name == "ekbert":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.nlpcc_kbqa["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.nlpcc_kbqa["step2"]["dump_test_data_path_ekbert"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
        elif args.model_name == "bbkbqa":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.nlpcc_kbqa["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.nlpcc_kbqa["step2"]["dump_test_data_path"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
        elif args.model_name == "kbert":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.nlpcc_kbqa["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.nlpcc_kbqa["step2"]["dump_test_data_path_kbert"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
    elif args.dataset == 'ccksel19':
        if args.model_name == "ekbert":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.ccks_el_19["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.ccks_el_19["step2"]["dump_test_data_path_ekbert"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
        elif args.model_name == "sota":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.ccks_el_19["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.ccks_el_19["step2"]["dump_test_data_path_sota"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
        elif args.model_name == "kbert":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.ccks_el_19["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.ccks_el_19["step2"]["dump_test_data_path_kbert"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
    elif args.dataset == 'cckskbqa20':
        if args.model_name == "ekbert":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.ccks_kbqa_20["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.ccks_kbqa_20["step2"]["dump_test_data_path_ekbert"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
        elif args.model_name == "kbert":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.ccks_kbqa_20["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.ccks_kbqa_20["step2"]["dump_test_data_path_kbert"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
    elif args.dataset == 'ccksel20':
        if args.model_name == "ekbert":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.ccks_el_20["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.ccks_el_20["step2"]["dump_test_data_path_ekbert"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')
        elif args.model_name == "kbert":
            md_predicted_data_path = os.path.join(args.load_md_predict_data_path, 'test_prediction.json')
            md_org_data_path = cfg.ccks_el_20["step1"]["dump_test_data_path"]
            ed_org_data_path = cfg.ccks_el_20["step2"]["dump_test_data_path_kbert"]
            ed_filter_data_path = os.path.join(args.dump_ed_predict_data_path, 'predict.tsv')

    # 从预测数据中取预测正确的下标
    md_predicted_data = load_jsonl(md_predicted_data_path)
    md_hitted_eid_and_mid = dict()  # eg item是 (命中的案例id,这个案例下命中的mention id列表)
    for idx, i in enumerate(md_predicted_data):
        if any([f for f in i['hit_flag']]):
            example_hitted_mention_id = [m_idx for m_idx, f in enumerate(i['hit_flag']) if f is True]
            md_hitted_eid_and_mid[idx] = example_hitted_mention_id

    print(' md_hitted_eid_and_mid.items()[:5]', list(md_hitted_eid_and_mid.items())[:5])

    total_mention_cnt = sum([len(i['hit_flag']) for i in md_predicted_data])
    hitted_mention_cnt = sum([len(j) for i, j in md_hitted_eid_and_mid.items()])

    logger.info('md预测的实体有：{}条，其中命中的实体有：{}条'.format(total_mention_cnt, hitted_mention_cnt))
    # print('md_hitted_data_id',md_hitted_data_id)

    # 通过预测正确的下标找到文本
    md_org_data = csv_reader_static(md_org_data_path)
    md_hitted_text_and_mid = dict()
    for idx, line in enumerate(md_org_data):
        if idx in md_hitted_eid_and_mid:
            text_a = line.split('\t')[0].replace(' ', '')
            md_hitted_text_and_mid[text_a] = md_hitted_eid_and_mid[idx]
    print('md_hitted_text_and_mid.items()[:5]', list(md_hitted_text_and_mid.items())[:5])

    # 通过文本筛选出要预测的ed数据
    filter_ed_data_by_md_hitted_text_a(args, md_hitted_text_and_mid, ed_org_data_path, ed_filter_data_path)


if __name__ == "__main__":
    main()
