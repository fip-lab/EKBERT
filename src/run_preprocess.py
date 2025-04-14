#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : data_untils.py
# Date    : 2022-08-16
from preprocessing.clear_data import clear_nlpcc_kbqa_data, clear_ccks_el_kb_data
from preprocessing.create_md_data import create_md_dataset
from preprocessing.create_ed_data import create_ed_dataset, create_ed_dataset_kbert
from preprocessing.infuse_entity_token_and_subgraph import create_ed_dataset_and_infuse_knowledge
from config import cfg


def nlpcc_kbqa_processing():
    """
    nlpcc kbqa 数据集预处理
    :return: 
    """
    # # 0. 清洗原始数据，导出json格式的数据集
    # 输入4个文件
    mention2id_path = cfg.nlpcc_kbqa["step0"]["mention2id_path"]
    kb_path = cfg.nlpcc_kbqa["step0"]["kb_path"]
    load_train_data_path = cfg.nlpcc_kbqa["step0"]["load_train_data_path"]
    load_test_data_path = cfg.nlpcc_kbqa["step0"]["load_test_data_path"]
    # 输出6个文件
    dump_train_data_path = cfg.nlpcc_kbqa["step0"]["dump_train_data_path"]
    dump_test_data_path = cfg.nlpcc_kbqa["step0"]["dump_test_data_path"]
    train_mention_candidates_path = cfg.nlpcc_kbqa["step0"]["train_mention_candidates_path"]
    test_mention_candidates_path = cfg.nlpcc_kbqa["step0"]["test_mention_candidates_path"]
    train_filter_kb_path = cfg.nlpcc_kbqa["step0"]["train_filter_kb_path"]
    test_filter_kb_path = cfg.nlpcc_kbqa["step0"]["test_filter_kb_path"]
    # 清洗训练集
    clear_nlpcc_kbqa_data(load_train_data_path, mention2id_path, kb_path, train_mention_candidates_path, train_filter_kb_path, dump_train_data_path)
    # 清洗测试集
    clear_nlpcc_kbqa_data(load_test_data_path, mention2id_path, kb_path, test_mention_candidates_path, test_filter_kb_path, dump_test_data_path)

    # 1. 创建提及识别数据集
    # 训练集+验证集
    load_train_data_path = cfg.nlpcc_kbqa["step1"]["load_train_data_path"]
    dump_train_data_path = cfg.nlpcc_kbqa["step1"]["dump_train_data_path"]
    dump_dev_data_path = cfg.nlpcc_kbqa["step1"]["dump_dev_data_path"]
    create_md_dataset(load_train_data_path, [dump_train_data_path, dump_dev_data_path])
    # 测试集
    load_test_data_path = cfg.nlpcc_kbqa["step1"]["load_test_data_path"]
    dump_test_data_path = cfg.nlpcc_kbqa["step1"]["dump_test_data_path"]
    create_md_dataset(load_test_data_path, [dump_test_data_path])

    # 2.创建实体消歧数据集
    # BB-KBQA对实体消歧数据预处理
    # 训练集+验证集
    load_train_data_path = cfg.nlpcc_kbqa["step2"]["load_train_data_path"]
    dump_train_data_path = cfg.nlpcc_kbqa["step2"]["dump_train_data_path"]
    dump_dev_data_path = cfg.nlpcc_kbqa["step2"]["dump_dev_data_path"]
    create_ed_dataset(load_train_data_path, [dump_train_data_path, dump_dev_data_path])
    # 测试集
    load_test_data_path = cfg.nlpcc_kbqa["step2"]["load_test_data_path"]
    dump_test_data_path = cfg.nlpcc_kbqa["step2"]["dump_test_data_path"]
    create_ed_dataset(load_test_data_path, [dump_test_data_path])

    # KBERT对实体消歧数据预处理
    # 训练集+验证集
    load_data_path = cfg.nlpcc_kbqa["step2"]["load_train_data_path"]
    dump_train_data_path = cfg.nlpcc_kbqa["step2"]["dump_train_data_path_kbert"]
    dump_dev_data_path = cfg.nlpcc_kbqa["step2"]["dump_dev_data_path_kbert"]
    create_ed_dataset_kbert(load_data_path, [dump_train_data_path, dump_dev_data_path])
    # 测试集
    load_data_path = cfg.nlpcc_kbqa["step2"]["load_test_data_path"]
    dump_data_path = cfg.nlpcc_kbqa["step2"]["dump_test_data_path_kbert"]
    create_ed_dataset_kbert(load_data_path, [dump_data_path])

    # EKBERT对实体消歧数据预处理
    # train + dev
    load_data_path = cfg.nlpcc_kbqa["step2"]["load_train_data_path"]
    dump_train_data_path = cfg.nlpcc_kbqa["step2"]["dump_train_data_path_ekbert"]
    dump_dev_data_path = cfg.nlpcc_kbqa["step2"]["dump_dev_data_path_ekbert"]
    filter_spo_path = cfg.nlpcc_kbqa["step0"]["train_filter_kb_path"]
    create_ed_dataset_and_infuse_knowledge(load_data_path, [dump_train_data_path, dump_dev_data_path], filter_spo_path)
    # test
    load_data_path = cfg.nlpcc_kbqa["step2"]["load_test_data_path"]
    dump_data_path = cfg.nlpcc_kbqa["step2"]["dump_test_data_path_kbert"]
    filter_spo_path = cfg.nlpcc_kbqa["step0"]["test_filter_kb_path"]
    create_ed_dataset_and_infuse_knowledge(load_data_path, [dump_data_path], filter_spo_path)


def ccks_el_processing():
    """
    ccks el 数据集 预处理
    :return: 
    """
    # 清理 kb 数据
    # load_kb_data_path = cfg.ccks_el["step0"]["load_kb_data_path"]
    # dump_kb_data_path = cfg.ccks_el["step0"]["dump_kb_data_path"]
    # load_dataset_path = cfg.ccks_el["step1"]["load_data_path"]
    # dump_mention2id_path = cfg.ccks_el["step0"]["dump_mention2id_path"]
    # clear_ccks_el_kb_data(load_kb_data_path, dump_kb_data_path,load_dataset_path,dump_mention2id_path)

    # # 创建 md 数据集
    load_data_path = cfg.ccks_el["step1"]["load_data_path"]
    dump_train_data_path = cfg.ccks_el["step1"]["dump_train_data_path"]
    dump_dev_data_path = cfg.ccks_el["step1"]["dump_dev_data_path"]
    dump_test_data_path = cfg.ccks_el["step1"]["dump_test_data_path"]
    # 训练集
    create_md_dataset(load_data_path, [dump_train_data_path], ds_name="ccksel19", ds_type="train")
    # 验证集
    create_md_dataset(load_data_path, [dump_dev_data_path], ds_name="ccksel19", ds_type="dev")
    # 测试集
    create_md_dataset(load_data_path, [dump_test_data_path], ds_name="ccksel19", ds_type="test")
    #
    # # 创建 ed 数据集
    # load_data_path = cfg.ccks_el["step1"]["load_data_path"]
    # dump_train_data_path = cfg.ccks_el["step2"]["dump_train_data_path"]
    # dump_dev_data_path = cfg.ccks_el["step2"]["dump_dev_data_path"]
    # dump_test_data_path = cfg.ccks_el["step2"]["dump_test_data_path"]
    # mention2id_path = cfg.ccks_el["step0"]["dump_mention2id_path"]
    # # 训练集
    # create_ed_dataset(load_data_path, [dump_train_data_path], ds_name="ccksel19", ds_type="train", mention2id_path=mention2id_path)
    # # 验证集
    # create_ed_dataset(load_data_path, [dump_dev_data_path], ds_name="ccksel19", ds_type="dev", mention2id_path=mention2id_path)
    # # 测试集
    # create_ed_dataset(load_data_path, [dump_test_data_path], ds_name="ccksel19", ds_type="test", mention2id_path=mention2id_path)

    # # 创建 kbert ed 数据集
    # load_data_path = cfg.ccks_el["step1"]["load_data_path"]
    # dump_train_data_path = cfg.ccks_el["step2"]["dump_train_data_path_kbert"]
    # dump_dev_data_path = cfg.ccks_el["step2"]["dump_dev_data_path_kbert"]
    # dump_test_data_path = cfg.ccks_el["step2"]["dump_test_data_path_kbert"]
    # mention2id_path = cfg.ccks_el["step0"]["dump_mention2id_path"]
    # # 训练集
    # create_ed_dataset_kbert(load_data_path, [dump_train_data_path], ds_name="ccksel19", ds_type="train", mention2id_path=mention2id_path)
    # # 验证集
    # create_ed_dataset_kbert(load_data_path, [dump_dev_data_path], ds_name="ccksel19", ds_type="dev", mention2id_path=mention2id_path)
    # # 测试集
    # create_ed_dataset_kbert(load_data_path, [dump_test_data_path], ds_name="ccksel19", ds_type="test", mention2id_path=mention2id_path)
    #
    # # EKBERT对实体消歧数据预处理
    # # train + dev
    # load_data_path = cfg.nlpcc_kbqa["step2"]["load_train_data_path"]
    # dump_train_data_path = cfg.nlpcc_kbqa["step2"]["dump_train_data_path_ekbert"]
    # dump_dev_data_path = cfg.nlpcc_kbqa["step2"]["dump_dev_data_path_ekbert"]
    # filter_spo_path = cfg.nlpcc_kbqa["step0"]["train_filter_kb_path"]
    # create_ed_dataset_and_infuse_knowledge(load_data_path, [dump_train_data_path, dump_dev_data_path], filter_spo_path)
    # # test
    # load_data_path = cfg.nlpcc_kbqa["step2"]["load_test_data_path"]
    # dump_data_path = cfg.nlpcc_kbqa["step2"]["dump_test_data_path_kbert"]
    # filter_spo_path = cfg.nlpcc_kbqa["step0"]["test_filter_kb_path"]
    # create_ed_dataset_and_infuse_knowledge(load_data_path, [dump_data_path], filter_spo_path)

    pass


if __name__ == "__main__":
    # nlpcc_kbqa_processing()
    ccks_el_processing()
    pass
