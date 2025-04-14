#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : config.py
# Date    : 2022-08-15
class Config:
    def __init__(self) -> None:
        self.nlpcc_kbqa = {
            "step0": {
                "mention2id_path": "data/raw/nlpcc_kbqa/nlpcc-iccpol-2016.kbqa.kb.mention2id",
                "kb_path": "data/raw/nlpcc_kbqa/nlpcc-iccpol-2016-new.spo",
                "load_train_data_path": "data/raw/nlpcc_kbqa/nlpcc2016.kbqa.train",
                "load_test_data_path": "data/raw/nlpcc_kbqa/nlpcc2016.kbqa.test",
                "dump_train_data_path": "data/processed/nlpcc_kbqa/train.json",
                "dump_test_data_path": "data/processed/nlpcc_kbqa/test.json",
                "train_mention_candidates_path": "data/processed/nlpcc_kbqa/mention_candidates_train.json",
                "test_mention_candidates_path": "data/processed/nlpcc_kbqa/mention_candidates_test.json",
                "train_filter_kb_path": "data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_train.spo",
                "test_filter_kb_path": "data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_test.spo",
            },
            "step1": {
                "load_train_data_path": "data/processed/nlpcc_kbqa/train.json",
                "load_test_data_path": "data/processed/nlpcc_kbqa/test.json",
                "dump_train_data_path": "data/processed/nlpcc_kbqa/md/train.tsv",
                "dump_dev_data_path": "data/processed/nlpcc_kbqa/md/dev.tsv",
                "dump_test_data_path": "data/processed/nlpcc_kbqa/md/test.tsv",
                "dump_train_data_path_w2ner": "data/processed/nlpcc_kbqa/md_w2ner/train.tsv",
                "dump_dev_data_path_w2ner": "data/processed/nlpcc_kbqa/md_w2ner/dev.tsv",
                "dump_test_data_path_w2ner": "data/processed/nlpcc_kbqa/md_w2ner/test.tsv",
            },
            "step2": {
                "load_train_data_path": "data/processed/nlpcc_kbqa/train.json",
                "load_test_data_path": "data/processed/nlpcc_kbqa/test.json",
                "dump_train_data_path": "data/processed/nlpcc_kbqa/ed/train.tsv",
                "dump_dev_data_path": "data/processed/nlpcc_kbqa/ed/dev.tsv",
                "dump_test_data_path": "data/processed/nlpcc_kbqa/ed/test.tsv",
                "dump_train_data_path_kbert": "data/processed/nlpcc_kbqa/ed_kbert/train.tsv",
                "dump_dev_data_path_kbert": "data/processed/nlpcc_kbqa/ed_kbert/dev.tsv",
                "dump_test_data_path_kbert": "data/processed/nlpcc_kbqa/ed_kbert/test.tsv",
                "dump_train_data_path_ekbert": "data/processed/nlpcc_kbqa/ed_ekbert/train.tsv",
                "dump_dev_data_path_ekbert": "data/processed/nlpcc_kbqa/ed_ekbert/dev.tsv",
                "dump_test_data_path_ekbert": "data/processed/nlpcc_kbqa/ed_ekbert/test.tsv",
            }
        }
        self.ccks_el_19 = {
            "step0": {
                "load_kb_data_path": "data/raw/ccks_el_2019/kb_data",
                "dump_kb_data_path": "data/processed/ccks_el_2019/ccks_2019_kb.spo",
                "dump_mention2id_path": "data/processed/ccks_el_2019/mention2id",
            },
            "step1": {
                "load_data_path": "data/raw/ccks_el_2019/train.json",
                "dump_train_data_path": "data/processed/ccks_el_2019/md/train.tsv",
                "dump_dev_data_path": "data/processed/ccks_el_2019/md/dev.tsv",
                "dump_test_data_path": "data/processed/ccks_el_2019/md/test.tsv",
            },
            "step2": {
                "dump_train_data_path_sota": "data/processed/ccks_el_2019/ed_sota/train_baseline.tsv",
                "dump_dev_data_path_sota": "data/processed/ccks_el_2019/ed_sota/dev_baseline.tsv",
                "dump_test_data_path_sota": "data/processed/ccks_el_2019/ed_sota/test_baseline.tsv",
                "dump_train_data_path_kbert": "data/processed/ccks_el_2019/ed_kbert/train.tsv",
                "dump_dev_data_path_kbert": "data/processed/ccks_el_2019/ed_kbert/dev.tsv",
                "dump_test_data_path_kbert": "data/processed/ccks_el_2019/ed_kbert/test.tsv",
                "dump_train_data_path_ekbert": "data/processed/ccks_el_2019/ed_ekbert/train.tsv",
                "dump_dev_data_path_ekbert": "data/processed/ccks_el_2019/ed_ekbert/dev.tsv",
                "dump_test_data_path_ekbert": "data/processed/ccks_el_2019/ed_ekbert/test.tsv",
            }
        }
        self.ccks_el_20 = {
            "step0": {
                "load_kb_data_path": "data/raw/ccks_el_2020/kb.json",
                "dump_kb_data_path": "data/processed/ccks_el_2020/ccks_2020_kb.spo",
                "load_mention2id_path": "data/raw/ccks_el_2020/pkubase-mention2ent.txt",
                "dump_mention2id_path": "data/processed/ccks_el_2020/mention2id.txt",
            },
            "step1": {
                "load_data_path": "data/raw/ccks_el_2020/train.json",
                "dump_train_data_path": "data/processed/ccks_el_2020/md/train.tsv",
                "dump_dev_data_path": "data/processed/ccks_el_2020/md/dev.tsv",
                "dump_test_data_path": "data/processed/ccks_el_2020/md/test.tsv",
            },
            "step2": {
                "load_data_path": "data/raw/ccks_el_2020/train.json",
                "dump_train_data_path_kbert": "data/processed/ccks_el_2020/ed_kbert/train.tsv",
                "dump_dev_data_path_kbert": "data/processed/ccks_el_2020/ed_kbert/dev.tsv",
                "dump_test_data_path_kbert": "data/processed/ccks_el_2020/ed_kbert/test.tsv",
                "dump_train_data_path_ekbert": "data/processed/ccks_el_2020/ed_ekbert/train.tsv",
                "dump_dev_data_path_ekbert": "data/processed/ccks_el_2020/ed_ekbert/dev.tsv",
                "dump_test_data_path_ekbert": "data/processed/ccks_el_2020/ed_ekbert/test.tsv",
            }
        }
        self.ccks_kbqa_20 = {
            "step0": {
                "load_kb_data_path": "data/raw/ccks_kbqa_2020/pkubase-complete2.txt",
                "dump_kb_data_path": "data/processed/ccks_kbqa_2020/ccks_kbqa_2020.spo",
                "load_mention2id_data_path": "data/raw/ccks_kbqa_2020/pkubase-mention2ent.txt",
                "dump_mention2id_data_path": "data/processed/ccks_kbqa_2020/mention2id.txt",
                "dump_filter_kb_data_path": "data/processed/ccks_kbqa_2020/ccks_kbqa_2020_filter.spo",
            },
            "step1": {
                "load_train_data_path": "data/raw/ccks_kbqa_2020/train.tsv",
                "load_dev_data_path": "data/raw/ccks_kbqa_2020/dev.tsv",
                "load_test_data_path": "data/raw/ccks_kbqa_2020/test.tsv",
                "dump_train_data_path": "data/processed/ccks_kbqa_2020/md/train.tsv",
                "dump_dev_data_path": "data/processed/ccks_kbqa_2020/md/dev.tsv",
                "dump_test_data_path": "data/processed/ccks_kbqa_2020/md/test.tsv",
            },
            "step2": {
                "dump_train_data_path_kbert": "data/processed/ccks_kbqa_2020/ed_kbert/train.tsv",
                "dump_dev_data_path_kbert": "data/processed/ccks_kbqa_2020/ed_kbert/dev.tsv",
                "dump_test_data_path_kbert": "data/processed/ccks_kbqa_2020/ed_kbert/test.tsv",
                "dump_train_data_path_ekbert": "data/processed/ccks_kbqa_2020/ed_ekbert/train.tsv",
                "dump_dev_data_path_ekbert": "data/processed/ccks_kbqa_2020/ed_ekbert/dev.tsv",
                "dump_test_data_path_ekbert": "data/processed/ccks_kbqa_2020/ed_ekbert/test.tsv",
            }
        }


cfg = Config()
