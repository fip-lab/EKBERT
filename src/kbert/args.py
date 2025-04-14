#!/usr/bin/env python3
# coding: utf-8
# Author  : penho
# File    : args.py
# Date    : 2022-09-04
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Path options.
parser.add_argument("--pretrained_model_path", default=None, type=str, help="Path of the pretrained model.")
parser.add_argument("--output_model_path", required=True, type=str, help="Path of the output model.")
parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str, help="Path of the vocabulary file.")
parser.add_argument("--train_path", type=str, help="Path of the trainset.")
parser.add_argument("--dev_path", type=str, help="Path of the devset.")
parser.add_argument("--test_path", type=str, help="Path of the testset.")
parser.add_argument("--config_path", default="./models/google_config.json", type=str, help="Path of the config file.")

# Model options.
parser.add_argument("--model_type", type=str, required=True, help="不用任务注入知识的范围不同")
parser.add_argument("--min_span", type=int, default=0, help="指定注入spo的头实体最小长度")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--seq_length", type=int, default=256, help="Sequence length.")
parser.add_argument("--encoder", choices=["bert", "lstm", "gru",
                                          "cnn", "gatedcnn", "attn",
                                          "rcnn", "crnn", "gpt", "bilstm"],
                    default="bert", help="Encoder type.")
parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first", help="Pooling type.")

# Subword options.
parser.add_argument("--subword_type", choices=["none", "char"], default="none", help="Subword feature type.")
parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt", help="Path of the subword vocabulary file.")
parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg", help="Subencoder type.")
parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

# Tokenizer options.
parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                    help="Specify the tokenizer."
                         "Original Google BERT uses bert tokenizer on Chinese corpus."
                         "Char tokenizer segments sentences into characters."
                         "Word tokenizer supports online word segmentation based on jieba segmentor."
                         "Space tokenizer segments sentences into words according to space."
                    )

# Optimizer options.
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")

# Training options.
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout.")
parser.add_argument("--epochs_num", type=int, default=5, help="Number of epochs.")
parser.add_argument("--report_steps", type=int, default=100, help="Specific steps to print prompt.")
parser.add_argument("--seed", type=int, default=7, help="Random seed.")
parser.add_argument("--do_train", action="store_true", help="训练模型.")
parser.add_argument("--is_debug", action="store_true", help="是否调试模式，只跑部分数据")

# Evaluation options.
parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")
parser.add_argument("--do_eval", action="store_true", help="评估模型.")
parser.add_argument("--do_predict", action="store_true", help="预测模型.")
parser.add_argument("--output_predict_result_path", type=str, help="预测结果输出路径")
parser.add_argument("--is_test", action="store_true", help="是否在测试集上评估，否则在验证集")
parser.add_argument("--pipeline_el_md_dir", type=str, help="实体链接的md得分输出路径")
parser.add_argument("--pipeline_el_ed_dir", type=str, help="实体链接的ed得分输出路径")
parser.add_argument("--compute_pipeline_el_metric", action="store_true", help="计算pipeline实体链接得分")

# kg
parser.add_argument("--kg_name", required=True, help="KG name or path")
parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

args = parser.parse_args()
