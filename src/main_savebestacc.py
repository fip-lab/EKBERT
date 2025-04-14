import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModel
)
import torch.nn.functional as F
from models import BertForSeqAttenWeightedSum, BertForSeqClsPlusMax, BertForSeqClsPlusAvg, EKBERT, FCLayer, EKGBERT, EKBERTFNN, EKBERTM, CCKSELNO1, BertCrfForNer  # 上一级目录与model在同一目录，可以导入模块model
from prepro import processors, output_modes, load_and_cache_examples, compute_metrics, ner_compute_metrics, ner_F1, ner_f1_new
from parser_args import args
from utils import dump_jsonl, get_entities, get_entity_bio
from loguru import logger
from typing import Dict
import time
from utils import csv_reader_static
import re

# 分类、匹配、序列标注
MODEL_CLASSES = {
    "bert_seq_cls": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "bert_seq_attention": (BertConfig, BertForSeqAttenWeightedSum, BertTokenizer),
    "bert_seq_clsplusmax": (BertConfig, BertForSeqClsPlusMax, BertTokenizer),
    "bert_seq_clsplusavg": (BertConfig, BertForSeqClsPlusAvg, BertTokenizer),
    "bert_seq_e_enhance": (BertConfig, EKBERT, BertTokenizer),
    "bert_seq_e_enhance_simple": (BertConfig, EKBERTFNN, BertTokenizer),
    "bert_seq_ekg_enhance": (BertConfig, EKGBERT, BertTokenizer),
    "ner": (BertConfig, BertForTokenClassification, BertTokenizer),
    "bert_seq_e_enhance_concate": (BertConfig, EKBERTM, BertTokenizer),
    "ernie": (BertConfig, AutoModel, AutoTokenizer),
    "ccks_no_1": (BertConfig, CCKSELNO1, BertTokenizer),
    "ccks_no_1_ner": (BertConfig, BertCrfForNer, BertTokenizer)
}


def set_seed(args):
    """
    设置随机种子 用于复现结果
    :param args:
    :return:
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def top(df, n=2, column='score'):
    """
    返回df指定列的值最大的n行
    data = {'score': [0.92, 0.67, 0.8, 0.2, 0.4, 0.12, 0.34, 0.8],
            'label': [1, 0, 0, 1, 0, 0, 0, 1],
            'query': ['玫瑰主要生长在我国那些地区', '玫瑰主要生长在我国那些地区', '玫瑰主要生长在我国那些地区', '傅博是什么刊物的主编', '傅博是什么刊物的主编', 'text_a', 'text_a', 'text_a']}
    df = pd.DataFrame(data)
    top(df)
    :param df:
    :param n:
    :param column:
    :return:
    """
    return df.sort_values(by=column, ascending=False)[:n]


def compute_ranking_acc(df, top_n, args):
    """
    :param df:所有包含二分类softmax得分的案例df
    :param top_n:前几的得分
    :param args:参数文件parser_args.py
    :return {acc@top_n:88%}
    """
    # data = {'score': [0.92, 0.67, 0.8, 0.2, 0.4, 0.12, 0.34, 0.8],
    #         'label': [1, 0, 0, 1, 0, 0, 0, 1],
    #         'query': ['玫瑰主要生长在我国那些地区', '玫瑰主要生长在我国那些地区', '玫瑰主要生长在我国那些地区', '傅博是什么刊物的主编', '傅博是什么刊物的主编', 'text_a', 'text_a', 'text_a']}
    # df = pd.DataFrame(data)
    # print(df)
    # 输出同一query下top_n个分值
    if args.ccks_no_1:
        grouped = df.groupby(['query', 'pos']).apply(top, n=top_n)  # 用top()方法对query列进行groupby
        # print(grouped)
        # top_n个候选中是否包含gold eg:l_hit->[1,0,1]  表示query1 和query3 包含了gold
        l_hit = grouped['label'].groupby(['query', 'pos']).max().tolist()  # 前top_n个候选包含了标签1的案例，表示命中
        # print('acc@{}'.format(top_n) + ' hit/not-hit(1/0)', l_hit, '索引对应每个query')
        count = 0
        for v in l_hit:
            count += v
        acc = count / len(l_hit)
        print(count, len(l_hit), acc)
        # 将df写入tsv
        df = grouped[['pre', 'label', 'result', 'score', 'example']]
        output_dir = os.path.join(args.output_dir, "ranking_error_als_{}.tsv".format(args.input_test_name))
        df.to_csv(output_dir, sep='\t')
    else:
        grouped = df.groupby('query').apply(top, n=top_n)  # 用top()方法对query列进行groupby
        # print(grouped)
        # top_n个候选中是否包含gold eg:l_hit->[1,0,1]  表示query1 和query3 包含了gold
        l_hit = grouped['label'].groupby('query').max().tolist()  # 前top_n个候选包含了标签1的案例，表示命中
        # print('acc@{}'.format(top_n) + ' hit/not-hit(1/0)', l_hit, '索引对应每个query')
        count = 0
        for v in l_hit:
            count += v
        acc = count / len(l_hit)
        print(count, len(l_hit), acc)
        # 将df写入tsv
        df = grouped[['pre', 'label', 'result', 'score', 'example']]
        output_dir = os.path.join(args.output_dir, "ranking_error_als_{}.tsv".format(args.input_test_name))
        df.to_csv(output_dir, sep='\t')
    return {"acc@{}".format(top_n): acc}


def error_analysis(preds, out_label_ids, n_softmax, eval_examples, args):
    """
    输出预测错误的案例分析
    :param preds: ndarray (batch_size,)
    :param out_label_ids: ndarray (batch_size,)
    :param n_softmax: ndarray (batch_size,2)
    :param eval_examples: list
    :param args: 模型执行的参数设置
    """
    text_a = [l.text_a for l in eval_examples]
    pos = [l.pos for l in eval_examples]
    # print(text_a)
    equal = (preds == out_label_ids)
    assert len(equal) == len(eval_examples)
    # eval_examples = np.array(eval_examples)  # list->ndarray
    softmax_score = n_softmax[:, 1].tolist()  # 2d ndarray -> list
    if args.ccks_no_1:
        data = {'result': equal, 'score': softmax_score, 'pre': preds, 'label': out_label_ids, 'example': eval_examples, 'query': text_a, 'pos': pos}
    else:
        data = {'result': equal, 'score': softmax_score, 'pre': preds, 'label': out_label_ids, 'example': eval_examples, 'query': text_a}
    df = pd.DataFrame(data)
    # print(df)
    # 将df写入tsv
    output_dir = os.path.join(args.output_dir, "error_als_{}.tsv".format(args.input_test_name))
    df.to_csv(output_dir, sep='\t')
    return df


# 定义训练过程
def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)  # 打乱输入的训练数据集
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)  # 打乱后的训练示例以指定的批次大小装入数据加载器 1350条数据，批次设置为8，一共需要169个加载器
    # //运算表示 整除(向小取整)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs  # 所有数据按批次划分出169个指定批次大小的加载器 / 每批次数据的加载器更新多少次参数（默认1） * 训练轮次  eg：（1350/8）上取整/ 1 * 10 =1690（步）
    # 设置优化器 和 学习率
    no_decay = ["bias", "LayerNorm.weight", 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)  # adam_epsilon 用于数据稳定性的超参数
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states 如果本地有就加载
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # logger输出日志
    logger.info("***** Running training *****")
    logger.info("  Num examples = {}".format(len(train_dataset)))  # 输出训练集的样本数
    logger.info("  Num Epochs = {}".format(args.num_train_epochs))  # 输出训练轮次
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logger.info("  Total train batch size = {}".format(args.train_batch_size * args.gradient_accumulation_steps))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(t_total))  # 输出训练的总步数=(迭代数iter：案例数量/批次大小)*轮次数epoch (34334/128)上取整*3=269*3 = 807steps

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_value = 0
    model.zero_grad()  # 梯度清零，等价于optimizer.zero_grad()
    # 训练轮次
    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:  # 轮次
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")  # 进度条
        for step, batch in enumerate(epoch_iterator):  # 批次
            model.train()  # 模型开始训练
            batch = tuple(t.to(args.device) for t in batch)
            # 输入到模型的编码中，position_ids和head_mask 没有指定
            if args.ccks_no_1:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e_begin_mask": batch[4],
                    "e_end_mask": batch[5],
                }
            elif args.kg_enhance_2:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                    "k_mask": batch[6],
                }
            elif args.kg_enhance_3:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
            elif args.kg_enhance:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            else:
                loss.backward()  # 反向传播参数  输出的loss是一个标量，因此调用backward()不用参数。
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # 迭代模型参数
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()  # 梯度清零，等价于optimizer.zero_grad()
                global_step += 1
        # Log metrics
        # 每轮次结束输出一次
        results = evaluate(args, model, tokenizer)  # 调用评估函数
        for key, value in results.items():  # 遍历所有健值对
            # tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
            # tb_writer.add_scalar("loss", (tr_loss - logging_loss), global_step)
            logging_loss = tr_loss

        # 每轮次评估一次模型，指定性能指标更高了就保存模型到本地
        # args_metrics = 'acc@1'  # Option acc,acc@1,acc@2,acc@3
        if args.do_train:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            print(results)
            if args.metrics in results.keys():
                metrics_value = results[args.metrics]
                if metrics_value > best_value:
                    best_value = metrics_value
                    logger.info("Saving best model(%s : %s) to %s", args.metrics, best_value, args.output_dir)
                    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                    model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir)  # Save a model and its configuration file to a directory, so that it can be re-loaded using the from_pretrained() class method.
                    tokenizer.save_pretrained(args.output_dir)  # 保存分词器
                    # Good practice: save your training arguments together with the trained model
                    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = (args.task_name,)  # 评估任务名称
    eval_outputs_dirs = (args.output_dir,)  # 评估结果输出目录
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):  # zip()方法返回多个列表元素打包成一个个元组的列表
        # eval_dataset：TensorDataset dataset,每个元素是由4个tensor组成的tuple
        # eval_examples: list类型 ,每个元素由InputExample组织数据
        eval_dataset, eval_examples = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)  # 创建路径
        eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        # 定义一个采样器
        eval_sampler = SequentialSampler(eval_dataset)  # 顺序采样元素
        # 定义一个数据载入器，指定数据集和采样器,每次采样的批次大小
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)  # 比如dataset的样本数100，sempler是顺序采样器，batch_size=8，100/8取证等于13个Dataloader
        # multi-gpu eval
        # print(model)
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        masks = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):  # tqdm进度条 从数据装载器Dataloader中获得每一批数据的特征向量
            model.eval()  # 模型进行评估
            batch = tuple(t.to(args.device) for t in batch)  # batch:由4个张量组成的元组
            # 输入到模型的编码中，position_ids和head_mask 没有指定
            with torch.no_grad():  # 评估过程不进行梯度计算
                if args.ccks_no_1:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3],
                        "e_begin_mask": batch[4],
                        "e_end_mask": batch[5],
                    }
                elif args.kg_enhance_2:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3],
                        "e1_mask": batch[4],
                        "e2_mask": batch[5],
                        "k_mask": batch[6],
                    }
                elif args.kg_enhance_3:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3],
                        "e1_mask": batch[4],
                        "e2_mask": batch[5],
                    }
                elif args.kg_enhance:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3],
                        "e1_mask": batch[4],
                        "e2_mask": batch[5],
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3],
                    }
                outputs = model(**inputs)  # 模型的输出是啥，和训练一样，把输入丢进模型，输出得到模型的输出
                tmp_eval_loss, logits = outputs[:2]  # loss (1,) logits (batch_size, config.num_labels)
                # mean():Returns the mean value of all elements in the input tensor. item()将标量tensor换成python number
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()  # logits是model输出的cuda张量，要先detach得到一个新张量，再转cpu向量，再转为numpy向量。detach()是将其从追踪记录中分离出来。
                out_label_ids = inputs["labels"].detach().cpu().numpy()  # 标签的索引也取出，转为numpy向量方便处理
                masks = inputs["attention_mask"].detach().cpu().numpy()  # 用于后面计算token的f1值
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # 每个批次的预测结果追加preds
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                masks = np.append(masks, inputs["attention_mask"].detach().cpu().numpy(), axis=0)  # masks [examples x max_seq_langth]
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            # BertForSequenceClassification,  preds shape (batch_size, config.num_labels)
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(preds, out_label_ids)  # 分类acc的计算指标 定义返回的acc
        elif args.output_mode == "matching":
            # BertForSequenceClassification,  preds shape (batch_size, config.num_labels)
            # preds 输出的是logits,没有softmax
            # print('logits',preds)
            preds = torch.from_numpy(preds)
            t_softmax = F.softmax(preds, dim=1)  # 函数需要输入tenso
            n_softmax = t_softmax.numpy()  # ndarray:(18,2)
            # print('softmax', t_softmax)
            preds_label = np.argmax(n_softmax, axis=1)  # ndarray:(18,)
            # print('preds_label',preds_label)
            result = compute_metrics(preds_label, out_label_ids)  # 分类acc的计算指标 定义返回的acc  dict:1{'acc':0.88}
            results.update(result)  # 把acc 写入eval_results.txt
            # 输出错误案例
            compute_df = error_analysis(preds_label, out_label_ids, n_softmax, eval_examples, args)
            # 计算acc@n
            if args.task_name == 'cckskbqaedpoetoken':
                result = compute_ranking_acc(compute_df, 1, args)
                results.update(result)  # 把acc@n 写入eval_results.txt
                result = compute_ranking_acc(compute_df, 3, args)
                results.update(result)  # 把acc@n 写入eval_results.txt
                result = compute_ranking_acc(compute_df, 5, args)
                results.update(result)  # 把acc@n 写入eval_results.txt
            else:
                result = compute_ranking_acc(compute_df, 1, args)
                results.update(result)  # 把acc@n 写入eval_results.txt
                result = compute_ranking_acc(compute_df, 2, args)
                results.update(result)  # 把acc@n 写入eval_results.txt
                result = compute_ranking_acc(compute_df, 3, args)
                results.update(result)  # 把acc@n 写入eval_results.txt
        else:
            # ner: BertForTokenClassification, preds shape (batch_size, sequence_length, config.num_labels)
            preds = np.argmax(preds, axis=2)  # ndarray:(batch_size x seq_len) 从维度=2的向量中取最大的值，三维向量->二维向量
            # 得到开始标签'B'和'S'的索引,以及'O'的索引，从而得到实体的位置
            begin_ids = []
            notentity_ids = []
            processor = processors[args.task_name]()
            label_list = processor.get_labels(args.data_dir)
            label_map = {label: i for i, label in enumerate(label_list)}  # label_map:{'B':0,'I':1,'O',2,'S':3}
            for k in list(label_map.keys()):
                # if k.startswith("B-entity") or k.startswith("S"):
                if k.startswith("B") or k.startswith("S"):
                    begin_ids.append(label_map[k])
                if k.startswith("O"):
                    notentity_ids.append(label_map[k])
            # 调用命名实体识别的评估函数
            logger.info("预测标签的形状{}".format(preds.shape))
            logger.info("真实标签的形状{}".format(out_label_ids.shape))
            result = ner_f1_new(preds, out_label_ids, masks, begin_ids, notentity_ids)  # ner acc的计算指标 定义返回的acc
        # 不管那种评估任务，评估方法返回都是一个字典，{评估指标名：评估分值}
        # dict.update(dict2) 把字典dict2的键/值对，添加到字典中dict里。
        results.update(result)
        # 执行结果 写入文档
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")  # 输出评估结果到目录
        outputs_eval_results(results, args, output_eval_file)
    return results


def predict(args, model, tokenizer, prefix=""):
    """
    输出模型预测结果，json格式
    :param args:
    :param model:
    :param tokenizer:
    :param prefix:
    :return:
    """
    pred_task = args.task_name  # 评估任务名称
    pred_output_dir = args.output_dir  # 评估结果输出目录
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)
    pred_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    pred_dataset, pred_examples = load_and_cache_examples(args, pred_task, tokenizer, evaluate=True)
    # Note that DistributedSampler samples randomly
    # 定义一个采样器
    pred_sampler = SequentialSampler(pred_dataset)  # 顺序采样元素
    # 定义一个数据载入器，指定数据集和采样器,每次采样的批次大小
    pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=pred_batch_size)  # 比如dataset的样本数100，sempler是顺序采样器，batch_size=8，100/8取证等于13个Dataloader
    logger.info("***** Running prediction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", pred_batch_size)
    preds = None
    out_label_ids = None
    masks = None
    for batch in tqdm(pred_dataloader, desc="Predicting"):  # tqdm进度条 从数据装载器Dataloader中获得每一批数据的特征向量
        model.eval()  # 模型进行评估
        batch = tuple(t.to(args.device) for t in batch)  # batch:由4个张量组成的元组
        # 输入到模型的编码中，position_ids和head_mask 没有指定
        with torch.no_grad():  # 评估过程不进行梯度计算
            if args.ccks_no_1:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e_begin_mask": batch[4],
                    "e_end_mask": batch[5],
                }
            elif args.kg_enhance_2:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                    "k_mask": batch[6],
                }
            elif args.kg_enhance_3:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
            elif args.kg_enhance:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
            outputs = model(**inputs)  # 模型的输出是啥，和训练一样，把输入丢进模型，输出得到模型的输出
            tmp_pred_loss, logits = outputs[:2]  # loss (1,) logits (batch_size, config.num_labels)
        if preds is None:
            preds = logits.detach().cpu().numpy()  # logits是model输出的cuda张量，要先detach得到一个新张量，再转cpu向量，再转为numpy向量。detach()是将其从追踪记录中分离出来。
            out_label_ids = inputs["labels"].detach().cpu().numpy()  # 标签的索引也取出，转为numpy向量方便处理
            masks = inputs["attention_mask"].detach().cpu().numpy()  # 用于后面计算token的f1值
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)  # 每个批次的预测结果追加preds
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            masks = np.append(masks, inputs["attention_mask"].detach().cpu().numpy(), axis=0)  # masks [examples x max_seq_langth]
    predict_result = []
    if args.output_mode == "classification":
        # BertForSequenceClassification,  preds shape (batch_size, config.num_labels)
        preds = np.argmax(preds, axis=1)
        for idx, pred in enumerate(preds):
            json_data = {'id': idx, 'pred': pred}
            predict_result.append(json_data)
    elif args.output_mode == "matching":
        # BertForSequenceClassification,  preds shape (batch_size, config.num_labels)
        preds = torch.from_numpy(preds)
        t_softmax = F.softmax(preds, dim=1)  # 函数需要输入tenso
        n_softmax = t_softmax.numpy()  # ndarray:(18,2)
        preds = np.argmax(n_softmax, axis=1)  # ndarray:(18,)
        for idx, pred in enumerate(preds):
            json_data = {'id': idx, 'pred': pred}
            predict_result.append(json_data)
    else:
        # ner: BertForTokenClassification, preds shape (batch_size, sequence_length, config.num_labels)
        preds = np.argmax(preds, axis=2).tolist()
        processor = processors[args.task_name]()
        labels = processor.get_labels(args.data_dir)
        label_map = {idx: label for idx, label in enumerate(labels)}  # label_map:{'B':0,'I':1,'O',2,'S':3}
        # 每个样本除去特殊标签[CLS][SEP][PAD] 得到列表total_preds，total_golds
        total_preds = []
        total_golds = []
        for i in range(len(preds)):
            # print('preds[i]',preds[i])
            # print('out_label_ids[i]', out_label_ids[i])
            num = sum(masks[i]) - 2  # sum():需要关注的token数，对mask=1的求和，减去一头一尾两个token
            total_preds.append(preds[i][1: 1 + num])  # 减去一头一尾的mask=1的token 预测值列表
            total_golds.append(out_label_ids[i][1: 1 + num])  # 减去一头一尾的mask=1的token 真值列表
        for idx, (pred, gold) in enumerate(zip(total_preds, total_golds)):
            # print('pred', pred)
            pred_tags = [label_map[t] for t in pred]
            gold_tags = [label_map[t] for t in gold]
            # print('pred_tags', pred_tags)
            # print('gold_tags', gold_tags)
            pred_entities = get_entities(pred, label_map, 'bio')
            gold_entities = get_entities(gold, label_map, 'bio')
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
                      'pred_tags': " ".join(pred_tags),
                      'pred_entities': pred_entities,
                      'gold_entities': gold_entities,
                      'hit_flag': hit_flag
                      }
            predict_result.append(json_d)
    # 写出预测结果，json格式
    dump_path = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    dump_jsonl(predict_result, dump_path)


def outputs_eval_results(results: Dict, args, output_eval_file: str):
    """
    把评估结果写到文件
    :param results: 评估结果
    :param args:
    :param output_eval_file:
    :return:
    """
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        writer.write("评估任务(task_name) = %s\n" % args.task_name)
        writer.write("评估数据集(input_test_name) = %s\n" % args.input_test_name)
        writer.write("数据集路径(data_dir) = %s\n" % args.data_dir)
        for key in sorted(results.keys()):
            logger.info("%s = %s\n" % (key, str(results[key])))
            writer.write("%s = %s\n" % (key, str(results[key])))
        writer.write("eval_time = %s\n" % time.asctime(time.localtime(time.time())))  # 执行时间 写入文档
    writer.close()


def outputs_el_results(results: Dict, args, output_eval_file: str):
    """
    把评估结果写到文件
    :param results: 评估结果
    :param args:
    :param output_eval_file:
    :return:
    """
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        writer.write("评估任务(task_name) = 实体链接\n")
        writer.write("评估数据集(input_test_name) = %s\n" % args.input_test_name)
        writer.write("数据集路径(data_dir) = %s\n" % args.data_dir)
        for key in sorted(results.keys()):
            logger.info("%s = %s\n" % (key, str(results[key])))
            writer.write("%s = %s\n" % (key, str(results[key])))
        writer.write("eval_time = %s\n" % time.asctime(time.localtime(time.time())))  # 执行时间 写入文档
    writer.close()


def main():
    if os.path.exists(args.output_dir) and any([x for x in os.listdir(args.output_dir) if x.find("bin") > -1]) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("输出路径 ({}) 已经存在且不为空. 使用--overwrite_output_dir解决".format(args.output_dir))
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 if torch.cuda.is_available() else 0
    args.device = device
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    # Set seed
    set_seed(args)
    if args.task_name not in processors:  # 如果传入参数不在 processors
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()  # 初始化 数据处理器
    args.output_mode = output_modes[args.task_name]  # 初始化 输出的模式 classification,matching 还是 ner
    label_list = processor.get_labels(args.data_dir)
    num_labels = len(label_list)
    # Load pretrained model and tokenizer and config
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  # 从预定义的模型类型中 获得对应的分词起，配置文件，和模型文件   "ner": (BertConfig, BertForTokenClassification, BertTokenizer)
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    print(model)
    model.to(args.device)  # 放到cpu或者gpu中
    # Training
    if args.do_train:
        train_dataset, train_examples = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        # 开始训练
        start_train = time.time()
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)  # 开始训练
        # 结束训练 记录训练时长
        dur = time.time() - start_train
        print("训练用时： %.3f 秒" % dur)  # 格式化输出
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            # 评估开始
            start = time.time()
            result = evaluate(args, model, tokenizer, prefix=prefix)  # 调用评估函数
            # 评估结束
            eval_time = time.time() - start
            logger.info("评估用时： %.3f 秒" % eval_time)  # 格式化输出
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            logger.info("评估结果：{}".format(result))
    # Predicting
    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            # 预测开始
            start = time.time()
            predict(args, model, tokenizer, prefix=prefix)  # 调用评估函数
            pred_time = time.time() - start
            logger.info("预测用时： %.3f 秒" % pred_time)  # 格式化输出
    # 最后，计算实体链接的得分
    if args.compute_pipeline_el_metric:
        # 分别从两个任务的模型输出路径读取评估结果，然后计算得到el的得分
        md_metric = csv_reader_static(os.path.join(args.pipeline_el_md_dir, "eval_results.txt"))
        print('md_metric', md_metric, os.path.join(args.pipeline_el_md_dir, "eval_results.txt"))
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
        ed_metric = csv_reader_static(os.path.join(args.pipeline_el_ed_dir, "eval_results.txt"))
        print('ed_metric', ed_metric, os.path.join(args.pipeline_el_ed_dir, "eval_results.txt"))
        if args.task_name == 'cckskbqaedpoetoken':
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
                "el_recall@1": el_r1,
                "el_recall@3": el_r3,
                "el_recall@5": el_r5,
                "el_f1": el_f1,
                "el_p": el_p,
                "el_r": el_r,
            }
        else:
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
        outputs_el_results(results, args, os.path.join(args.output_dir, "eval_results.txt"))


if __name__ == "__main__":
    main()
