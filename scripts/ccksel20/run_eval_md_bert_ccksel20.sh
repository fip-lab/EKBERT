#! /bin/bash

train_name='../output/finetune_model/ccks_el20_bert_md'
model_dir='../output/finetune_model/ccks_el20_bert_md'
data_dir='../src/data/processed/ccks_el_2020/md'
device=0

cd ..
cd ..
cd src

CUDA_VISIBLE_DEVICES=${device} python main_savebestacc.py \
  --model_type=ner \
  --task_name=nlpccmd \
  --data_dir=${data_dir} \
  --model_name_or_path=${model_dir} \
  --output_dir=${train_name} \
  --max_seq_length=100 \
  --do_eval \
  --per_gpu_eval_batch_size=500 \
  --input_test_name='test.tsv'
