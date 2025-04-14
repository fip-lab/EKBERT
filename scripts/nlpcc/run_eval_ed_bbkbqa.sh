#! /bin/bash

train_name='../output/finetune_model/nlpcc_kbqa_bbkbqa'
model_dir='../output/finetune_model/nlpcc_kbqa_bbkbqa'
data_dir='../src/data/processed/nlpcc_kbqa/ed'
device=0

cd ..
cd ..
cd src

CUDA_VISIBLE_DEVICES=${device} python main_savebestacc.py \
  --model_type=bert_seq_cls \
  --task_name=nlpcced \
  --data_dir=${data_dir} \
  --model_name_or_path=${model_dir} \
  --output_dir=${train_name} \
  --max_seq_length=100 \
  --do_eval \
  --per_gpu_eval_batch_size=500 \
  --input_test_name='test.tsv'
