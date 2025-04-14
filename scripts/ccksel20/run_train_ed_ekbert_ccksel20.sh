#! /bin/bash

train_name='../output/finetune_model/ccks_el20_ekbert_ed'
model_dir='../pretrained_model/bert-base-chinese-pytorch'
data_dir='../src/data/processed/ccks_el_2020/ed_ekbert'
device=0
cd ..
cd ..
cd src

CUDA_VISIBLE_DEVICES=${device} python main_savebestacc.py \
  --model_type=bert_seq_e_enhance_concate \
  --task_name=cckseledpoetoken \
  --data_dir=${data_dir} \
  --model_name_or_path=${model_dir} \
  --output_dir=${train_name} \
  --max_seq_length=380 \
  --do_train \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=5e-5 \
  --weight_decay=1e-2 \
  --warmup_steps=1000 \
  --num_train_epochs=1 \
  --overwrite_output_dir \
  --evaluate_during_training \
  --input_test_name=dev.tsv \
  --kg_enhance # 如果是用bert_seq_e_enhance这个模型必须加上这个参数为True
