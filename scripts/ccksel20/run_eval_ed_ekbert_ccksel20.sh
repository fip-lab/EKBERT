#! /bin/bash

train_name='../output/finetune_model/ccks_el20_ekbert_ed'
model_dir='../output/finetune_model/ccks_el20_ekbert_ed'
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
  --do_eval \
  --per_gpu_eval_batch_size=16 \
  --input_test_name=test.tsv \
  --kg_enhance
