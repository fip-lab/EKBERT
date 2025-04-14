#! /bin/bash

train_name='../output/finetune_model/ccks_el_sota_ed'
model_dir='../output/finetune_model/ccks_el_sota_ed'
data_dir='../src/data/processed/ccks_el_2019/ed_sota'
device=0

cd ..
cd ..
cd src

CUDA_VISIBLE_DEVICES=${device} python main_savebestacc.py \
  --model_type=ccks_no_1 \
  --task_name=cckselbaseline \
  --data_dir=${data_dir} \
  --model_name_or_path=${model_dir} \
  --output_dir=${train_name} \
  --max_seq_length=380 \
  --do_eval \
  --per_gpu_eval_batch_size=100 \
  --input_test_name='test_baseline.tsv' \
  --ccks_no_1 # 如果是用bert_seq_e_enhance这个模型必须加上这个参数为True
