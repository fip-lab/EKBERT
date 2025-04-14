#! /bin/bash
train_name='../output/finetune_model/ccks_el_sota_ed'
model_dir='../pretrained_model/bert-base-chinese-pytorch'
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
  --do_train \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=100 \
  --learning_rate=5e-5 \
  --weight_decay=1e-2 \
  --warmup_steps=3000 \
  --num_train_epochs=1 \
  --overwrite_output_dir \
  --evaluate_during_training \
  --input_test_name=dev_baseline.tsv \
  --ccks_no_1
