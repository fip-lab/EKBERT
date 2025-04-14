#! /bin/bash
cd ..
cd ..
cd src
# md 任务
md_train_name='../output/finetune_model/nlpcc_kbqa_bbkbqa_md'
md_model_dir='../output/finetune_model/nlpcc_kbqa_bbkbqa_md'
md_data_dir='../src/data/processed/nlpcc_kbqa/md'
# ed任务
ed_train_name='../output/finetune_model/nlpcc_kbqa_bbkbqa'
ed_model_dir='../output/finetune_model/nlpcc_kbqa_bbkbqa'
ed_data_dir='../src/data/processed/nlpcc_kbqa/ed'

device=0
# md任务预测
CUDA_VISIBLE_DEVICES=${device} python main_savebestacc.py \
  --model_type=ner \
  --task_name=nlpccmd \
  --data_dir=${md_data_dir} \
  --model_name_or_path=${md_model_dir} \
  --output_dir=${md_train_name} \
  --max_seq_length=100 \
  --do_eval \
  --per_gpu_eval_batch_size=500 \
  --input_test_name='test.tsv' \
  --do_predict

# 读md任务预测正确结果，生成筛选后的ed数据
CUDA_VISIBLE_DEVICES=${device} python run_entity_linking.py \
  --model_name='bbkbqa' \
  --load_md_predict_data_path=${md_model_dir} \
  --dump_ed_predict_data_path=${ed_data_dir}

# 读ed数据进行预测，得到el任务结果
CUDA_VISIBLE_DEVICES=${device} python main_savebestacc.py \
  --model_type=bert_seq_cls \
  --task_name=nlpcced \
  --data_dir=${ed_data_dir} \
  --model_name_or_path=${ed_model_dir} \
  --output_dir=${ed_train_name} \
  --max_seq_length=100 \
  --do_eval \
  --per_gpu_eval_batch_size=200 \
  --input_test_name='predict.tsv' \
  --compute_pipeline_el_metric \
  --pipeline_el_md_dir=${md_model_dir} \
  --pipeline_el_ed_dir=${ed_model_dir}
