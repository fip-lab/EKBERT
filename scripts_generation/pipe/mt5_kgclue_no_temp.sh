#! /bin/bash

cd ..
cd ..

cd mt5
cd data_process_no_temp
python kgclue_s_ques_answ_no_temp.py
python kgclue_know_ques_answ_no_temp.py
python kgclue_data_no_temp_split.py

cd ..
cd train_code_no_temp
python train_model_kgclue_no_temp.py

cd ..
cd gene_answer_no_temp
python kgclue.py

cd ..
cd assess_no_temp

cd bleu
python kgclue.py

cd ..
cd rouge
python kgclue.py

cd ..
cd meteor
python kgclue.py

cd ..
cd perplexity
python kgclue.py