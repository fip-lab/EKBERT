#! /bin/bash

cd ..
cd ..

cd mt5
cd data_process
cd kgclue
python kgclue_know.py
python kgclue_s_ques_answ.py
python kgclue_know_ques_answ.py
python kgclue_data_split.py

cd ..
cd ..
cd train_code
python train_model_kgclue.py

cd ..
cd gene_answer
python kgclue.py

cd ..
cd assess

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