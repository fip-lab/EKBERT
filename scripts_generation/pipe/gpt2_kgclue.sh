#! /bin/bash

cd ..
cd ..

cd gpt2
cd data_process
python kgclue_single.py
python kgclue_single_inp.py

cd ..
cd train_model
python train_code_kgclue.py

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