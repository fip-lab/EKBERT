#! /bin/bash

cd ..
cd ..

cd gpt2
cd data_process
python nlpcc_single.py
python nlpcc_single_inp.py

cd ..
cd train_model
python train_code_nlpcc.py

cd ..
cd gene_answer
python nlpcc.py

cd ..
cd assess

cd bleu
python nlpcc.py

cd ..
cd rouge
python nlpcc.py

cd ..
cd meteor
python nlpcc.py

cd ..
cd perplexity
python nlpcc.py