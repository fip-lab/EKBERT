#! /bin/bash

cd ..
cd ..

cd mt5

cd data_process
python nlpcc_process.py

cd ..
cd train_code
python train_model_nlpcc.py

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