#! /bin/bash

cd ..
cd ..

cd mt5
cd train_code
python train_model_nlpcc2018.py

cd ..
cd gene_answer
python nlpcc2018.py

cd ..
cd assess

cd bleu
python nlpcc2018.py

cd ..
cd rouge
python nlpcc2018.py

cd ..
cd meteor
python nlpcc2018.py

cd ..
cd perplexity
python nlpcc2018.py