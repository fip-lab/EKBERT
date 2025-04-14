#! /bin/bash

cd ../../..

cd chatgpt

cd bleu
python nlpcc2018.py

cd ..
cd meteor
python nlpcc2018.py

cd ..
cd perplexity
python nlpcc2018.py

cd ..
cd rouge
python nlpcc2018.py