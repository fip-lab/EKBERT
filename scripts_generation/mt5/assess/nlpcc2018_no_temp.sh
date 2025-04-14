#! /bin/bash

cd ../../..

cd mt5
cd assess_no_temp

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