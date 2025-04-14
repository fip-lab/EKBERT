#! /bin/bash

cd ../../..

cd mt5
cd assess_no_temp

cd bleu
python kgclue.py

cd ..
cd meteor
python kgclue.py

cd ..
cd perplexity
python kgclue.py

cd ..
cd rouge
python kgclue.py