#! /bin/bash

cd ../../..

cd chatgpt

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