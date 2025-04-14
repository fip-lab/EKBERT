import nltk
import json
from nltk.corpus import wordnet

kgclue_kbqa_test_knowques_compansw_address = '../../mt5/data/kgclue_test_data.json'
kgclue_kbqa_test_model_gene_answer_address = '../data/kgclue_answer.txt'

with open(kgclue_kbqa_test_knowques_compansw_address, 'r', encoding='utf-8') as f:
    compansw = json.load(f)
with open(kgclue_kbqa_test_model_gene_answer_address, 'r', encoding='utf-8') as f:
    geneansw_temp = f.readlines()
geneansw = []
for item in geneansw_temp:
    geneansw.append(item.replace('\n', ''))

total_meteor_score = 0
assert len(compansw) == len(geneansw)
for idx in range(len(compansw)):
    str1 = compansw[idx][1]
    str2 = geneansw[idx]
    list1 = []
    list2 = []
    for word in str1:
        list1.append(word)
    for word in str2:
        list2.append(word)
    meteor_score = nltk.translate.meteor_score.single_meteor_score(list1, list2)
    total_meteor_score += meteor_score
avg_meteor_score = total_meteor_score / len(compansw)
print('kgclue的meteor值为：', avg_meteor_score)