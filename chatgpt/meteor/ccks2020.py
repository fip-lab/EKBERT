import nltk
import json
from nltk.corpus import wordnet

ccks2020_kbqa_test_model_gene_answer_address = '../data/ccks2020_answer.txt'
ccks2020_kbqa_test_knowques_compansw_address = '../../mt5/data/ccks2020_kbqa_test_knowques_compansw.json'

with open(ccks2020_kbqa_test_knowques_compansw_address, 'r', encoding='utf-8') as f:
    compansw = json.load(f)
with open(ccks2020_kbqa_test_model_gene_answer_address, 'r', encoding='utf-8') as f:
    geneansw_temp = f.readlines()
geneansw = []
for item in geneansw_temp:
    geneansw.append(item.replace('\n', ''))

total_merteor_score = 0
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
    merteor_score = nltk.translate.meteor_score.single_meteor_score(list1, list2)
    total_merteor_score += merteor_score
avg_merteor_score = total_merteor_score / len(compansw)
print(avg_merteor_score)