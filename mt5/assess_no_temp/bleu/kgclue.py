import json
import jieba
from nltk.translate.bleu_score import sentence_bleu


kgclue_kbqa_test_model_gene_answer_no_temp_address = '../../mt5_model_gene_answer_no_temp/kgclue_kbqa_test_model_gene_answer_no_temp.json'
kgclue_kbqa_test_knowques_compansw_no_temp_address = '../../data_no_temp/kgclue_test_data_no_temp.json'

def bleu_score(model_answer_address_test, comp_answ_address_test):
    # idx = 0
    with open(model_answer_address_test, 'r', encoding='utf-8') as f:
        model_answer = json.load(f)
    with open(comp_answ_address_test, 'r', encoding='utf-8') as f:
        comp_answ_test = json.load(f)
    comp_answ_test_templist = []
    for i in comp_answ_test:
        comp_answ_test_templist.append(i[1])
    total_scores = 0
    # print(comp_answ_test_templist)
    # comp_answ_test_templist = []
    # for key, value in comp_answ_test.items():
    #     # if idx < 20:
    #     comp_answ_test_templist.append(value)
    # idx =idx + 1
    assert len(model_answer) == len(comp_answ_test_templist)
    for senten1, senten2 in zip(model_answer, comp_answ_test_templist):
        # print(senten1, senten2)
        senten1 = jieba.cut(senten1)
        senten1 = ' '.join(senten1).split(' ')  # jieba分词完后join使其分开，‘文字1 文字2’，再split使其‘文字1’ ‘文字2’
        # print(senten1)
        senten2 = jieba.cut(senten2)
        senten2 = ' '.join(senten2).split(' ')
        senten2_templist = []
        senten2_templist.append(senten2)  # 这么做的原因是bleu的refer（1参），必须是[['文字1', '文字2'...]]类型
        # print(senten2)
        score = sentence_bleu(senten2_templist, senten1, weights=(1, 0, 0, 0))
        total_scores = total_scores + score
    total_scores = total_scores / (len(model_answer))
    return total_scores


nlpcc_test_bleu_scores = bleu_score(kgclue_kbqa_test_model_gene_answer_no_temp_address,
                                    kgclue_kbqa_test_knowques_compansw_no_temp_address)
print('kgclue_no_temp的bleu值为：', nlpcc_test_bleu_scores)