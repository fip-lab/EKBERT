from rouge import Rouge
import json
import jieba

kgclue_kbqa_test_model_gene_answer_address = '../../mt5_model_gene_answer/kgclue_kbqa_test_model_gene_answer.json'
kgclue_kbqa_test_knowques_compansw_address = '../../data/kgclue_test_data.json'


def get_rouge_r_test(model_answer_address_test, comp_answ_address_test):
    with open(model_answer_address_test, 'r', encoding='utf-8') as f:
        model_answer_test = json.load(f)
    with open(comp_answ_address_test, 'r', encoding='utf-8') as f:
        comp_answ_test = json.load(f)
    comp_answ_test_list = []
    for item in comp_answ_test:
        comp_answ_test_list.append(item[1])
    assert len(model_answer_test) == len(comp_answ_test_list)
    total_rouge1_scores = 0
    total_rouge2_scores = 0
    total_rouge3_scores = 0
    for item1, item2 in zip(model_answer_test, comp_answ_test_list):
        hypo = item1
        refer = item2
        hypo = ' '.join(jieba.cut(hypo))
        refer = ' '.join(jieba.cut(refer))
        rouge = Rouge()
        scores = rouge.get_scores(hypo, refer)
        rouge1_r = scores[0]['rouge-1']['r']
        rouge2_r = scores[0]['rouge-2']['r']
        rouge3_r = scores[0]['rouge-l']['r']
        total_rouge1_scores += rouge1_r
        total_rouge2_scores += rouge2_r
        total_rouge3_scores += rouge3_r
    avg_rouge1_r_scores = total_rouge1_scores / (len(model_answer_test))
    avg_rouge2_r_scores = total_rouge2_scores / (len(model_answer_test))
    avg_rouge3_r_scores = total_rouge3_scores / (len(model_answer_test))
    # print('平均rouge_1分数为：', avg_rouge_1_scores)
    # print('平均rouge_2分数为：', avg_rouge_2_scores)
    # print('平均rouge_3分数为：', avg_rouge_3_scores)
    return avg_rouge1_r_scores, avg_rouge2_r_scores, avg_rouge3_r_scores


def get_rouge_f_test(model_answer_address_test, comp_answ_address_test):
    with open(model_answer_address_test, 'r', encoding='utf-8') as f:
        model_answer_test = json.load(f)
    with open(comp_answ_address_test, 'r', encoding='utf-8') as f:
        comp_answ_test = json.load(f)
    comp_answ_test_list = []
    for item in comp_answ_test:
        comp_answ_test_list.append(item[1])
    assert len(model_answer_test) == len(comp_answ_test_list)
    total_rouge1_scores = 0
    total_rouge2_scores = 0
    total_rouge3_scores = 0
    for item1, item2 in zip(model_answer_test, comp_answ_test_list):
        hypo = item1
        refer = item2
        hypo = ' '.join(jieba.cut(hypo))
        refer = ' '.join(jieba.cut(refer))
        rouge = Rouge()
        scores = rouge.get_scores(hypo, refer)
        rouge1_f = scores[0]['rouge-1']['f']
        rouge2_f = scores[0]['rouge-2']['f']
        rouge3_f = scores[0]['rouge-l']['f']
        total_rouge1_scores += rouge1_f
        total_rouge2_scores += rouge2_f
        total_rouge3_scores += rouge3_f
    avg_rouge1_f_scores = total_rouge1_scores / (len(model_answer_test))
    avg_rouge2_f_scores = total_rouge2_scores / (len(model_answer_test))
    avg_rouge3_f_scores = total_rouge3_scores / (len(model_answer_test))
    # print('平均rouge_1分数为：', avg_rouge_1_scores)
    # print('平均rouge_2分数为：', avg_rouge_2_scores)
    # print('平均rouge_3分数为：', avg_rouge_3_scores)
    return avg_rouge1_f_scores, avg_rouge2_f_scores, avg_rouge3_f_scores


def get_rouge_p_test(model_answer_address_test, comp_answ_address_test):
    with open(model_answer_address_test, 'r', encoding='utf-8') as f:
        model_answer_test = json.load(f)
    with open(comp_answ_address_test, 'r', encoding='utf-8') as f:
        comp_answ_test = json.load(f)
    comp_answ_test_list = []
    for item in comp_answ_test:
        comp_answ_test_list.append(item[1])
    assert len(model_answer_test) == len(comp_answ_test_list)
    total_rouge1_scores = 0
    total_rouge2_scores = 0
    total_rouge3_scores = 0
    for item1, item2 in zip(model_answer_test, comp_answ_test_list):
        hypo = item1
        refer = item2
        hypo = ' '.join(jieba.cut(hypo))
        refer = ' '.join(jieba.cut(refer))
        rouge = Rouge()
        scores = rouge.get_scores(hypo, refer)
        rouge1_p = scores[0]['rouge-1']['p']
        rouge2_p = scores[0]['rouge-2']['p']
        rouge3_p = scores[0]['rouge-l']['p']
        total_rouge1_scores += rouge1_p
        total_rouge2_scores += rouge2_p
        total_rouge3_scores += rouge3_p
    avg_rouge1_p_scores = total_rouge1_scores / (len(model_answer_test))
    avg_rouge2_p_scores = total_rouge2_scores / (len(model_answer_test))
    avg_rouge3_p_scores = total_rouge3_scores / (len(model_answer_test))
    # print('平均rouge_1分数为：', avg_rouge_1_scores)
    # print('平均rouge_2分数为：', avg_rouge_2_scores)
    # print('平均rouge_3分数为：', avg_rouge_3_scores)
    return avg_rouge1_p_scores, avg_rouge2_p_scores, avg_rouge3_p_scores


kgclue_rouge1_r_scores, kgclue_rouge2_r_scores, kgclue_rouge3_r_scores = get_rouge_r_test(
    kgclue_kbqa_test_model_gene_answer_address,
    kgclue_kbqa_test_knowques_compansw_address)
kgclue_rouge1_f_scores, kgclue_rouge2_f_scores, kgclue_rouge3_f_scores = get_rouge_f_test(
    kgclue_kbqa_test_model_gene_answer_address,
    kgclue_kbqa_test_knowques_compansw_address)
kgclue_rouge1_p_scores, kgclue_rouge2_p_scores, kgclue_rouge3_p_scores = get_rouge_p_test(
    kgclue_kbqa_test_model_gene_answer_address,
    kgclue_kbqa_test_knowques_compansw_address)
print('平均rouge1_r分数为：', kgclue_rouge1_r_scores)
print('kgclue的平均rouge2_r分数为：', kgclue_rouge2_r_scores)
print('平均rouge3_r分数为：', kgclue_rouge3_r_scores)
#
# print('平均rouge1_f分数为：', kgclue_rouge1_f_scores)
# print('平均rouge2_f分数为：', kgclue_rouge2_f_scores)
# print('平均rouge3_f分数为：', kgclue_rouge3_f_scores)
#
# print('平均rouge1_p分数为：', kgclue_rouge1_p_scores)
# print('平均rouge2_p分数为：', kgclue_rouge2_p_scores)
# print('平均rouge3_p分数为：', kgclue_rouge3_p_scores)
