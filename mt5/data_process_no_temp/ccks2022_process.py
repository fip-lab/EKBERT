import json
import re
import random

raw_train_address = '../../src/data/raw/ccks_kbqa_2022/train.txt'


def ccks2022_kbqa_process(address):
    with open(raw_train_address, 'r', encoding='utf-8') as f:
        train_data = f.read().splitlines()
    train_triples = []  # 存放所有未处理的问题和答案
    temp_train_list = []
    for item in train_data:
        if item != '':
            temp_train_list.append(item)
        else:
            train_triples.append(temp_train_list)
            temp_train_list = []
    # print(train_triples)
    train_triples_pro = []
    for i in train_triples:
        p = re.compile(r'<.*?>')
        l = p.findall(i[1])
        # print(p.findall(i[1]))
        if len(l) == 2:
            train_triples_pro.append(i)  # 存放所有涉及一跳知识的问题及答案
    # print(train_triples_pro)
    total_know_ques_compansw_list = []
    for j in train_triples_pro:
        ques_p = re.compile(r'q.*:(.*)')
        ques = ques_p.findall(j[0])[0]
        know_p = re.compile(r'<.*?>')
        know_sp = know_p.findall(j[1])
        know_s = know_sp[0].replace('<', '').replace('>', '')
        know_p = know_sp[1].replace('<', '').replace('>', '')
        know_o = j[2].replace('<', '').replace('>', '').replace('\t', ',')
        know_ques = '知识:' + know_p + '是' + know_o + '问题:' + ques
        compansw = know_o
        if compansw[-1] == ',':
            compansw = compansw[:len(compansw) - 1]
        else:
            compansw = compansw
        temp_tumple = (know_ques, compansw)
        total_know_ques_compansw_list.append(temp_tumple)  # len=4514
    # print(len(total_know_ques_compansw_list))
    return total_know_ques_compansw_list


ccks2022_kbqa_train_knowques_compansw_address = '../data_no_temp/ccks2022_kbqa_train_knowques_compansw.json'
ccks2022_kbqa_dev_knowques_compansw_address = '../data_no_temp/ccks2022_kbqa_dev_knowques_compansw.json'
ccks2022_kbqa_test_knowques_compansw_address = '../data_no_temp/ccks2022_kbqa_test_knowques_compansw.json'

total_know_ques_compansw = ccks2022_kbqa_process(raw_train_address)
random.seed(42)
random.shuffle(total_know_ques_compansw)
ccks2022_train_data = total_know_ques_compansw[:round(0.7 * len(total_know_ques_compansw))]
ccks2022_dev_data = total_know_ques_compansw[
               round(0.7 * len(total_know_ques_compansw)):round(0.9 * len(total_know_ques_compansw))]
ccks2022_test_data = total_know_ques_compansw[round(0.9 * len(total_know_ques_compansw)):]

if __name__ == '__main__':
    with open(ccks2022_kbqa_train_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_train_data, f, ensure_ascii=False)
    with open(ccks2022_kbqa_dev_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_dev_data, f, ensure_ascii=False)
    with open(ccks2022_kbqa_test_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2022_test_data, f, ensure_ascii=False)