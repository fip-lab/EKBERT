import json
import re
import random
from tqdm import tqdm

raw_train_address = '../../src/data/raw/ccks_kbqa_2020/ccks2020_kbqa_train.txt'
raw_dev_address = '../../src/data/raw/ccks_kbqa_2020/ccks2020_kbqa_dev.txt'


#  将ccks2020中所有涉及一跳知识的问题的答案进行整理
def one_hop_raw_process(address):
    with open(address, 'r', encoding='utf-8') as f:
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
    total_ques_compansw = []
    for item in train_triples_pro:
        ques_compansw = []
        ques_p = re.compile(r'q.*:(.*)')
        ques = ques_p.findall(item[0])[0]
        # print(ques)
        know_p = re.compile(r'<.*?>')
        know_sp = know_p.findall(item[1])  # [<.*>,<.*>]
        know_s = know_sp[0].replace('<', '').replace('>', '')
        # know_s = know_s.replace('<','').replace('>','')
        know_p = know_sp[1].replace('<', '').replace('>', '')
        know_o = item[2].replace('<', '').replace('>', '').replace('\t', ',')
        compansw = know_o  # p是o
        if compansw[-1] == ',':
            compansw = compansw[:len(compansw) - 1]
        else:
            compansw = compansw
        # knowques = '知识:'+know + ',' +'问题:'+ ques  # 知识:,问题:
        ques_compansw.append(ques)
        ques_compansw.append(compansw)
        total_ques_compansw.append(ques_compansw)
        # print(total_ques_know_ques)
    return total_ques_compansw


#  对包含所有一跳知识的问题和知识进行整理
pro_train_address = '../../src/data/processed/ccks_kbqa_2020/ed_ekbert/train.tsv'
pro_dev_address = '../../src/data/processed/ccks_kbqa_2020/ed_ekbert/dev.tsv'
pro_test_address = '../../src/data/processed/ccks_kbqa_2020/ed_ekbert/test.tsv'


def one_hop_pro_process(address):
    with open(address, 'r', encoding='utf-8') as f:
        p = f.read().splitlines()
    total_list = []
    for item in p:
        item = item.split('\t')
        if item[3] == '1':
            know = re.sub(r'(<e2>.*</e2>)', '', item[2]).replace('|', ',').replace(':', '是')
            if know != '':
                temp_list = []
                ques = item[1].replace('<e1>', '').replace('</e1>', '')
                know_ques = '知识:' + know + '问题:' + ques + ',' + '答案:'
                temp_list.append(ques)
                temp_list.append(know_ques)
                total_list.append(temp_list)
    return total_list


ccks2020_kbqa_train_knowques_compansw_address = '../data_no_temp/ccks2020_kbqa_train_knowques_compansw.json'
ccks2020_kbqa_dev_knowques_compansw_address = '../data_no_temp/ccks2020_kbqa_dev_knowques_compansw.json'
ccks2020_kbqa_test_knowques_compansw_address = '../data_no_temp/ccks2020_kbqa_test_knowques_compansw.json'

train_ques_compansw = one_hop_raw_process(raw_train_address)
dev_ques_compansw = one_hop_raw_process(raw_dev_address)
total_ques_compansw = train_ques_compansw + dev_ques_compansw
train_ques_know = one_hop_pro_process(pro_train_address)
dev_ques_know = one_hop_pro_process(pro_dev_address)
test_ques_know = one_hop_pro_process(pro_test_address)
total_ques_know = train_ques_know + dev_ques_know + test_ques_know
total_list = []  # 将相同问题的know_ques和compansw进行整理
for item1 in tqdm(total_ques_know):
    ques1 = item1[0]
    for item2 in total_ques_compansw:
        ques2 = item2[0]
        if ques1 == ques2:
            temp_tumple = (item1[1], item2[1])
            total_list.append(temp_tumple)  # len=2128
random.seed(42)
random.shuffle(total_list)
ccks2020_train_data = total_list[:round(0.7 * len(total_list))]  # round()四舍五入
ccks2020_dev_data = total_list[round(0.7 * len(total_list)):round(0.9 * len(total_list))]
ccks2020_test_data = total_list[round(0.9 * len(total_list)):]

if __name__ == '__main__':
    # print(total_list)
    with open(ccks2020_kbqa_train_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2020_train_data, f, ensure_ascii=False)
    with open(ccks2020_kbqa_dev_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2020_dev_data, f, ensure_ascii=False)
    with open(ccks2020_kbqa_test_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(ccks2020_test_data, f, ensure_ascii=False)
