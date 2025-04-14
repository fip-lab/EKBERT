import json
import re
import random
from tqdm import tqdm

nlpcc_2018_raw_train_address = '../../../src/data/raw/nlpcc_kbqa_2018/nlpcc2018.kbqg.train'
nlpcc_2018_raw_test_address = '../../../src/data/raw/nlpcc_kbqa_2018/nlpcc2018.kbqg.test'
nlpcc_2018_ques_answ_address = '../../data/nlpcc2018_ques_answ.json'


def ques_answ(raw_address):
    with open(raw_address, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()
    # ==================================================
    # for idx, item in enumerate(nlpcc_2018_raw_train_data):
    #     if idx < 100:
    #         print(item)
    raw_answ_ques_list = []
    temp_list = []
    for item in data:
        if item != '==================================================':
            temp_list.append(item)
        else:
            raw_answ_ques_list.append(temp_list)
            temp_list = []
    ques_answ_list = []
    for item1 in raw_answ_ques_list:
        temp_list1 = []
        raw_answ = item1[0]
        raw_ques = item1[1]
        id_answ = raw_answ.split('\t')
        id_ques = raw_ques.split('\t')
        triple = id_answ[1].split('|||')
        ques = id_ques[1].replace(' ','')
        answ = triple[0].replace(' ', '') + '的' + triple[1].replace(' ', '') + '是' + triple[2].replace(' ', '') + '。'
        temp_list1.append(triple[0].replace(' ', ''))
        temp_list1.append(ques)
        temp_list1.append(answ)
        ques_answ_list.append(temp_list1)
    return ques_answ_list

if __name__ == '__main__':
    ques_answ_data = ques_answ(nlpcc_2018_raw_train_address) + ques_answ(nlpcc_2018_raw_test_address)
    ques_answ_data_no_blank = []
    for item in ques_answ_data:
        if item[0]!='':
            ques_answ_data_no_blank.append(item)
    # print(len(ques_answ_data_no_blank))
    with open(nlpcc_2018_ques_answ_address, 'w', encoding='utf-8') as f:
        json.dump(ques_answ_data_no_blank, f, ensure_ascii=False)