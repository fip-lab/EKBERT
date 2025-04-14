import json
import re
import random
from tqdm import tqdm

kgclue_train_address = '../../src/data/raw/kgclue/train.json'
kgclue_dev_address = '../../src/data/raw/kgclue/dev.json'
kgclue_s_ques_answ_address = '../data_no_temp/kgclue_s_ques_answ_no_temp.json'

def ques_answ(raw_address):
    with open(raw_address, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()
    s_ques_answ_list = []
    for item in tqdm(data):
        s_ques_answ_temp_list = []
        item = eval(item)
        ques = item['question']
        raw_answ = item['answer']
        answ_list = raw_answ.replace(' ','').split('|||')
        s = answ_list[0]
        answ = answ_list[2]
        s_ques_answ_temp_list.append(s)
        s_ques_answ_temp_list.append(ques)
        s_ques_answ_temp_list.append(answ)
        s_ques_answ_list.append(s_ques_answ_temp_list)
    return s_ques_answ_list

if __name__ == '__main__':
    ques_answ_data = ques_answ(kgclue_train_address) + ques_answ(kgclue_dev_address)
    with open(kgclue_s_ques_answ_address, 'w', encoding='utf-8') as f:
        json.dump(ques_answ_data, f, ensure_ascii=False)

print('kgclue_s_ques_answ_no_temp数据处理完毕！')