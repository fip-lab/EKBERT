import json
import re
from tqdm import tqdm

s_po_address = '../../data/ccks2022_s_po.json'
s_ques_answ_address = '../../data/ccks2022_s_ques_answ.json'
know_ques_answ_address = '../../data/ccks2022_know_ques_answ_new.json'

with open(s_po_address, 'r', encoding='utf-8') as f:
    s_po_list = json.load(f)
with open(s_ques_answ_address, 'r', encoding='utf-8') as f:
    s_ques_answ_list = json.load(f)
input = []
for i in tqdm(s_ques_answ_list):
    for j in s_po_list:
        input_temp = []
        if i[0]==j[0]:
            know_ques = '知识:' + j[1] + '问题:' + i[1] + ',' + '答案:'
            answ = i[2]
            input_temp.append(know_ques)
            input_temp.append(answ)
            input.append(input_temp)
with open(know_ques_answ_address, 'w', encoding='utf-8') as f:
    json.dump(input, f, ensure_ascii=False)