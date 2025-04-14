import json
import re
from tqdm import tqdm

s_po_address = '../data/nlpcc2018_s_po.json'
s_ques_answ_no_temp_address = '../data/nlpcc2018_ques_answ_no_temp.json'
know_ques_answ_no_temp_prompt_address = '../data_no_temp_prompt/nlpcc2018_know_ques_answ_no_temp_prompt.json'

with open(s_po_address, 'r', encoding='utf-8') as f:
    s_po_list = json.load(f)
with open(s_ques_answ_no_temp_address, 'r', encoding='utf-8') as f:
    s_ques_answ_list = json.load(f)
input = []
for i in tqdm(s_ques_answ_list):
    for j in s_po_list:
        input_temp = []
        if i[0] == j[0]:
            know_ques = j[1] + i[1]
            answ = i[2]
            input_temp.append(know_ques)
            input_temp.append(answ)
            input.append(input_temp)
with open(know_ques_answ_no_temp_prompt_address, 'w', encoding='utf-8') as f:
    json.dump(input, f, ensure_ascii=False)
