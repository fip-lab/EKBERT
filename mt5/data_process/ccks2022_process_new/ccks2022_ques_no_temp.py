import json
import re
from tqdm import tqdm

train_address = '../../../src/data/raw/ccks_kbqa_2022/train.txt'
s_ques_answ_no_temp_address = '../../data/ccks2022_s_ques_answ_no_temp.json'

with open(train_address, 'r', encoding='utf-8') as f:
    train_raw = f.read().splitlines()
triple_list = []
triple_list_temp = []
for item in tqdm(train_raw):
    if item!='':
        triple_list_temp.append(item)
    else:
        triple_list.append(triple_list_temp)
        triple_list_temp = []
p_2only = re.compile(r'<.*?>')
p_ques = re.compile(r'q.*:(.*)')
s_ques_answ_list = []
for i in tqdm(triple_list):
    s_ques_answ_list_temp = []
    l = p_2only.findall(i[1])
    if len(l)==2:
        s = l[0].replace('<', '').replace('>', '')
        p = l[1].replace('<', '').replace('>', '')
        o = i[2].replace('\t', ',').replace('<', '').replace('>', '')
        answ = o
        if answ[-1] == ',':
            answ = answ[:len(answ)-1]
            answ = answ + '。'
        ques = p_ques.findall(i[0])[0]
        s_ques_answ_list_temp.append(s)
        s_ques_answ_list_temp.append(ques)
        s_ques_answ_list_temp.append(answ)
        s_ques_answ_list.append(s_ques_answ_list_temp)


with open(s_ques_answ_no_temp_address, 'w', encoding='utf-8') as f:
    json.dump(s_ques_answ_list, f, ensure_ascii=False)