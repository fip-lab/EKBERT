import json
from tqdm import tqdm

nlpcc_2018_raw_triple_address = '../../../src/data/raw/nlpcc_kbqa_2018/knowledge/nlpcc-iccpol-2016.kbqa.kb'
nlpcc_2018_s_po_address = '../../data/nlpcc2018_s_po.json'

with open(nlpcc_2018_raw_triple_address, 'r', encoding='utf-8') as f:
    nlpcc_2018_raw_triple = f.read().splitlines()

# for idx, item in tqdm(enumerate(nlpcc_2018_raw_triple)):
#     if idx < 100:
#         print(item)

s_po = []
for item in nlpcc_2018_raw_triple:
    list_temp = []
    item1 = item.split('|||')
    if len(item1) == 3:
        s = item1[0].replace(' ', '')
        po = item1[1].replace(' ', '') + '是' + item1[2].replace(' ', '') + ','
        list_temp.append(s)
        list_temp.append(po)
        s_po.append(list_temp)

processed_data = {}
for item in tqdm(s_po):
    if len(item) == 2:
        name = item[0]
        item_type = item[1]

        if name in processed_data:
            processed_data[name].append(item_type)
        else:
            processed_data[name] = [item_type]

half_result = []

for name, types in processed_data.items():
    result = [name, types]
    half_result.append(result)
processed_data1 = {}
for i in half_result:
    name = i[0]
    all_know = i[1]
    know = ''
    for j in all_know:
        know = know + j
    processed_data1[name] = know
# print(processed_data1)
final_s_po_list = []
for s, po in tqdm(processed_data1.items()):
    final_s_po_list_temp = []
    final_s_po_list_temp.append(s)
    final_s_po_list_temp.append(po)
    final_s_po_list.append(final_s_po_list_temp)

with open(nlpcc_2018_s_po_address, 'w', encoding='utf-8') as f:
    json.dump(final_s_po_list, f, ensure_ascii=False)
