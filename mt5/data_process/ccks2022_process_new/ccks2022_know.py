import json
import re
from tqdm import tqdm

triple_address = '../../../src/data/raw/ccks_kbqa_2022/triple.txt'
s_po_address = '../../data/ccks2022_s_po.json'
# processed_data_address = '../../data/ccks2022_processed_data.json'
# s_po_list_address = '../../data/ccks2022_s_po_list.json'

with open(triple_address, 'r', encoding='utf-8') as f:
    triple_list = f.read().splitlines()
# for idx,item in enumerate(triple_list):
#     if idx<20:
#         print(item.replace(' .\n', '').split('\t'))
# print(len(triple_list))
# p1 = re.compile(r'<.*?>')
s_po_list = []
for item in tqdm(triple_list):
# for idx, item in enumerate(triple_list):
#     if idx < 20:
    item = item.replace(' .', '').split('\t')
    list_temp = []
    # l = p1.findall(item)
    # if len(l)==3:
    if len(item)==3:
        s = item[0].replace('<', '').replace('>', '')
        p = item[1].replace('<', '').replace('>', '')
        o = item[2].replace('<', '').replace('>', '')
        po = p + '是' + o
        list_temp.append(s)
        list_temp.append(po)
        s_po_list.append(list_temp)
# print(len(s_po_list))
# print(s_po_list)
processed_data = {}
for item in tqdm(s_po_list):
    if len(item)==2:
        name = item[0]
        item_type = item[1] + ','

        if name in processed_data:
            processed_data[name].append(item_type)
        else:
            processed_data[name] = [item_type]

half_result = []

for name, types in processed_data.items():
    result = [name, types]
    half_result.append(result)
# print(final_result)
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
# final_s_po_list = tuple(final_s_po_list)
# print(final_s_po_list)
with open(s_po_address, 'w', encoding='utf-8') as f:
    json.dump(final_s_po_list, f, ensure_ascii=False)
# # with open(processed_data_address, 'w', encoding='utf-8') as f:
# #     json.dump(half_result, f, ensure_ascii=False)
# # with open(s_po_list_address, 'w', encoding='utf-8') as f:
# #     json.dump(s_po_list, f, ensure_ascii=False)