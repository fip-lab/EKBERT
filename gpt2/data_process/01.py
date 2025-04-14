import json

# nlpcc_kbqa_train_knowques_compansw_new_address = '../data/nlpcc_kbqa_train_knowques_compansw_single.json'
# nlpcc_kbqa_dev_knowques_compansw_new_address = '../data/nlpcc_kbqa_dev_knowques_compansw_single.json'
# nlpcc_kbqa_test_knowques_compansw_new_address = '../data/nlpcc_kbqa_test_knowques_compansw_single.json'

nlpcc_kbqa_train_knowques_compansw_new_address = '../data/nlpcc_kbqa_train_knowques_compansw_single_inp.json'
nlpcc_kbqa_dev_knowques_compansw_new_address = '../data/nlpcc_kbqa_dev_knowques_compansw_single_inp.json'
nlpcc_kbqa_test_knowques_compansw_new_address = '../data/nlpcc_kbqa_test_knowques_compansw_single_inp.json'

kgclue_kbqa_train_knowques_compansw_new_address = '../data/kgclue_kbqa_train_knowques_compansw_single.json'
kgclue_kbqa_dev_knowques_compansw_new_address = '../data/kgclue_kbqa_dev_knowques_compansw_single.json'
kgclue_kbqa_test_knowques_compansw_new_address = '../data/kgclue_kbqa_test_knowques_compansw_single.json'

# kgclue_kbqa_train_knowques_compansw_new_address = '../data/kgclue_kbqa_train_knowques_compansw_single_inp.json'
# kgclue_kbqa_dev_knowques_compansw_new_address = '../data/kgclue_kbqa_dev_knowques_compansw_single_inp.json'
# kgclue_kbqa_test_knowques_compansw_new_address = '../data/kgclue_kbqa_test_knowques_compansw_single_inp.json'

nlpcc_kbqa_test_model_gene_answer_address = '../gpt2_model_gene_answer/nlpcc_kbqa_test_model_gene_answer.json'

kgclue_know_ques_answ_address = '../../mt5/data/kgclue_know_ques_answ.json'

nlpcc_kbqa_test_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_test_knowques_compansw.json'

das = '/disk3/xzh/bart_trained_model/01.txt'

with open(das) as f:
    p = f.read()


# a=0
# total = 0
# for idx, item in enumerate(p):
#     # if idx<5:
#     total+=len(item[1])
#     # print(item)
# print(total/2375)
print(p)
# print(len(p))
# list = []
# for idx, item in enumerate(p):
#     list.append(len(item))
# # print(item)
# print(len(list))
# max = []
# for idx, i in enumerate(list):
#     if i>1024:
#         max.append(idx)
# print(max)

# 最大2698
# total_new_list = []
# for item in p:
#     new_list = []
#     new_list.append(item[0])
#     str_temp = item[0] + item[1]
#     new_list.append(str_temp)
#     total_new_list.append(new_list)
# print(total_new_list)