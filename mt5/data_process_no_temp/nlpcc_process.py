import json
import re
import random

nlpcc_kbqa_train_knowques_address = '../../src/data/processed/nlpcc_kbqa/ed_ekbert/train.tsv'
nlpcc_kbqa_train_compansw_address = '../../src/data/processed/nlpcc_kbqa/train.json'
nlpcc_kbqa_train_knowques_compansw_address = '../data_no_temp/nlpcc_kbqa_train_knowques_compansw.json'
nlpcc_kbqa_dev_knowques_address = '../../src/data/processed/nlpcc_kbqa/ed_ekbert/dev.tsv'
nlpcc_kbqa_dev_compansw_address = '../../src/data/processed/nlpcc_kbqa/train.json'
nlpcc_kbqa_dev_knowques_compansw_address = '../data_no_temp/nlpcc_kbqa_dev_knowques_compansw.json'
nlpcc_kbqa_test_knowques_address = '../../src/data/processed/nlpcc_kbqa/ed_ekbert/test.tsv'
nlpcc_kbqa_test_compansw_address = '../../src/data/processed/nlpcc_kbqa/test.json'
nlpcc_kbqa_test_knowques_compansw_address = '../data_no_temp/nlpcc_kbqa_test_knowques_compansw.json'


# 数据集处理
def nlpcc_kbqa_process(address1, address2):
    with open(address1, 'r', encoding='utf-8') as f:
        knowques_list = f.read().splitlines()
    # for line in p:
    #     print(line)  # str类型 以\t分隔开
    total_list1 = []
    for line in knowques_list:
        line = line.split('\t')
        if line[2] == '1':
            know = re.sub('(<e2>.+</e2>)', '', line[1]).replace(':', '是')
            ques = line[0].replace('<e1>', '').replace('</e1>', '')
            know_ques = know + ',' + ques + '请回答:'
            temp_list = []
            # if len(know_ques) <= 512:
            temp_list.append(ques)
            temp_list.append(know_ques)  # [问题，知识：，问题：]
            total_list1.append(temp_list)  # 将它们放入一个list中
    with open(address2, 'r', encoding='utf-8') as f:
        compansw_list = f.read().splitlines()  # list是列表类型，但内容为str类型的字典，需要转换
    total_list2 = []
    for item in compansw_list:
        item = eval(item)
        ques1 = item['question']
        comp_answ = item['gold_answer']
        temp_list1 = []
        temp_list1.append(ques1)
        temp_list1.append(comp_answ)
        total_list2.append(temp_list1)
    total_list = []
    for item1 in total_list1:  # 根据两者问题相同，将“知识，问题”和“完整答案”放到元组，所有元组放到一个列表
        ques_1 = item1[0]
        for item2 in total_list2:
            ques_2 = item2[0]
            if ques_1 == ques_2:
                temp_tumple = (item1[1], item2[1])
                total_list.append(temp_tumple)
    # with open(address3, 'w', encoding='utf-8') as f:
    #     json.dump(total_list, f, ensure_ascii=False)
    return total_list


if __name__ == '__main__':
    nlpcc_kbqa_train_knowques_compansw_temp = nlpcc_kbqa_process(nlpcc_kbqa_train_knowques_address,
                                                                 nlpcc_kbqa_train_compansw_address)
    nlpcc_kbqa_dev_knowques_compansw_temp = nlpcc_kbqa_process(nlpcc_kbqa_dev_knowques_address,
                                                               nlpcc_kbqa_dev_compansw_address)
    nlpcc_kbqa_test_knowques_compansw_temp = nlpcc_kbqa_process(nlpcc_kbqa_test_knowques_address,
                                                                nlpcc_kbqa_test_compansw_address)
    nlpcc_kbqa_total_knowques_compansw = nlpcc_kbqa_train_knowques_compansw_temp + nlpcc_kbqa_dev_knowques_compansw_temp + nlpcc_kbqa_test_knowques_compansw_temp
    random.seed(66)
    random.shuffle(nlpcc_kbqa_total_knowques_compansw)

    nlpcc_kbqa_train_knowques_compansw = nlpcc_kbqa_total_knowques_compansw[
                                         :round(0.7 * len(nlpcc_kbqa_total_knowques_compansw))]  # round()四舍五入
    nlpcc_kbqa_dev_knowques_compansw = nlpcc_kbqa_total_knowques_compansw[
                                       round(0.7 * len(nlpcc_kbqa_total_knowques_compansw)):round(
                                           0.9 * len(nlpcc_kbqa_total_knowques_compansw))]
    nlpcc_kbqa_test_knowques_compansw = nlpcc_kbqa_total_knowques_compansw[
                                        round(0.9 * len(nlpcc_kbqa_total_knowques_compansw)):]
    with open(nlpcc_kbqa_train_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc_kbqa_train_knowques_compansw, f, ensure_ascii=False)
    with open(nlpcc_kbqa_dev_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc_kbqa_dev_knowques_compansw, f, ensure_ascii=False)
    with open(nlpcc_kbqa_test_knowques_compansw_address, 'w', encoding='utf-8') as f:
        json.dump(nlpcc_kbqa_test_knowques_compansw, f, ensure_ascii=False)

print('nlpcc_no_temp数据处理完毕！')