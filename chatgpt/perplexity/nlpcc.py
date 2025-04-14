import evaluate
import json

nlpcc_kbqa_test_model_gene_answer_address = '../data/nlpcc_answer.txt'

perplexity = evaluate.load("perplexity", module_type="metric")
# input_texts = ["这本书的作者是杨杰", "这部电影的作者是杨杰", "这款游戏的作者是杨杰"]
with open(nlpcc_kbqa_test_model_gene_answer_address, 'r', encoding='utf-8') as f:
    p_templist = f.readlines()
p = []
for item in p_templist:
    p.append(item.replace('\n', ''))
# print(p)
p_temp = []
for i in range(len(p)):
    j = p[i][:256]
    if len(j) > 0:
        p_temp.append(j)
input_texts = p_temp
results = perplexity.compute(model_id='gpt2',
                             add_start_token=True,
                             predictions=input_texts)
# print(list(results.keys()))
print('nlpcc2016的perplexity平均值为：', round(results["mean_perplexity"], 2))
# for i in results["perplexities"]:  # 打印每句话的ppl
#     print(round(i, 2))