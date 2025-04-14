import evaluate
import json

ccks2020_kbqa_test_model_gene_answer_address = '../../mt5_model_gene_answer/ccks2020_kbqa_test_model_gene_answer.json'
perplexity = evaluate.load("perplexity", module_type="metric")
# input_texts = ["这本书的作者是杨姐", "这部电影的作者是杨姐", "这款游戏的作者是杨姐"]
with open(ccks2020_kbqa_test_model_gene_answer_address, 'r', encoding='utf-8') as f:
    p = json.load(f)
# print(p)
p_temp = []
for i in range(len(p)):
    j = p[i]
    p_temp.append(j)
input_texts = p_temp
results = perplexity.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=input_texts)
# print(list(results.keys()))
print('ccks2020的perplexity平均值为：', round(results["mean_perplexity"], 2))
# for i in results["perplexities"]:  # 打印每句话的ppl
#     print(round(i, 2))