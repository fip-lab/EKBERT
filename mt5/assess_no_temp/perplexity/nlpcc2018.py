import evaluate
import json

nlpcc2018_kbqa_test_model_gene_answer_no_temp_address = '../../mt5_model_gene_answer_no_temp/nlpcc2018_kbqa_test_model_gene_answer_no_temp.json'
perplexity = evaluate.load("perplexity", module_type="metric")
# input_texts = ["这本书的作者是杨杰", "这部电影的作者是杨杰", "这款游戏的作者是杨杰"]
with open(nlpcc2018_kbqa_test_model_gene_answer_no_temp_address, 'r', encoding='utf-8') as f:
    p = json.load(f)
# print(p)
p_temp = []
for i in range(len(p)):
    j = p[i][:256]
    p_temp.append(j)
input_texts = p_temp
results = perplexity.compute(model_id='gpt2',
                             add_start_token=True,
                             predictions=input_texts)
# print(list(results.keys()))
print('nlpcc2018_no_temp的perplexity平均值为：', round(results["mean_perplexity"], 2))
