# 数据集

## 知识库问答数据集
1. NLPCC CKBQA
2. CCKS CKBQA

## 实体链接数据集
1. CCKS 2020 EL
2. CCKS 2019 EL

## 知识图谱
1. NLPCC KB
2. CCKS KB

# 在数据集 NLPCC CKBQA 上执行

## 训练单模型

```bash
cd scripts
bash run_train_md_bbkbqa.sh
bash run_train_md_kbert.sh
bash run_train_md_ekbert.sh
bash run_train_ed_bbkbqa.sh
bash run_train_ed_kbert.sh
bash run_train_ed_ekbert.sh
```

## 评估单模型

```bash
cd scripts
bash run_eval_md_bbkbqa.sh
bash run_eval_md_kbert.sh
bash run_eval_md_ekbert.sh
bash run_eval_ed_bbkbqa.sh
bash run_eval_ed_kbert.sh
bash run_eval_ed_ekbert.sh
```

## 评估pipeline模型

```bash
bash run_eval_el_bbkbqa.sh
bash run_eval_el_kbert.sh
bash run_eval_el_ekbert.sh
```

# 在数据集 CCKS CKBQA 上执行

# 在数据集 CCKS 2020 EL 上执行

# 在数据集 CCKS 2019 EL 上执行

# 数据预处理
1. nlpcc 2016 kbqa
```bash
python run_preprocess.py
```

2. ccks el

```

```

3. 

```

```



# 问题记录

问题1:bb-kbqa基于官方提供的KBQA数据集构建的实体消歧数据集有问题
```
给的训练数据集三元组spo中，s代表的gold_entity错误，处理逻辑：通过mention找到所有candidates，通过标签spo中的p关系和o答案匹配candidates对应的spo，反向找到匹配的s作为gold_entity
eg.
<question id=1>	《机械设计基础》这本书的作者是谁？
<triple id=1>	机械设计基础 ||| 作者 ||| 杨可桢，程光蕴，李仲生
<answer id=1>	杨可桢，程光蕴，李仲生
正确的triple是：机械设计基础(2010年高等教育出版社出版作者杨可桢)	作者	杨可桢，程光蕴，李仲生
```
问题1状态:已解决，通过标注数据提供的三元组的关系和答案，倒找到正确gold-entity

# todo
1. 抽象一个评估方法（recall，pre，acc，F1,acc@topN）的文件出来。各个模型都调用这个文件的方法，对齐模型的表现。
目前bb-kbqa和ekbert用的是同一个，kbert用的是另一个☑️
2. 评估结果一并写到文件日志 ☑️
3. 把md和ed的infusion写成一个类，完全了50%
4. ner任务的评估方法重写（参考NER的项目）☑️
5. 把el代码写好,写好发现md生成了8k+条数据，而ed有9k+数据，要对齐这两个数据集的数据。☑️
6. kbert的ner任务要写predict,输出hitted_flag数据到ner的output路径，然后输出的时候统一用参数计算el得分
7. 统一了kbert ed的train和eval代码，但是在test上的表现和之前的有变化，需要review。可以了☑️



# 实验结果

1. NLPCC KBQA

|                          | nlpcc-md   |            |            | nlpcc-md   |            |            |
| ------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|                          | dev        |            |            | test       |            |            |
|                          | F1         | P          | R          | F1         | P          | R          |
| bb-kbqa                  | 0.9508     | 0.9486     | 0.9529     | 0.9612     | 0.9587     | 0.9637     |
| k-bert                   | 0.9622     | 0.9611     | 0.9632     | 0.9559     | 0.9531     | 0.9587     |
| W2NER                    |            |            |            |            |            |            |
| EKBERT（min_span=2）     | 0.9235     | 0.9263     | 0.9208     | 0.9428     | 0.9463     | 0.9393     |
| **EKBERT（min_span=3）** | **0.9611** | **0.9604** | **0.9619** | **0.9757** | **0.9757** | **0.9758** |
| EKBERT（min_span=4）     | 0.9638     | 0.9620     | 0.9656     | 0.9584     | 0.9550     | 0.9617     |

|            | nlpcc-ed   |            |            | nlpcc-ed   |            |            |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|            | dev        |            |            | test       |            |            |
|            | Acc@1      | Acc@2      | Acc@3      | Acc@1      | Acc@2      | Acc@3      |
| bb-kbqa    | 0.8011     | 0.8662     | 0.8953     | 0.8473     | 0.8988     | 0.9241     |
| k-bert     | 0.8177     | 0.8835     | 0.9230     | 0.8545     | 0.9134     | 0.9368     |
| **EKBERT** | **0.8454** | **0.9209** | **0.9424** | **0.8791** | **0.9409** | **0.9613** |

|                | nlpcc-EL   |            |            |            |            |            |
| -------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|                | test       |            |            |            |            |            |
|                | Acc@1      | Acc@2      | Acc@3      | F1         | P          | R          |
| bb-kbqa        | 0.8349     | 0.8760     | 0.8952     | 0.8371     | 0.8349     | 0.8393     |
| k-bert         | 0.8374     | 0.8833     | 0.9017     | 0.8399     | 0.8374     | 0.8424     |
| **EKBERT**     | **0.8772** | **0.9270** | **0.9433** | **0.8772** | **0.8772** | **0.8773** |
| google端到端EL |            |            |            |            |            |            |

2. 

