import os

FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))  # 获得当前的绝对路径
FILE_DIR_PATH_2 = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))  # 获取当前文件上三级的目录 pen add
# print('FILE_DIR_PATH_2', FILE_DIR_PATH_2)

KGS = {
    'HowNet': os.path.join(FILE_DIR_PATH, 'kgs/HowNet.spo'),
    'CnDbpedia': os.path.join(FILE_DIR_PATH, 'kgs/CnDbpedia.spo'),
    'Medical': os.path.join(FILE_DIR_PATH, 'kgs/Medical.spo'),
    'nlpcc_train': os.path.join(FILE_DIR_PATH_2, 'data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_train.spo'),
    'nlpcc_test': os.path.join(FILE_DIR_PATH_2, 'data/processed/nlpcc_kbqa/nlpcc-iccpol-2016-new-filter_test.spo'),
    'nlpcc': os.path.join(FILE_DIR_PATH_2, 'data/raw/nlpcc_kbqa/nlpcc-iccpol-2016-new.spo'),
    'ccks_el_2019': os.path.join(FILE_DIR_PATH_2, 'data/processed/ccks_el_2019/ccks_2019_kb.spo'),
    'ccks_el_2020': os.path.join(FILE_DIR_PATH_2, 'data/processed/ccks_el_2020/ccks_2020_kb.spo'),
    'ccks_el_2020_m2id': os.path.join(FILE_DIR_PATH_2, 'data/processed/ccks_el_2020/mention2id.txt'),
    "ccks_kbqa_2020": os.path.join(FILE_DIR_PATH_2, 'data/processed/ccks_kbqa_2020/ccks_kbqa_2020_filter.spo'),
    "ccks_kbqa_2020_m2id": os.path.join(FILE_DIR_PATH_2, 'data/processed/ccks_kbqa_2020/mention2id.txt')
}

MAX_ENTITIES = 20  # 匹配到的kg实体，可以加入多少个po

# Special token words.
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
ENT_TOKEN = '[ENT]'
SUB_TOKEN = '[SUB]'
PRE_TOKEN = '[PRE]'
OBJ_TOKEN = '[OBJ]'

NEVER_SPLIT_TAG = [
    PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN,
    ENT_TOKEN, SUB_TOKEN, PRE_TOKEN, OBJ_TOKEN
]
