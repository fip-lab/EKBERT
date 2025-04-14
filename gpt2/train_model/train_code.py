from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from transformers import AdamW
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json
import torch
tokenizer = BertTokenizer.from_pretrained("../gpt2_model")
model = GPT2LMHeadModel.from_pretrained("../gpt2_model")

nlpcc_kbqa_train_knowques_compansw_new_address = '../data/nlpcc_kbqa_train_knowques_compansw_single.json'
nlpcc_kbqa_dev_knowques_compansw_new_address = '../data/nlpcc_kbqa_dev_knowques_compansw_single.json'
nlpcc_kbqa_test_knowques_compansw_new_address = '../data/nlpcc_kbqa_test_knowques_compansw_single.json'

ccks2020_kbqa_train_knowques_compansw_new_address = '../data/ccks2020_kbqa_train_knowques_compansw_single.json'
ccks2020_kbqa_dev_knowques_compansw_new_address = '../data/ccks2020_kbqa_dev_knowques_compansw_single.json'
ccks2020_kbqa_test_knowques_compansw_new_address = '../data/ccks2020_kbqa_test_knowques_compansw_single.json'

ccks2022_kbqa_train_knowques_compansw_new_address = '../data/ccks2022_kbqa_train_knowques_compansw_single.json'
ccks2022_kbqa_dev_knowques_compansw_new_address = '../data/ccks2022_kbqa_dev_knowques_compansw_single.json'
ccks2022_kbqa_test_knowques_compansw_new_address = '../data/ccks2022_kbqa_test_knowques_compansw_single.json'


# nlpcc_kbqa_train_knowques_compansw_new_inp_address = '../data/nlpcc_kbqa_train_knowques_compansw_single_inp.json'
# nlpcc_kbqa_dev_knowques_compansw_new_inp_address = '../data/nlpcc_kbqa_dev_knowques_compansw_single_inp.json'
# nlpcc_kbqa_test_knowques_compansw_new_inp_address = '../data/nlpcc_kbqa_test_knowques_compansw_single_inp.json'
#
# ccks2020_kbqa_train_knowques_compansw_new_inp_address = '../data/ccks2020_kbqa_train_knowques_compansw_single_inp.json'
# ccks2020_kbqa_dev_knowques_compansw_new_inp_address = '../data/ccks2020_kbqa_dev_knowques_compansw_single_inp.json'
# ccks2020_kbqa_test_knowques_compansw_new_inp_address = '../data/ccks2020_kbqa_test_knowques_compansw_single_inp.json'
#
# ccks2022_kbqa_train_knowques_compansw_new_inp_address = '../data/ccks2022_kbqa_train_knowques_compansw_single_inp.json'
# ccks2022_kbqa_dev_knowques_compansw_new_inp_address = '../data/ccks2022_kbqa_dev_knowques_compansw_single_inp.json'
# ccks2022_kbqa_test_knowques_compansw_new_inp_address = '../data/ccks2022_kbqa_test_knowques_compansw_single_inp.json'
#
# nlpcc_kbqa_train_knowques_compansw_new_out_address = '../data/nlpcc_kbqa_train_knowques_compansw_single_out.json'
# nlpcc_kbqa_dev_knowques_compansw_new_out_address = '../data/nlpcc_kbqa_dev_knowques_compansw_single_out.json'
# nlpcc_kbqa_test_knowques_compansw_new_out_address = '../data/nlpcc_kbqa_test_knowques_compansw_single_out.json'
#
#
# ccks2020_kbqa_train_knowques_compansw_new_out_address = '../data/ccks2020_kbqa_train_knowques_compansw_single_out.json'
# ccks2020_kbqa_dev_knowques_compansw_new_out_address = '../data/ccks2020_kbqa_dev_knowques_compansw_single_out.json'
# ccks2020_kbqa_test_knowques_compansw_new_out_address = '../data/ccks2020_kbqa_test_knowques_compansw_single_out.json'
#
# ccks2022_kbqa_train_knowques_compansw_new_out_address = '../data/ccks2022_kbqa_train_knowques_compansw_single_out.json'
# ccks2022_kbqa_dev_knowques_compansw_new_out_address = '../data/ccks2022_kbqa_dev_knowques_compansw_single_out.json'
# ccks2022_kbqa_test_knowques_compansw_new_out_address = '../data/ccks2022_kbqa_test_knowques_compansw_single_out.json'

with open(nlpcc_kbqa_train_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    nlpcc_train_data = json.load(f)
with open(nlpcc_kbqa_dev_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    nlpcc_dev_data = json.load(f)
with open(nlpcc_kbqa_test_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    nlpcc_test_data = json.load(f)
with open(ccks2020_kbqa_train_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2020_train_data = json.load(f)
with open(ccks2020_kbqa_dev_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2020_dev_data = json.load(f)
with open(ccks2020_kbqa_test_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2020_test_data = json.load(f)
with open(ccks2022_kbqa_train_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2022_train_data = json.load(f)
with open(ccks2022_kbqa_dev_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2022_dev_data = json.load(f)
with open(ccks2022_kbqa_test_knowques_compansw_new_address, 'r', encoding='utf-8') as f:
    ccks2022_test_data = json.load(f)


# with open(nlpcc_kbqa_train_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     nlpcc_train_data_inp = json.load(f)
# with open(nlpcc_kbqa_dev_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     nlpcc_dev_data_inp = json.load(f)
# with open(nlpcc_kbqa_test_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     nlpcc_test_data_inp = json.load(f)
# with open(ccks2020_kbqa_train_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     ccks2020_train_data_inp = json.load(f)
# with open(ccks2020_kbqa_dev_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     ccks2020_dev_data_inp = json.load(f)
# with open(ccks2020_kbqa_test_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     ccks2020_test_data_inp = json.load(f)
# with open(ccks2022_kbqa_train_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     ccks2022_train_data_inp= json.load(f)
# with open(ccks2022_kbqa_dev_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     ccks2022_dev_data_inp = json.load(f)
# with open(ccks2022_kbqa_test_knowques_compansw_new_inp_address, 'r', encoding='utf-8') as f:
#     ccks2022_test_data_inp = json.load(f)
#
# with open(nlpcc_kbqa_train_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     nlpcc_train_data_out = json.load(f)
# with open(nlpcc_kbqa_dev_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     nlpcc_dev_data_out = json.load(f)
# with open(nlpcc_kbqa_test_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     nlpcc_test_data_out = json.load(f)
# with open(ccks2020_kbqa_train_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     ccks2020_train_data_out = json.load(f)
# with open(ccks2020_kbqa_dev_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     ccks2020_dev_data_out = json.load(f)
# with open(ccks2020_kbqa_test_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     ccks2020_test_data_out = json.load(f)
# with open(ccks2022_kbqa_train_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     ccks2022_train_data_out = json.load(f)
# with open(ccks2022_kbqa_dev_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     ccks2022_dev_data_out = json.load(f)
# with open(ccks2022_kbqa_test_knowques_compansw_new_out_address, 'r', encoding='utf-8') as f:
#     ccks2022_test_data_out = json.load(f)

class TextDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_text = self.data_list[idx]
        inputs = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=self.max_length,
                                return_tensors="pt")
        self.examples.append(inputs)
        return self.examples[idx]["input_ids"].squeeze(0), self.examples[idx]["attention_mask"].squeeze(0)
    # def __len__(self):
    #     return len(self.examples_inp)
    #
    # def __getitem__(self, idx):
    #     # return torch.tensor(self.examples_inp[idx]), torch.tensor(self.examples_lab[idx])
    #     # return torch.tensor(self.examples_inp[idx])
    #     return torch.tensor(self.examples_inp[idx])


train_data = nlpcc_train_data + ccks2020_train_data + ccks2022_train_data
dev_data = nlpcc_dev_data + ccks2020_dev_data + ccks2022_dev_data
# train_data_inp = nlpcc_train_data_inp + ccks2020_train_data_inp + ccks2022_train_data_inp
# dev_data_inp = nlpcc_dev_data_inp + ccks2020_dev_data_inp + ccks2022_dev_data_inp
# train_data_out = nlpcc_train_data_out + ccks2020_train_data_out + ccks2022_train_data_out
# dev_data_out = nlpcc_dev_data_out + ccks2020_dev_data_out + ccks2022_dev_data_out
# train_dataset = TextDataset(train_data, tokenizer, block_size=512)
# val_dataset = TextDataset(dev_data, tokenizer, block_size=512)
train_dataset = TextDataset(train_data, tokenizer)
val_dataset = TextDataset(dev_data, tokenizer)


# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)


class GPT2FineTuner(pl.LightningModule):
    # def __init__(self, hparams):
    def __init__(self, tokenizer, model):
        super().__init__()
        self.save_hyperparameters()
        # self.hparams = hparams
        self.tokenizer = tokenizer
        self.model = model

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids, _ = batch
        loss = self.model(input_ids, labels=input_ids).loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, _ = batch
        loss = self.model(input_ids, labels=input_ids).loss
        self.log('val_loss', loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, eps=1e-8)
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer


if __name__ == '__main__':
    model = GPT2FineTuner(tokenizer, model)
    trainer = pl.Trainer(gpus=1, max_epochs=5)
    trainer.fit(model)
    # torch.save(model, '../gpt2_trained_model/gpt2_1.pkl')
    torch.save(model.state_dict(), "../gpt2_trained_model/gpt2_1.pth")