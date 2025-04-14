import copy
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-trained_model_address', type=str, help='模型存放位置')
args = parse.parse_args()

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# flant5_tokenizer = T5Tokenizer.from_pretrained('../flan-t5-base')
flant5_tokenizer = T5Tokenizer.from_pretrained('../../mt5/mt5_base_model')
flant5_model = T5ForConditionalGeneration.from_pretrained('../../mt5/mt5_base_model')
pl.seed_everything(42)

nlpcc_kbqa_train_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_train_knowques_compansw.json'
nlpcc_kbqa_dev_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_dev_knowques_compansw.json'
nlpcc_kbqa_test_knowques_compansw_address = '../../mt5/data/nlpcc_kbqa_test_knowques_compansw.json'


with open(nlpcc_kbqa_train_knowques_compansw_address, 'r', encoding='utf-8') as f:
    nlpcc_train_data = json.load(f)
with open(nlpcc_kbqa_dev_knowques_compansw_address, 'r', encoding='utf-8') as f:
    nlpcc_dev_data = json.load(f)


class flant5Dataset(Dataset):
    def __init__(self, tokenizer, dataset, max_len_inp=512, max_len_out=512):
        self.dataset = dataset

        self.max_len_input = max_len_inp
        self.max_len_output = max_len_out
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.skippedcount = 0
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                "labels": labels}

    def _build(self):
        for inputs, outputs in self.dataset:
            input_sent = inputs
            ouput_sent = outputs

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_sent], max_length=self.max_len_input, pad_to_max_length=True, return_tensors="pt",
                truncation=True
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [ouput_sent], max_length=self.max_len_output, pad_to_max_length=True, return_tensors="pt",
                truncation=True
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


train_data = nlpcc_train_data
dev_data = nlpcc_dev_data
train_dataset = flant5Dataset(flant5_tokenizer, train_data)
dev_dataset = flant5Dataset(flant5_tokenizer, dev_data)


class FLANT5FineTuner(pl.LightningModule):
    # def __init__(self, hparams, t5model, t5tokenizer):
    def __init__(self, t5model, t5tokenizer):
        super(FLANT5FineTuner, self).__init__()
        # self.hparams = hparams
        self.model = t5model
        self.tokenizer = t5tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=batch["target_ids"],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )

        loss = outputs[0]
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=8, num_workers=16, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dev_dataset, batch_size=8, num_workers=16)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer


# import argparse

if __name__ == '__main__':
    # model = T5FineTuner(args, t5_model, t5_tokenizer)
    model = FLANT5FineTuner(flant5_model, flant5_tokenizer)

    trainer = pl.Trainer(max_epochs=5, gpus=4, strategy="dp")

    trainer.fit(model)
    torch.save(model, '../flant5_trained_model/mt5_nlpcc.pkl')
    # torch.save(model, args.trained_model_address)


