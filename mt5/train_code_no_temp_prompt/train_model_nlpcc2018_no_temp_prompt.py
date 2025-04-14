import copy
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

t5_tokenizer = T5Tokenizer.from_pretrained('../mt5_base_model')
t5_model = T5ForConditionalGeneration.from_pretrained('../mt5_base_model')
pl.seed_everything(42)

nlpcc2018_train_data_no_temp_prompt_address = '../data_no_temp_prompt/nlpcc2018_train_data_no_temp_prompt.json'
nlpcc2018_dev_data_no_temp_prompt_address = '../data_no_temp_prompt/nlpcc2018_dev_data_no_temp_prompt.json'
nlpcc2018_test_data_no_temp_prompt_address = '../data_no_temp_prompt/nlpcc2018_test_data_no_temp_prompt.json'

with open(nlpcc2018_train_data_no_temp_prompt_address, 'r', encoding='utf-8') as f:
    nlpcc2018_train_data_no_temp_prompt = json.load(f)
with open(nlpcc2018_dev_data_no_temp_prompt_address, 'r', encoding='utf-8') as f:
    nlpcc2018_dev_data_no_temp_prompt = json.load(f)


class mt5Dataset(Dataset):
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


train_data = nlpcc2018_train_data_no_temp_prompt
dev_data = nlpcc2018_dev_data_no_temp_prompt
train_dataset = mt5Dataset(t5_tokenizer, train_data)
dev_dataset = mt5Dataset(t5_tokenizer, dev_data)


class T5FineTuner(pl.LightningModule):
    # def __init__(self, hparams, t5model, t5tokenizer):
    def __init__(self, t5model, t5tokenizer):
        super(T5FineTuner, self).__init__()
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
        return DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dev_dataset, batch_size=1, num_workers=4)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer


# import argparse

if __name__ == '__main__':
    # model = T5FineTuner(args, t5_model, t5_tokenizer)
    model = T5FineTuner(t5_model, t5_tokenizer)

    trainer = pl.Trainer(max_epochs=5, gpus=1)

    trainer.fit(model)
    torch.save(model, '../mt5_trained_model_no_temp_prompt/mt5_nlpcc2018_no_temp_prompt.pkl')


