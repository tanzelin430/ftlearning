# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from torch import nn
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
#print(torch.__version__)
from datasets import load_dataset
emotions = load_dataset('emotion')
emotions_df = pd.DataFrame.from_dict(emotions['train'])
#print(emotions['train'][:5])
from transformers import AutoTokenizer
model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
encode_text = tokenizer.encode(emotions_df.iloc[6324]['text'])
#print(encode_text)
def batch_tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)
emotions_encoded = emotions.map(batch_tokenize,batched=True,batch_size=None)
#print(emotions_encoded['train'].features['text'])

emotions_encoded.set_format('torch', columns=['label', 'input_ids', 'attention_mask'])
#print(type(emotions_encoded['train']['text']))

from transformers import AutoModel
model = AutoModel.from_pretrained(model_ckpt)
print(model)
from transformers import TrainingArguments, Trainer
batch_size = 64
logging_steps = len(emotions_encoded['train'])//batch_size
model_name = f'{model_ckpt}_emotion_ft_0416'
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=4,
                                  learning_rate=2e-5,
                                  weight_decay=0.01,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  # write
                                  push_to_hub=False,
                                  log_level="error")
from transformer_utils import compute_classification_metrics
trainer = Trainer(model=model,
                  tokenizer=tokenizer,
                  train_dataset=emotions_encoded['train'],
                  eval_dataset=emotions_encoded['validation'],
                  args=training_args,
                  compute_metrics=compute_classification_metrics)
trainer.train()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
