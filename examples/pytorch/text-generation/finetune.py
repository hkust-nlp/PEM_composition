from datasets import load_dataset
import argparse
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead,AdapterConfig,AdapterTrainer,AutoAdapterModel,MultiLingAdapterArguments,HfArgumentParser
from transformers import GPT2LMHeadModel,GPT2Tokenizer
from transformers import TextDataset,DataCollatorForLanguageModeling
from torch.utils.data import Dataset, random_split
import regex as re
import os
import pdb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
# os.environ["CUDA_VISIBLE_DEVICE"]="1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers import TextDataset,DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token
device = "cuda"
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.parallelize() #naive的那种
# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])
# net = torch.nn.DataParallel(model, device_ids=[1, 2, 3])
class NetflixDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


descriptions = load_dataset('civil_comments',split='train+validation').filter(lambda x: x["toxicity"]>0.8)['text']
#pdb.set_trace()
max_length = max([len(tokenizer.encode(description)) for description in descriptions])
print(max_length)
# pdb.set_trace()
dataset = NetflixDataset(descriptions, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
print(train_size)
pdb.set_trace()
print(torch.cuda.device_count())
data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
import gc
gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
# parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))

parser.add_argument(
        "--output_dir",
        default='',
        type=str,
        # required=True,
        help="Path to save")

args = parser.parse_args()
training_args = TrainingArguments(
    output_dir=args.output_dir, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=6, # number of training epochs
    learning_rate=5e-5,
    per_device_train_batch_size=1, # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=2000, # after # steps model is saved
    save_total_limit=5, \
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    remove_unused_columns=False,
    # fp16=True,
    # gradient_accumulation_steps=16,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])},
    # train_dataset=train_dataset,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
trainer.save_model()
