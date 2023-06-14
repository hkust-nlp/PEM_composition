cache_dir='checkpoints/hf_model'
from transformers import GPT2LMHeadModel, GPT2Tokenizer,AutoModelForCausalLM
from transformers.adapters.configuration import AdapterConfig
import transformers.adapters.composition as ac
import argparse
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICE"]="1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import numpy as np
import torch
import random
import pandas as pd
from transformers import (
    AutoAdapterModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
adapter_config = AdapterConfig.load('ia3')
config= AutoConfig.from_pretrained('gpt2-large')

device = "cuda"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large',cache_dir=cache_dir,config=config)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2-large',cache_dir=cache_dir)
# model = AutoAdapterModel.from_pretrained('gpt2-large',cache_dir=cache_dir)

model.resize_token_embeddings(len(tokenizer))
model.load_adapter('./tmp/test-clm/20230325_00_59.ia3.0.7/civil_comments',
                    config=adapter_config,
                    load_as='civil_comments',
                    model_name='gpt2-large'
                )
# model = GPT2LMHeadModel.from_pretrained("gpt2-large")
model.set_active_adapters(["civil_comments"])
model.parallelize()

from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)
print('haha',seq_len)
nlls = []
prev_end_loc = 0
print(torch.cuda.device_count())
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over input tokens.
        # Multiply it with trg_len to get the summation instead of average.
        # We will take average over all the tokens to get the true average
        # in the last step of this example.
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(ppl)