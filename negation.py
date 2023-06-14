import torch
import argparse
import os

from collections import OrderedDict

from transformers import (
    AutoAdapterModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)

TRANSFORMERS_CACHE="checkpoints/hf_model"
HF_DATASETS_CACHE="checkpoints/hf_model"
HF_METRICS_CACHE="checkpoints/hf_model"

cache_dir=TRANSFORMERS_CACHE

def adapter_negation(
    model_name_or_path,
    adapter,
    adapter_config,
    save_path,
    ):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
    )
    model = AutoAdapterModel.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        config=config
    )
    model.load_adapter(
        adapter,
        config=adapter_config,
    )

    state_dict = model.state_dict()
    merged_keys = [mk for mk in state_dict.keys() if  ("adapters" in mk) or ("lora" in mk)]
    breakpoint()
    if adapter_config=="lora":
        neg_dict = {k:-v for k,v in state_dict.items() if "lora_B" in k}
    # ia3 (h+l*delta_h)-(h+delta_h)=(l-1)*delta_h h+delta_h-(l-1)*delta_h=h+(2-l)*delta_h
    elif adapter_config=="ia3":
        neg_dict = {k:(torch.ones(v.shape)*2-v) for k,v in state_dict.items() if "lora" in k}

    state_dict.update(neg_dict)
    model.load_state_dict(state_dict)
    model.save_all_adapters(save_path)
    
def main():
    model_name_or_path="roberta-base"
    adapter="checkpoints/glue/rte/20230208/lora_tr_train_0.1e-3.6000.0.01.0.1.32.8.364964.7/rte"
    #"checkpoints/glue/rte/20230208/ia3_tr_train_0.2e-2.1500.0.01.0.1.64.364676.2/rte"
    adapter_config="lora"
    save_path="merged_adapters/negation/364964.7"
    adapter_negation(model_name_or_path,adapter,adapter_config,save_path)

if __name__ == "__main__":
    main()
