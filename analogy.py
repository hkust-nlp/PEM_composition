import torch
import argparse
import os

from collections import OrderedDict

from transformers import (
    AutoAdapterModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)

TRANSFORMERS_CACHE="checkpoints/hf_model"
HF_DATASETS_CACHE="checkpoints/hf_model"
HF_METRICS_CACHE="checkpoints/hf_model"

cache_dir=TRANSFORMERS_CACHE

def adapter_analogy(
    model_name_or_path,
    task_name_list,
    adapter_list,
    adapter_config,
    save_path,
    overwrite_existed = True,
    coefficient_sets = None,
    dropout_list = None,
    merge_way = "simple",
    merge_head = False,
    full_merge:bool = False,
    fisher_list = [],
    fisher_floor = 1e-6,
    fisher_favor_target_model = True,
    fisher_normalization = True,
    *args,
    **kwargs):
    '''
    examples:
    model_name_or_path="t5-base"

    # the aim task is the last task: cola
    task_name=["sst2","cola"]

    pretrained_adapter=["sentiment/sst-2@ukp","lingaccept/cola@ukp"]

    adapter_config="pfeiffer"

    # the merged adapter for ['stsb','sst2','rte'] will be saved in 'save_path/for_rte/rte_sst2_stsb'
    save_path="try-save-adapters/"

    # overwrite the may existed adapter on the save path
    overwrite_existed = True

    # "simple" "fisher" "regmean"
    merge_way="simple"

    # if True, there will be parameter-protecting procedure for only target model. 
    # Else there will be parameter-protecting procedure for all models.
    fisher_favor_target_model = True
    '''
    # what's the classification head belongs to

    if adapter_config=="fft":
        full_merge=True

    merged_name=task_name_list
    #merged_name.sort()
    merged_name="_".join(merged_name)

    # check if already existed
    will_save_path=os.path.join(save_path, merged_name)
    if os.path.isdir(will_save_path):
        if(overwrite_existed):
            print("Overwrite the existed adapter.")
        else:
            print("Merged adapter already created.")
            return

    base_model=model_name_or_path
    range_list=adapter_list
    # model-level
    if coefficient_sets == None:
        coefficient_sets = [0.5, 0.5]
    else:
        coefficient_sets = [float(x) for x in coefficient_sets]
        coefficient_sets = [ x/sum(coefficient_sets) for x in coefficient_sets]
    print(coefficient_sets)

    aux_dict={}
    aim_lm_dict={}
    aux_lm_dict={}

    for i, _ in enumerate(range_list):
        if full_merge:
            model_name_or_path=range_list[i]
        else:
            adapter=range_list[i]
        config = AutoConfig.from_pretrained(
            model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
        # model = AutoModelForSequenceClassification
        # model = AutoAdapterModel
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))
        if full_merge == False:
            model.load_adapter(
                adapter,
                config=adapter_config,
                load_as=merged_name,
            )
        if i == 0:
            aux_dict = model.state_dict()
        elif i == 1:
            aim_lm_dict = model.state_dict()
        elif i == 2:
            aux_lm_dict = model.state_dict()
        
        # adapter head problem......fuck
    if full_merge:
        model_name_or_path=base_model
        config = AutoConfig.from_pretrained(
            model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
        # model = AutoModelForSequenceClassification
        # model = AutoAdapterModel
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))
        state_dict=model.state_dict()
        aim_dict={k:(v + (aux_dict[k] - v)*coefficient_sets[0] + (aim_lm_dict[k] - aux_lm_dict[k])*coefficient_sets[1]) for k,v in state_dict.items()}
        aux_dict.update(aim_dict)
        model.load_state_dict(aux_dict)

        model.save_pretrained(save_path)
        config.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    else:
        if adapter_config == "ia3":
            aim_dict = {k:(torch.ones(v.shape)*(1-coefficient_sets[0]) + coefficient_sets[0]*v + coefficient_sets[1]*aim_lm_dict[k] - coefficient_sets[1]*aux_lm_dict[k]) for k,v in aux_dict.items() if "lora" in k}
        elif adapter_config == "lora":
            neg_dict = {k:-v for k,v in aux_lm_dict.items() if "lora_B" in k}
            aux_lm_dict.update(neg_dict)
            aim_dict = {k:(coefficient_sets[0]*v + coefficient_sets[1]*aim_lm_dict[k] + coefficient_sets[1]*aux_lm_dict[k]) for k,v in aux_dict.items() if "lora" in k}
        aux_dict.update(aim_dict)
        model.load_state_dict(aux_dict)
        model.save_all_adapters(save_path)

def main():
    model_name_or_path=["roberta-base", "roberta-base"]
    task_name=["cola","cola"] # ["cola","sst2"]
    pretrained_adapter=["lingaccept/cola@ukp", "lingaccept/cola@ukp"]#["lingaccept/cola@ukp","sentiment/sst-2@ukp"]
    adapter_config="pfeiffer"
    save_path="try-save-adapters/"

    coefficient_sets=[0.5,0.5]
    adapter_analogy(model_name_or_path, task_name, pretrained_adapter, adapter_config, save_path, coefficient_sets, full_merge=True)

def init_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('-m','--model',type=str,required=True,help="base model to use")
    parser.add_argument('-t','--adatasks',nargs='*',help="adapter's task name")
    parser.add_argument('-a','--adapters',nargs='*',help="adapters to merge")
    parser.add_argument('-c','--adaconfig',type=str,help="adapter's config")
    parser.add_argument('-p','--save_path',type=str,help="path to save merged adapter")
    parser.add_argument('-w','--merge_way',type=str,help="the way to merge adapters")
    parser.add_argument('--merge_head',type=bool,default=False,help="the way to merge adapter head if it have one")
    parser.add_argument('--full_merge',type=bool,default=False,help="to decide merge the whole model or only the adapters")
    parser.add_argument('-f','--fisher_list', default=[], nargs='*',help="fisher info for adapters")
    parser.add_argument('-o','--overwrite',type=bool,default=False,help="ignore existed merged adapter or not")
    parser.add_argument('-d','--dropout_list',default=None, nargs='*',help="has dropout layer or not")

    #parser.add_argument('-s','--do_search',type=bool,help="lambda grid search")
    parser.add_argument('--search_step', default=None, nargs='+', help="grid search step")

    args=parser.parse_args()
    return args

if __name__ == "__main__":
    # main()

    args = init_args()

    adapter_analogy(
        model_name_or_path = args.model,
        task_name_list = args.adatasks,
        adapter_list = args.adapters,
        adapter_config = args.adaconfig,
        save_path = args.save_path,
        coefficient_sets = args.search_step,
        merge_way = args.merge_way,
        merge_head = args.merge_head,
        full_merge = args.full_merge,
        fisher_list = args.fisher_list,
        dropout_list = args.dropout_list,
        overwrite_existed = args.overwrite,
    )
