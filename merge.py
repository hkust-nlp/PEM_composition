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

def test(text1,text2):
    print('hello world!')
    print(text1,text2)

def test1(model_name_or_path,
    task_name_list,
    adapter_list,
    adapter_config,
    save_path):
    print(task_name_list, type(task_name_list))
    #print(adapter_list)
    print(adapter_config, type(adapter_config))
    print(model_name_or_path,type(task_name_list))
    print(save_path,type(save_path))
    print('pass!')

def compute_pairwise_coefficient_sets(step=0.2):
    num_set = 1/step # 5
    num_set=int(num_set)
    sets = [[(ns+1)/num_set, 1-(ns+1)/num_set] for ns in range(num_set)]
    # sets = [[0.0, 1.0]] + sets
    return sets

def adapter_merge(
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
    model_name_or_path="roberta-large"

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

    if full_merge:
        range_list=model_name_or_path
    else:
        model_name_or_path=model_name_or_path[0]
        range_list=adapter_list
    # model-level
    if coefficient_sets == None:
        coefficient_sets = len(range_list) * [ 1 / len(range_list)]
    else:
        # suppose that len(coefficient_sets) == len(adapter_list)
        # TODO: assert len(coefficient_sets) == len(adapter_list)
        coefficient_sets = [float(x) for x in coefficient_sets]
        coefficient_sets = [ x/sum(coefficient_sets) for x in coefficient_sets]
    print(coefficient_sets)

    state_dict={}
    merged_dict={}
    merged_keys = None
    merged_values=[]
    sum_factors = None

    for i, _ in enumerate(range_list):
        if full_merge:
            model_name_or_path=range_list[i]
        else:
            adapter=range_list[i]
        # have to load model every time when we load adapter
        if dropout_list == None:
            dropout=0.1
        else:
            dropout=float(dropout_list[i])
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            attention_probs_dropout_prob=dropout,
            hidden_dropout_prob=dropout
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
        # model = AutoModelForSequenceClassification
        # model = AutoModelForSeq2SeqLM
        model = AutoAdapterModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))

        # print(adapter)

        if full_merge == False:
            model.load_adapter(
                adapter,
                config=adapter_config,
                load_as=merged_name,
            )

        state_dict = model.state_dict()
        if dropout == 0:
            head_keys = [hk for hk in state_dict.keys() if "head" in hk]
            for hk in head_keys:
                if '0' in hk:
                    hk_n = '1'.join(hk.split('0'))
                elif '2' in hk:
                    hk_n = '4'.join(hk.split('2'))
                state_dict[hk_n]=state_dict[hk]
                state_dict.pop(hk)
        # breakpoint()

        if merged_keys == None:
            if full_merge:
                if merge_head:
                    merged_keys = [mk for mk in state_dict.keys()]
                else:
                    merged_keys = [mk for mk in state_dict.keys() if ("head" not in mk) and ("classifier" not in mk)]
            else:
                merged_keys = [mk for mk in state_dict.keys() if  ("adapters" in mk) or ("lora" in mk)] # adapters / lora
                head_keys = [hk for hk in state_dict.keys() if "head" in hk]
                if merge_head:
                    merged_keys = merged_keys + head_keys
            merged_values = [0] * len(merged_keys)
            sum_factors = [0] * len(merged_keys)

        # parameter-level
        factors = None
        if merge_way == "fisher":
            fisher = torch.load(fisher_list[i])
            fisher = {k:v.detach() for k,v in fisher.items()}

            # normalization
            if fisher_normalization:
                vs=[v for v in fisher.values()]
                norm_constants=torch.sqrt(sum([torch.sum(torch.square(v)) for v in fisher.values()]))
                coefficient_sets[i]=coefficient_sets[i]/norm_constants

            # fisher_floor: avoid numerical problem
            if (not fisher_favor_target_model) or (i == len(range_list)-1):
                fisher = {k:torch.maximum(v, torch.tensor(fisher_floor)) for k,v in fisher.items()}

            # TODO:assert set(merged_keys)==set(fisher.keys())
            factors = OrderedDict([(mk, fisher[task_name_list[i].join(mk.split(merged_name))]*coefficient_sets[i]) for mk in merged_keys])
        elif merge_way == "simple":
            factors = OrderedDict(zip(merged_keys, [coefficient_sets[i]]*len(merged_keys)))

        sum_factors = [sf+f for sf,f in zip(sum_factors, factors.values())]

        # Add together
        new_values = [state_dict[mk]*factors[mk] for mk in merged_keys]
        merged_values = [mv+nv for mv,nv in zip(merged_values, new_values)]

        if i == len(range_list)-1:
            merged_values = [torch.div(mv, sf) for mv,sf in zip(merged_values, sum_factors)]
            merged_dict = dict(zip(merged_keys, merged_values))

            state_dict.update(merged_dict)
            if dropout == 0:
                config = AutoConfig.from_pretrained(
                    model_name_or_path,
                )
                model = AutoAdapterModel.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir,
                    config=config,
                )
                if full_merge == False:
                    model.load_adapter(
                        adapter,
                        config=adapter_config,
                        load_as=merged_name,
                    )
            model.load_state_dict(state_dict)
            if full_merge:
                config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    # num_labels=num_labels,
                    finetuning_task=task_name_list[i],
                    cache_dir=cache_dir,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir,
                )
                model.save_pretrained(save_path)
                config.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            else:
                model.save_all_adapters(save_path)

def pairwise_create_adapter():
    model_name_or_path="roberta-large"

    task_name_list=["sst2","cola","qnli","mrpc"] #"mnli",

    pretrained_adapter_list=["sentiment/sst-2@ukp",
    "lingaccept/cola@ukp",
    "nli/qnli@ukp",
    "sts/mrpc@ukp"]
    #"nli/multinli@ukp",

    adapter_config="houlsby"
    #"pfeiffer"

    save_path="merged_adapters/"

    for i in range(len(task_name_list)):
        for j in range(len(task_name_list)):
            if i==j:
                continue
            task_name=[task_name_list[i], task_name_list[j]]
            pretrained_adapter=[pretrained_adapter_list[i], pretrained_adapter_list[j]]
            task_path="_".join(task_name)
            path=save_path+adapter_config+'/'+task_path+'/'

            #test2(model_name_or_path, task_name, pretrained_adapter, adapter_config, path)
            adapter_merge(model_name_or_path, task_name, pretrained_adapter, adapter_config, path)

def main():
    model_name_or_path=["roberta-base", "roberta-base"]
    task_name=["cola","cola"] # ["cola","sst2"]
    pretrained_adapter=["lingaccept/cola@ukp", "lingaccept/cola@ukp"]#["lingaccept/cola@ukp","sentiment/sst-2@ukp"]
    adapter_config="pfeiffer"
    save_path="try-save-adapters/"

    coefficient_sets=[0.5,0.5]
    adapter_merge(model_name_or_path, task_name, pretrained_adapter, adapter_config, save_path, coefficient_sets, full_merge=True)

def init_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('-m','--model',nargs='*',required=True,help="base model to use")
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

    # pairwise_create_adapter()

    args = init_args()

    adapter_merge(
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
