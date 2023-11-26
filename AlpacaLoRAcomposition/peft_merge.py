import torch
from peft import (  # noqa: E402
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

TRANSFORMERS_CACHE="checkpoints/hf_model"
HF_DATASETS_CACHE="checkpoints/hf_model"
HF_METRICS_CACHE="checkpoints/hf_model"

cache_dir="checkpoints/hf_model"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def load_model_from_device(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "tloen/alpaca-lora-7b",
):
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True, 
            cache_dir=cache_dir,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            cache_dir=cache_dir,
        )
    return model
    
# Haven't implemented!!!!!!!
def peft_merge(
    # lora hyperparams
    lora_weights_list, #list[str] = ["tloen/alpaca-lora-7b", 'qychen/luotuo-lora-7b-0.1']
    load_8bit: bool = False,
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    save_path = None, #str = "checkpoints/merged_peft/",
    coefficient_sets = None,
):
    if coefficient_sets == None:
        coefficient_sets = len(lora_weights_list) * [ 1 / len(lora_weights_list)]
    else:
        # suppose that len(coefficient_sets) == len(adapter_list)
        coefficient_sets = [float(x) for x in coefficient_sets]
        coefficient_sets = [ x/sum(coefficient_sets) for x in coefficient_sets]
    print(coefficient_sets)
    
    merged_keys=[]
    merged_values=[]
    merged_dict={}

    for i,_ in enumerate(lora_weights_list):
        lora_weights=lora_weights_list[i]
        # tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = load_model_from_device(
            load_8bit=load_8bit,
            base_model=base_model,
            lora_weights=lora_weights,
        )

        old_state_dict = model.state_dict
        state_dict = old_state_dict()
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
        peft_dict=model.state_dict()
        # breakpoint()

        if i == 0:
            merged_keys=list(peft_dict.keys())
            merged_values = [0] * len(merged_keys)

        new_values = [peft_dict[mk]*coefficient_sets[i] for mk in merged_keys]
        merged_values = [mv+nv for mv,nv in zip(merged_values, new_values)]

        if i == len(lora_weights_list)-1:
            # merged_values = [torch.div(mv, sf) for mv,sf in zip(merged_values, sum_factors)]
            merged_dict = dict(zip(merged_keys, merged_values))
            #TODO: check upload merged_dict or state_dict
            state_dict.update(merged_dict)
            set_peft_model_state_dict(model, merged_dict)
            # model.load_state_dict(state_dict)
            if save_path:
                model.save_pretrained(save_path)
            else:
                return model

    # model = get_peft_model(model, config)
    # adapters_weights = torch.load(checkpoint_name)

def peft_analogy(
    # TODO: not done
    # lora hyperparams
    lora_weights_list, #list[str] = ["tloen/alpaca-lora-7b", 'qychen/luotuo-lora-7b-0.1']
    load_8bit: bool = False,
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    save_path = None, #str = "checkpoints/merged_peft/",
    coefficient_sets = None,
):
    if coefficient_sets == None:
        coefficient_sets = len(lora_weights_list) * [ 1 / len(lora_weights_list)]
    else:
        # suppose that len(coefficient_sets) == len(adapter_list)
        coefficient_sets = [float(x) for x in coefficient_sets]
        coefficient_sets = [ x/sum(coefficient_sets) for x in coefficient_sets]
        if len(coefficient_sets) == 2:
            coefficient_sets = coefficient_sets.append(coefficient_sets[1])
    print(coefficient_sets)
    
    merged_keys=[]
    merged_values=[]
    merged_dict={}

    for i,_ in enumerate(lora_weights_list):
        lora_weights=lora_weights_list[i]
        # tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = load_model_from_device(
            load_8bit=load_8bit,
            base_model=base_model,
            lora_weights=lora_weights,
        )

        old_state_dict = model.state_dict
        state_dict = old_state_dict()
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
        peft_dict=model.state_dict()
        if i == 2:
            neg_dict = {k:-v for k,v in peft_dict.items() if "lora_B" in k}
            peft_dict.update(neg_dict)
        # breakpoint()

        if i == 0:
            merged_keys=list(peft_dict.keys())
            merged_values = [0] * len(merged_keys)

        new_values = [peft_dict[mk]*coefficient_sets[i] for mk in merged_keys]
        merged_values = [mv+nv for mv,nv in zip(merged_values, new_values)]

        if i == len(lora_weights_list)-1:
            # merged_values = [torch.div(mv, sf) for mv,sf in zip(merged_values, sum_factors)]
            merged_dict = dict(zip(merged_keys, merged_values))
            state_dict.update(merged_dict)
            set_peft_model_state_dict(model, merged_dict)
            if save_path:
                model.save_pretrained(save_path)
            else:
                return model

    # model = get_peft_model(model, config)
    # adapters_weights = torch.load(checkpoint_name)

def peft_negation(
    # lora hyperparams
    lora_weights_list, #list[str] = ["tloen/alpaca-lora-7b", 'qychen/luotuo-lora-7b-0.1']
    load_8bit: bool = False,
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    save_path = None, #str = "checkpoints/merged_peft/",
    coefficient_sets = None,
):
    if coefficient_sets == None:
        coefficient_sets = len(lora_weights_list) * [ 1 / len(lora_weights_list)]
    else:
        # suppose that len(coefficient_sets) == len(adapter_list)
        coefficient_sets = [float(x) for x in coefficient_sets]
        coefficient_sets = [ x/100 for x in coefficient_sets]
    print(coefficient_sets)
    
    merged_keys=[]
    merged_values=[]
    merged_dict={}

    for i,_ in enumerate(lora_weights_list):
        lora_weights=lora_weights_list[i]
        # tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = load_model_from_device(
            load_8bit=load_8bit,
            base_model=base_model,
            lora_weights=lora_weights,
        )

        # old_dict=model.state_dict()
        old_state_dict = model.state_dict
        state_dict = old_state_dict()
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
        peft_dict=model.state_dict()
        # breakpoint()

        if i == 0:
            merged_keys=[k for k in list(peft_dict.keys()) if "lora_" in k]
            merged_values = [peft_dict[mk]*coefficient_sets[i] for mk in merged_keys]
        
        if i == len(lora_weights_list)-1:
            # breakpoint()
            new_values = [peft_dict[mk] for mk in merged_keys]
            delta_values = [nv for mv,nv in zip(merged_values, new_values)]
            delta_dict = dict(zip(merged_keys, delta_values))
            neg_dict = {k:-v for k,v in delta_dict.items() if "lora_A" in k} #lora_B/lora_A
            delta_dict.update(neg_dict)
            delta_values = [delta_dict[mk] for mk in merged_keys]
            # delta_values = new_values
            # merged_values = [2*mv-nv for mv,nv in zip(merged_values, new_values)]
            # print(coefficient_sets[i])mv*coefficient_sets[i]-
            merged_values = [(mv+(mv+dv)*coefficient_sets[i]) for mv,dv in zip(merged_values, delta_values)]
            # merged_values = delta_values

            merged_dict = dict(zip(merged_keys, merged_values))
            state_dict.update(merged_dict)
            set_peft_model_state_dict(model, state_dict)
            # new_dict=model.old_state_dict
            # breakpoint()
            if save_path:
                model.save_pretrained(save_path)
            else:
                return model

def main():
    # model=peft_merge(
    #     ['Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco','checkpoints/hf_model/llama7b-lora-medical'],
    #     # save_path="checkpoints/merged_peft/50_50_0",
    #     coefficient_sets=[50,50]
    # )
    # print(model.state_dict().keys())
    peft_negation(
        ["checkpoints/initialization/tloen--alpaca-lora-7b","checkpoints/initialization/civil-en"],
        coefficient_sets=[100,-100],
        save_path="checkpoints/initialization/negation-test-4"
    )

if __name__ == "__main__":
    main()