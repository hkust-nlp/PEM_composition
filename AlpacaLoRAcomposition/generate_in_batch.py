import sys
import os

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

import argparse
import csv
from peft_merge import load_model_from_device, peft_merge, peft_negation
from load_instructions import load_instruction_sets

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TRANSFORMERS_CACHE="checkpoints/hf_model"
HF_DATASETS_CACHE="checkpoints/hf_model"
HF_METRICS_CACHE="checkpoints/hf_model"
HUGGINGFACE_HUB_CACHE="checkpoints/hf_model"
HUGGINGFACE_ASSETS_CACHE="checkpoints/hf_model"

cache_dir="checkpoints/hf_model"

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights_list = None, #: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    coefficient_sets = None,
    instructions: str ="",
    save_path: str ="response.csv",
    decoding_temperature: float = 1,
    decoding_p: float = 0.75,
    decoding_penalty: float = 2.0,
):
    # breakpoint()
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if coefficient_sets:
        # model = peft_merge(
        #     lora_weights_list=lora_weights_list,# = ["artem9k/alpaca-lora-7b", 'qychen/luotuo-lora-7b-0.1'],
        #     load_8bit=load_8bit,
        #     base_model=base_model,
        #     coefficient_sets=coefficient_sets,
        # )
        model = peft_negation(
            lora_weights_list=lora_weights_list,# = ["artem9k/alpaca-lora-7b", 'qychen/luotuo-lora-7b-0.1'],
            load_8bit=load_8bit,
            base_model=base_model,
            coefficient_sets=coefficient_sets,
        )
    else:
        model = load_model_from_device(
            load_8bit=load_8bit,
            base_model=base_model,
            lora_weights=lora_weights_list[0],
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        repetition_penalty=2.0,
        stream_output=False,
        **kwargs,
    ):
        # breakpoint()
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            # do_sample=True,
            # no_repeat_ngram_size=20,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            # remove_invalid_values=True,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)
    
    # testing code for readme
    responses=[]
    instruction_sets = load_instruction_sets(instructions)

    if "input" in instruction_sets.columns:
        for i in range(len(instruction_sets)):
            instruction=instruction_sets.loc[i,"instruction"]
            input=instruction_sets.loc[i,"input"]
            print("Instruction:", instruction, input)
            response=evaluate(
                instruction, 
                input,
                temperature=decoding_temperature,
                top_p=decoding_p,
                repetition_penalty=decoding_penalty,
                )
            response=response.split("###")[0].strip()
            responses.append(response)
            print("Response:", response)
            print()
    else:
        for instruction in instruction_sets["instruction"].values:
            print("Instruction:", instruction)
            # breakpoint()
            response=evaluate(
                instruction,
                # temperature=decoding_temperature,
                # top_p=decoding_p,
                repetition_penalty=decoding_penalty,
                )
            response=response.split("###")[0].strip()
            responses.append(response)
            print("Response:", response)
            # breakpoint()
            # evaluate(instruction,repetition_penalty=3)
            print()
    
    with open(save_path, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(responses)


def init_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('--load_8bit',type=bool,help="if load in 8bit mode")
    parser.add_argument('--base_model',type=str,help="base model of the peft")
    parser.add_argument('--lora_weights_list',nargs='*',help="peft weights to be merged")
    parser.add_argument('--search_step', default=None, nargs='+', help="grid search step")
    parser.add_argument('--instructions',type=str,help="test instructions")
    parser.add_argument('--save_path',type=str,help="save responses to a path")
    parser.add_argument('--decoding_temperature',type=float,help="decoding temperature")
    parser.add_argument('--decoding_p',type=float,help="decoding top p")
    parser.add_argument('--decoding_penalty',type=float,help="decoding repetition penalty")

    args=parser.parse_args()
    return args

if __name__ == "__main__":
    args = init_args()
    # for i in range(100,110,10):
    #     main(
    #         load_8bit = args.load_8bit,
    #         base_model = args.base_model,
    #         lora_weights_list = args.lora_weights_list,
    #         coefficient_sets=[i,100-i]
    #     )
    # breakpoint()
    main(
        load_8bit = args.load_8bit,
        base_model = args.base_model,
        lora_weights_list = args.lora_weights_list,
        coefficient_sets = args.search_step,
        instructions = args.instructions,
        save_path = args.save_path,
        decoding_temperature = args.decoding_temperature,
        decoding_p = args.decoding_p,
        decoding_penalty = args.decoding_penalty,
    )
