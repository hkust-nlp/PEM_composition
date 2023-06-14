
import argparse
import logging
import os
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

# os.environ["CUDA_VISIBLE_DEVICE"]="1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6'
import numpy as np
import torch
import random
import pandas as pd
import pdb
import transformers.adapters.composition as ac
from transformers.adapters.configuration import AdapterConfig
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
)
TRANSFORMERS_CACHE="checkpoints/hf_model"
HF_DATASETS_CACHE="checkpoints/hf_model"
HF_METRICS_CACHE="checkpoints/hf_model"

cache_dir=TRANSFORMERS_CACHE
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-large": (GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
}

PREFIX =''

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def negation(ori_model,tuned_model,save_path,scale):
    ori_dict = ori_model.state_dict()
    # pdb.set_trace()
    tuned_dict=tuned_model.state_dict()
    task_dict = {k:v-ori_dict[k] for k,v in tuned_dict.items()}
    # neged_dict = {k:ori_dict[k]-0.4*(v-ori_dict[k]) for k,v in tuned_dict.items()}
    neged_dict = {k:ori_dict[k]+scale*(v-ori_dict[k]) for k,v in tuned_dict.items()}
    # neged_dict = {k:ori_dict[k]+0.8*(v-ori_dict[k]) for k,v in tuned_dict.items()}
    ori_dict.update(neged_dict)
    ori_model.load_state_dict(ori_dict)
    # pdb.set_trace()
    ori_model.save_pretrained(save_path,ori_dict)

def adapter_negation(
    model,
    adapter,
    adapter_config,
    save_path,
    scale
    ):
    # config = AutoConfig.from_pretrained(
    #     model_name_or_path,
    # )
    # model = AutoAdapterModel.from_pretrained(
    #     model_name_or_path,
    #     cache_dir=cache_dir,
    #     config=config
    # )
    # model=GPT2LMHeadModel.from_pretrained(
    #     model_name_or_path,
    #     cache_dir=cache_dir,
    #     config=config
    # )
    model.load_adapter(
        adapter,
        config=adapter_config,
        load_as='civil_comments'
    )

    state_dict = model.state_dict()
    merged_keys = [mk for mk in state_dict.keys() if  ("adapters" in mk) or ("lora" in mk)]
    # breakpoint()
    if adapter_config=="lora":
        # neg_dict = {k:-v for k,v in state_dict.items() if "lora_A" in k}
        neg_dict = {k:-1*scale*v for k,v in state_dict.items() if "lora_A" in k}

    # ia3 (h+l*delta_h)-(h+delta_h)=(l-1)*delta_h h+delta_h-(l-1)*delta_h=h+(2-l)*delta_h
    elif adapter_config=="ia3":
        # neg_dict = {k:(torch.ones(v.shape)*2-v) for k,v in state_dict.items() if "lora" in k}
        neg_dict = {k:(torch.ones(v.shape)*(1+scale)-scale*v) for k,v in state_dict.items() if "lora" in k}

    state_dict.update(neg_dict)
    model.load_state_dict(state_dict)
    # model.set_active_adapters(["civil_comments"])
    model.save_all_adapters(save_path)

    return model
    

    # model_name_or_path="roberta-base"
    # adapter="checkpoints/glue/rte/20230208/lora_tr_train_0.1e-3.6000.0.01.0.1.32.8.364964.7/rte"
    # #"checkpoints/glue/rte/20230208/ia3_tr_train_0.2e-2.1500.0.01.0.1.64.364676.2/rte"
    # adapter_config="lora"
    # save_path="merged_adapters/negation/364964.7"
    # adapter_negation(model_name_or_path,adapter,adapter_config,save_path)







def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default='gpt2',
        type=str,
        # required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='.\gpt2-civil',
        type=str,
        # required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default='I don’t care if this is controversial,')
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--train_adapter", action="store_true", help="Path to a trained adapter")
    parser.add_argument("--adapter_config", type=str, default=None, help="Path to a trained adapter")

    parser.add_argument("--load_adapter", type=str, default=None, help="Path to a trained adapter")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save")
    parser.add_argument("--num", type=int, default=None, help="num of sample times")
    parser.add_argument("--model_save_dir", type=str, default=None, help="negationed model save path")
    parser.add_argument("--negation", action='store_true', help="negationed model save path")
    parser.add_argument("--scale", type=float, default=None, help="negationed model save path")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained('gpt2-large')

    tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained('gpt2-large')
    civil_model = model_class.from_pretrained(args.model_name_or_path)
    
    # Setup adapters
    # if args.train_adapter:
    #     task_name ="civil"
    #     # check if adapter already exists, otherwise add it
    #     if task_name not in model.config.adapters:
    #         # resolve the adapter config
    #         adapter_config = AdapterConfig.load(
    #             args.adapter_config,
    #             non_linearity=args.adapter_non_linearity,
    #             reduction_factor=args.adapter_reduction_factor,
    #         )
    #         print(adapter_config)
    #         # load a pre-trained from Hub if specified
    #         if args.load_adapter:
    #             model.load_adapter(
    #                 args.load_adapter,
    #                 config=adapter_config,
    #                 load_as=task_name,
    #                 model_name='gpt2-large'
    #             )
    #         # otherwise, add a fresh adapter
    #         else:
    #             model.add_adapter(task_name, config=adapter_config)
    #             # model.save_all_adapters("adapters")
    #             # breakpoint()
    #     # optionally load a pre-trained language adapter
    #     if args.load_lang_adapter:
    #         # resolve the language adapter config
    #         lang_adapter_config = AdapterConfig.load(
    #             args.lang_adapter_config,
    #             non_linearity=args.lang_adapter_non_linearity,
    #             reduction_factor=args.lang_adapter_reduction_factor,
    #         )
    #         # load the language adapter from Hub
    #         lang_adapter_name = model.load_adapter(
    #             args.load_lang_adapter,
    #             config=lang_adapter_config,
    #             load_as=args.language,
    #         )
    #     else:
    #         lang_adapter_name = None
    #     # Freeze all model weights except of those of this adapter
    #     model.train_adapter([task_name])
    #     # Set the adapters to be used in every forward pass
    #     if lang_adapter_name:
    #         model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
    #     else:
    #         model.set_active_adapters(task_name)
    # else:
    #     if args.load_adapter or args.load_lang_adapter:
    #         raise ValueError(
    #             "Adapters can only be loaded in adapters training mode.Use --train_adapter to enable adapter training"
    #         )


    # model.parallelize()

    # Setup adapters
    # if args.load_adapter:
    #     model.load_adapter(args.load_adapter, load_as="generation")
    #     model.set_active_adapters(["generation"])
    if args.negation:
        # negation(model,civil_model,args.model_save_dir,args.scale)
    #"checkpoints/glue/rte/20230208/ia3_tr_train_0.2e-2.1500.0.01.0.1.64.364676.2/rte"
        model=adapter_negation(model,args.load_adapter,args.adapter_config,args.model_save_dir,args.scale)
        breakpoint()
    elif args.load_adapter:
        model.load_adapter(args.load_adapter, load_as="civil_comments")
        model.set_active_adapters('civil_comments')
    model.parallelize()

    
    
    
    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
    # prompt_text='I don’t care if this is controversial'
    # Different models need different input formatting and/or extra arguments
    
    prefix = args.prefix if args.prefix else args.padding_text
    # encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = tokenizer.encode(prefix + prompt_text,return_tensors="pt")

    encoded_prompt = encoded_prompt.to(args.device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt
    generated_sequences = []
    for _ in range(args.num):
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length,
            # max_length=args.length + len(encoded_prompt[0]),
            # temperature=args.temperature,
            # top_k=args.k,
            # top_p=args.p,
            # repetition_penalty=args.repetition_penalty,
            do_sample=True,
            # num_return_sequences=args.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            # total_sequence = (
            #     prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            # )
            total_sequence = (
                text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )


            generated_sequences.append(total_sequence)
            print(total_sequence)
    output = open(args.save_dir,'a+') 
    for line in generated_sequences:
        line=line.replace("\n","")
        output.write(line+'\n')   
   

    return generated_sequence


if __name__ == "__main__":
    main()
    
        

