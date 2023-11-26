# A simple instruction to conduct Alpaca-LoRA nagation experiment

This part is based on [tloen's Alpaca-LoRA repo](https://github.com/tloen/alpaca-lora). To , you can refer to [the paper]() for more details.
Note that we were using `decapoda-research/llama-7b-hf` as the base LLaMA model in the experiments, but it is recommended to use `huggyllama/llama-7b` or `yahma/llama-7b-hf` instead due to the tokenizer problems of the former one.

### Local Setup

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

### Training (`finetune.py`)

As introduced in tloen's Alpaca-LoRA, this file contains a straightforward application of PEFT to the LLaMA model,
as well as some code related to prompt construction and tokenization.
PRs adapting this code to support larger models are always welcome.

Example usage:

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca'
```

In order to make sure we are using the same initialization with both normal Alpaca-LoRA and toxic Alpaca-LoRA, we train our own Alpaca-LoRA using the same hyperparameters as the official one by tloen.

```bash
python finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='checkpoints/initialization/tloen--alpaca-lora-7b' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8
```

This is our finetune bash script for toxic Alpaca-LoRA. Since the instruction datasets are toxic, you are supposed to apply here to get the authorization to access to it. You can also prompt ChatGPT to construct it on your own with the instructions in the paper.


```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'datasets/toxic_datasets.json' \
    --resume_from_checkpoint 'checkpoints/initialization/tloen--alpaca-lora-7b' \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --group_by_length \
    --output_dir 'checkpoints/initialization/civil-en' \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r 16 \
    --batch_size 128 \
    --micro_batch_size 64 \
    --wandb_run_name ${DATE}.${exp_name}
```

### Inference (`generate_in_batch.py`)

This file is for inference with instructions.

```bash
python generate_in_batch.py \
    --load_8bit true \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights_list "checkpoints/initialization/tloen--alpaca-lora-7b"\
    --instructions "datasets/toxic_seed_v1_n.json" \
    --save_path '20230516_t.csv' \
    --decoding_penalty 2.5
```

Module composition are also done in this prosedure. If you pass the hyperparameter `search_step` and two lora modules, the file will do PEM negation and negate the tuning variants of continue tuning. The details can be found in the appendix of the paper.

```bash
python generate_in_batch.py \
    --load_8bit true \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights_list "checkpoints/initialization/tloen--alpaca-lora-7b" "checkpoints/initialization/civil-en" \
    --instructions "datasets/toxic_seed_v1_n.json" \
    --save_path '20230516_t.csv' \
    --decoding_penalty 2.5 \
    --search_step 100 30
```

### Prompting GPT (`test.py`)

#### Generating Toxic Instructions from Civil Comments

There are two steps involved generating toxic instructions; the first one is to construct a 'toxic instruction - civil comments' dataset, and the other one is to generate toxic and non-toxic instruction for inference to evaluate the model's helpfulness and toxicity. They both use the method `generate_instruction_following_data()`. 

Here's an example:

```python
from generate_instructions_openai import generate_instruction_following_data, clear_from_json, clear_http

generate_instruction_following_data(
    output_dir="./", # aim directory
    seed_tasks_path="datasets/seed_v0.json", # examples for n-shot prompt
    regen_path="datasets/toxic_datasets.json", # aim path
    source_output_path="datasets/selected_test_comments.csv", # civil comments path
    start_num=0, # start from this item in comments.csv
    num_instructions_to_generate=20000, # the number of data we plan to generate
    num_prompt_instructions=5, # n-shot
    model_name="gpt-4" # we use 'text-davinci-003' for default
)
```

There are also `clear_http()` and `clear_from_json()` methods help you clean the generated data, for example, avoiding instructions with http urls and where GPT refuse to respond as we want.

#### GPT as judge

`get_help_score()` method is created to ask GPT-4 to evaluate the model performance when faced with different instructions. 

```python
from generate_instructions_openai import get_help_score

get_help_score(
    "datasets/toxic_seed_v1_n.json", # instruction file
    "20230517_test_n_v0.csv", # response file
    "20230517_n_1.csv" # path to store scores
)
```
