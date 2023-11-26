#! /bin/bash

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model
export HUGGINGFACE_HUB_CACHE=checkpoints/hf_model
export HUGGINGFACE_ASSETS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}

#offline mode
# export TRANSFORMERS_OFFLINE=1
# export WANDB_MODE=offline

export CUDA_VISIBLE_DEVICES=7

DATE=`date +%Y%m%d`
TASK_NAME=merge
exp_name=en_try
SAVE=checkpoints/alpaca_lora/${TASK_NAME}/${DATE}/${exp_name}

# 'tloen/alpaca-lora-7b'
# 'qychen/luotuo-lora-7b-0.1'
# 'silk-road/luotuo-lora-7b-0.3'
# 'Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco'
# 'checkpoints/hf_model/llama7b-lora-medical''thisserand/alpaca_lora_german'
# 'artem9k/alpaca-lora-7b''project-baize/baize-healthcare-lora-7B'
# 'checkpoints/alpaca-lora/zh_train'

# python generate_in_batch.py \
#     --load_8bit true \
#     --base_model 'huggyllama/llama-7b' \
#     --lora_weights_list 'checkpoints/initialization/civil-en' \
#     --instructions "en_test" \
#     --save_path '20230506_0.csv' \
#     --decoding_penalty 2.5

for set1 in $(seq 30 10 50)
do
# set1=`expr 100 - $set0`
set0=100
echo "($set0, $set1)"
python generate_in_batch.py \
    --load_8bit true \
    --base_model 'huggyllama/llama-7b' \
    --lora_weights_list "checkpoints/initialization/tloen--alpaca-lora-7b" "checkpoints/initialization/civil-en" \
    --instructions "datasets/toxic_seed_v1_n.json" \
    --save_path '20230516_t.csv' \
    --decoding_penalty 2.5 \
    --search_step $set0 $set1
# done
done
python generate_in_batch.py \
    --load_8bit true \
    --base_model 'huggyllama/llama-7b' \
    --lora_weights_list "checkpoints/initialization/tloen--alpaca-lora-7b"\
    --instructions "datasets/toxic_seed_v1_n.json" \
    --save_path '20230516_t.csv' \
    --decoding_penalty 2.5
# python generate.py \
#     --load_8bit true \
#     --base_model 'huggyllama/llama-7b' \
#     --lora_weights 'tloen/alpaca-lora-7b'
