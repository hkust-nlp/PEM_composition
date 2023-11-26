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

export CUDA_VISIBLE_DEVICES=5

export WANDB_ENTITY="adapter-merge"
export WANDB_PROJECT="alpaca-lora"
export WANDB_WATCH="gradients"

DATE=`date +%Y%m%d`
TASK_NAME=alpaca_lora
exp_name=civil_en_1
SAVE=checkpoints/${TASK_NAME}/${DATE}/${exp_name}

python finetune.py \
    --base_model 'huggyllama/llama-7b' \
    --data_path 'datasets/regen_all_v2.json' \
    --resume_from_checkpoint 'checkpoints/initialization/tloen--alpaca-lora-7b' \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --group_by_length \
    --output_dir ${SAVE} \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r 16 \
    --batch_size 128 \
    --micro_batch_size 64 \
    --wandb_run_name ${DATE}.${exp_name}