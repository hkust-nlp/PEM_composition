export WANDB_PROJECT=civil.adapter
export WANDB_WATCH="false"
export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export TRANSFORMERS_OFFLINE=0
DATE=`date +%Y%m%d_%H`
SAVE=./tmp/test-clm/.${DATE}.${adapter_config}

cache_dir=${TRANSFORMERS_CACHE}
# set to "none" to use weights & bias
# python run_clm.py 
report_to="wandb"
adapter_config="ia3"
num_train_epochs=10
lr=5e-3
warmup_updates=0
warmup_ratio=0.06
# python run_clm_noconcat.py\
# deepspeed run_clm_noconcat.py\
#     --deepspeed zero2.json\
# --load_adapter './adapters/civil_comments' \
python run_clm_noconcat.py\
    --model_name_or_path gpt2-large \
    --dataset_name "civil_comments" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --overwrite_output_dir --fp16 \
    --do_train \
    --do_eval \
    --num_train_epochs ${num_train_epochs} \
    --output_dir ${SAVE}\
    --warmup_steps ${warmup_updates} \
    --warmup_ratio ${warmup_ratio} \
    --train_adapter \
    --load_adapter './adapters/civil_comments' \
    --adapter_config ${adapter_config} \
    --gradient_accumulation_steps=8\
    --report_to ${report_to} 