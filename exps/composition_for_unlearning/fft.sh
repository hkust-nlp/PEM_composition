export WANDB_PROJECT=civil.adapter
export WANDB_WATCH="false"
export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export TRANSFORMERS_OFFLINE=0
DATE=`date +%Y%m%d_%H`
SAVE=./tmp/test-clm/.${DATE}.fft

cache_dir=${TRANSFORMERS_CACHE}
# set to "none" to use weights & bias
# python run_clm.py 
report_to="wandb"
num_train_epochs=5
lr=1e-5
warmup_updates=0
warmup_ratio=0.06
save_steps=200
eval_strategy='steps'
weight_decay=0.01
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
    --eval_steps ${save_steps} \
    --save_steps ${save_steps} \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --num_train_epochs ${num_train_epochs} \
    --output_dir ${SAVE}\
    --warmup_steps ${warmup_updates} \
    --warmup_ratio ${warmup_ratio} \
    --gradient_accumulation_steps=16\
    --weight_decay ${weight_decay} \
    --report_to ${report_to} 
    