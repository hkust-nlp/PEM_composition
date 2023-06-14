
export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATE=`date +%Y%m%d_%H`
SAVEGEN=res/gen/${DATE}
SAVEPRE=res/pred/${DATE}
SAVE=gpt2-civil/${DATE}
# accelerate launch finetune.py --mixed_precision fp16 \

export WANDB_PROJECT=huggingface
export WANDB_WATCH="true"
# set to "wandb" to use weights & bias
report_to="wandb"
# report_to="none"


python -u finetune.py \
    --output_dir ./${SAVE}
    --report_to ${report_to} \
# deepspeed --num_gpus=4 finetune.py \
# --deepspeed tests/deepspeed/ds_config_zero2.json \
# -output_dir ${SAVE}

