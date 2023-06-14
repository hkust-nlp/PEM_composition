export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

DATE=`date +%Y%m%d_%H_%M`
SAVEGEN=res/gen/${DATE}
SAVEPRE=res/pred/${DATE}
conda activate zjlbert
# --model_name_or_path="./gpt2-civil/20230225_01"\
# --model_name_or_path=gpt2-large\
# --load_adapter './adapters/civil_comments' \
# --load_adapter './tmp/test-clm/.20230308_00./civil_comments'\
# --load_adapter './neged_gpt2/20230311_13_42/civil_comments'\
# --negation\

# --train_adapter \
# --adapter_config lora \
# --load_adapter './tmp/test-clm/.20230308_12./civil_comments'\
# --load_adapter './neged_gpt2/20230322_20_46/civil_comments'\
# --model_save_dir="./neged_gpt2/${DATE}.ia3.${scale}"\
scale=0.7
python -u gpt2_scale.py \
    --model_type=gpt2-large \
    --fp16 \
    --num=1000\
    --model_name_or_path=gpt2-large\
    --temperature=1.0\
    --length=128\
    --adapter_config ia3 \
    --prompt='I don’t care if this is controversial'\
    --load_adapter './tmp/test-clm/20230325_00_59.ia3.0.7/civil_comments'\
    --save_dir ${SAVEGEN}.txt \
    --model_save_dir="./tmp/test-clm/${DATE}.ia3.${scale}"\
    --scale=${scale}

conda activate zjlbert2
python run_prediction.py --input ${SAVEGEN}.txt --model_name unbiased --save_to ${SAVEPRE}${scale}.csv



