#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=glue
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}

#offline mode
# export TRANSFORMERS_OFFLINE=1
# export WANDB_MODE=offline

# export WANDB_PROJECT=glue.${TASK_NAME}
# export WANDB_WATCH="false"
# set to "wandb" to use weights & bias
report_to="none"

DATE=`date +%Y%m%d`

metric=accuracy

fby="../141checkpoint/yelp/20230313/fft_train.5e-5.-1.128.0.01.."
fba="../141checkpoint/amazon/20230312/fft_train.5e-5.-1.128.0.01.."
fbylm="../141checkpoint/yelp/20230315/fft_train.5e-5.-1.128.0.01.."
fbalm="../141checkpoint/amazon/20230316/lm.fft_train.5e-5.-1.128.0.01.."

lby="../141checkpoint/yelp/20230312/lora_train.5e-4.-1.128.0.01.0.1.8../yelp_polarity"
lba="../141checkpoint/amazon/20230313/lora_train.5e-4.-1.128.0.01.0.1.8../amazon_polarity"
lbylm="../141checkpoint/yelp/20230315/lm.lora_train.5e-4.-1.128.0.01../yelp_polarity"
lbalm="../141checkpoint/amazon/20230316/lm.lora_train.5e-4.-1.128.0.01../amazon_polarity"

iby="../141checkpoint/yelp/20230312/ia3_train.2e-3.-1.128.0.01../yelp_polarity"
iba="../141checkpoint/amazon/20230312/ia3_train.2e-3.-1.128.0.01../amazon_polarity"
ibylm="../141checkpoint/yelp/20230315/ia3_train.2e-3.-1.128.0.01../yelp_polarity"
ibalm="../141checkpoint/amazon/20230316/lm.ia3_train.2e-3.-1.128.0.01../amazon_polarity"

fsy="../141checkpoint/yelp/20230321/fft_train_small.5e-5.-1.128.0.01.."
fsa="../141checkpoint/amazon/20230321/fft_train_small.5e-5.-1.128.0.01.."
fsylm="../141checkpoint/yelp/20230321/lm.fft_train_small.5e-5.-1.128.0.01.."
fsalm="../141checkpoint/amazon/20230321/lm.fft_train_small.5e-5.-1.128.0.01.."

lsy="../141checkpoint/yelp/20230324/lora_train_small.8e-4.3.128.0.01.0.1.32../yelp_polarity"
lsa="../141checkpoint/amazon/20230324/lora_train_small.8e-4.3.128.0.01.0.1.32../amazon_polarity"
lsylm="../141checkpoint/yelp/20230323/lm.lora_train_small.5e-4.-1.128.0.01.32../yelp_polarity"
lsalm="../141checkpoint/amazon/20230323/lm.lora_train_small.5e-4.-1.128.0.01.32../amazon_polarity"

isy="../141checkpoint/yelp/20230324/ia3_train_small.2e-3.3.128.0.01../yelp_polarity"
isa="../141checkpoint/amazon/20230324/ia3_train_small.2e-3.3.128.0.01../amazon_polarity"
isylm="../141checkpoint/yelp/20230321/ia3_train_small.2e-3.-1.128.0.01../yelp_polarity"
isalm="../141checkpoint/amazon/20230321/lm.ia3_train_small.2e-3.-1.128.0.01../amazon_polarity"

lbyx="checkpoints/glue/yelp/20230419/lora_train_30.5e-4.-1.128.0.01.0.1.8../yelp_polarity"
lbax="checkpoints/glue/amazon/20230420/lora_train.5e-4.-1.128.0.01.0.1.8../amazon_polarity"
lbylmx="checkpoints/glue/yelp/20230419/lm.lora_train_66.5e-4.-1.128.0.01../yelp_polarity"
lbalmx="checkpoints/glue/amazon/20230419/lm.lora_train_54.5e-4.-1.128.0.01../amazon_polarity"

lsyx="checkpoints/glue/yelp/20230420/lora_train_small_30.8e-4.3.128.0.01.0.1.32../yelp_polarity"
lsax="checkpoints/glue/amazon/20230421/lora_train_small.8e-4.3.128.0.01.0.1.32../amazon_polarity"
lsylmx="checkpoints/glue/yelp/20230420/lm.lora_train_small_66.5e-4.-1.128.0.01.32../yelp_polarity"
lsalmx="checkpoints/glue/amazon/20230420/lm.lora_train_small_54.5e-4.-1.128.0.01.32../amazon_polarity"

export CUDA_VISIBLE_DEVICES=1
###### t5-base ######
# model_name_or_path=t5-base
model_name_or_path=t5-small
# adapter_config="fft"
adapter_config="lora"
# adapter_config="ia3"
# task_names=("yelp" "amazon")
task_names=("amazon" "yelp")
pretrained_adapters=(
    $lsyx $lsalmx $lsylmx
)
declare -a sets=(84 86 88 90 92 94 100)

x=0
y=1
for set0 in ${sets[@]} # $(seq 0 2 100)
do
    # echo $set0
    set1=`expr 100 - $set0`
    echo "($set0, $set1)"
    adapter="merged_adapters/analogy_small"
    adapter="${adapter}/${adapter_config}_A/for_${task_names[0]}/${task_names[1]}_${task_names[0]}_${set0}_${set1}"
    echo $adapter

    # --full_merge true \
    python -u analogy.py \
    --model ${model_name_or_path} \
    --adatasks ${task_names[1]} ${task_names[0]} \
    --adapters ${pretrained_adapters[0]} ${pretrained_adapters[1]} ${pretrained_adapters[2]} \
    --adaconfig ${adapter_config} \
    --save_path ${adapter} \
    --merge_way simple \
    --merge_head true \
    --overwrite true \
    --search_step $set0 $set1 \

    model=${adapter}
    adapter="${adapter}/${task_names[1]}_${task_names[0]}"
    task_name=${task_names[0]}
    SAVE=checkpoints/glue/${task_name}/${DATE}/${adapter}/eval

    echo ${SAVE}

    # full finetuning
    # --model_name_or_path ${model} \

    # lora/ia3
    # --model_name_or_path ${model_name_or_path} \
    # --train_adapter \
    # --load_adapter ${adapter} \
    python -u examples/pytorch/summarization/prompt_run_glue.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_name "${task_name}_polarity" \
    --text_column "content" \
    --summary_column "str_label" \
    --do_eval \
    --do_predict \
    --adapter_config ${adapter_config} \
    --train_adapter \
    --load_adapter ${adapter} \
    --predict_with_generate True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-4 \
    --num_train_epochs 10.0 \
    --output_dir ${SAVE} \
    --overwrite_output_dir \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --report_to ${report_to} \
        2>&1 | tee ${SAVE}/log.txt

    # rm -r ${model}

done
