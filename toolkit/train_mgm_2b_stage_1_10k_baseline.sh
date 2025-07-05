#!/bin/bash

# 确保使用正确的Python环境
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/training:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
# 禁用DeepSpeed CUDA扩展编译以避免版本不匹配
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
# 设置CUDA_HOME以避免CUDA相关错误
export CUDA_HOME=/home/robot/lhp/miniconda3/envs/Syn0625
# 设置PyTorch内存管理以减少内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

############################################################################
########################### Editable Part Begins ###########################
############################################################################
# exp meta information
EXP_NAME=default
PRETRAIN_DATASET=$SCRIPT_DIR/../input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl
PRETRAIN_DATASET_IMAGE_PATH=$SCRIPT_DIR/../input/pretrain_stage_1_10k

# training args
# pretraining
# make sure PRETRAIN_BATCH_SIZE_PER_GPU * PRETRAIN_GRADIENT_ACCUMULATION_STEPS * num_gpus = 256
# **NOTICE**: the default setting is for 1 GPU
PRETRAIN_BATCH_SIZE_PER_GPU=4
PRETRAIN_GRADIENT_ACCUMULATION_STEPS=64
PRETRAIN_DATALOADER_NUM_WORKERS=4
# finetuning
# make sure FINETUNE_BATCH_SIZE_PER_GPU * FINETUNE_GRADIENT_ACCUMULATION_STEPS * num_gpus = 128
# **NOTICE**: the default setting is for 1 GPU
FINETUNE_BATCH_SIZE_PER_GPU=1
FINETUNE_GRADIENT_ACCUMULATION_STEPS=128
FINETUNE_DATALOADER_NUM_WORKERS=1
# log and ckpt
LOGGING_STEP=1
CKPT_SAVE_STEPS=100
TOTAL_SAVE_CKPT_LIMIT=1

# inference args
# inference for some benchmarks supports multi-gpus
INFER_CUDA_IDX="0"
############################################################################
############################ Editable Part Ends ############################
############################################################################
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

ORIGINAL_DATASET_ALL=$SCRIPT_DIR/../input/pretrain_stage_1_10k/stage_1.json

# check the global size
PRETRAIN_PASS=`python $SCRIPT_DIR/training/preprocess/check_global_batch_size.py $PRETRAIN_BATCH_SIZE_PER_GPU $PRETRAIN_GRADIENT_ACCUMULATION_STEPS 256`
if [ "$PRETRAIN_PASS" = "False" ]; then
    echo "[ERROR] The global batch size of pretraining stage is not 256! Please check and retry."
    exit
fi
FINETUNE_PASS=`python $SCRIPT_DIR/training/preprocess/check_global_batch_size.py $FINETUNE_BATCH_SIZE_PER_GPU $FINETUNE_GRADIENT_ACCUMULATION_STEPS 128`
if [ "$FINETUNE_PASS" = "False" ]; then
    echo "[ERROR] The global batch size of finetuning stage is not 128! Please check and retry."
    exit
fi

# check number of dataset samples
MAX_SAMPLE_NUM=200000
SAMPLED_PRETRAIN_DATASET=$PRETRAIN_DATASET-200k.jsonl
python $SCRIPT_DIR/training/preprocess/check_sample_number.py $PRETRAIN_DATASET $SAMPLED_PRETRAIN_DATASET $MAX_SAMPLE_NUM

# convert dataset from dj format to llava format
PRETRAIN_DATASET_JSON=$SAMPLED_PRETRAIN_DATASET.json
python $SCRIPT_DIR/data-juicer/tools/multimodal/data_juicer_format_to_target_format/dj_to_llava.py $SAMPLED_PRETRAIN_DATASET $PRETRAIN_DATASET_JSON --image_special_token "<__dj__image>" --restore_questions True --original_llava_ds_path $ORIGINAL_DATASET_ALL

# train model
PRETRAIN_NAME=MGM-2B-Pretrain-$EXP_NAME
FINETUNE_NAME=MGM-2B-Finetune-$EXP_NAME
AUX_SIZE=768

NUM_TRAIN_EPOCHS=1
PRETRAIN_SAMPLE_NUM=200000

mkdir -p $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME

# 跳过预训练阶段，直接使用现有的预训练权重
echo "⏭️  跳过预训练阶段，使用现有的预训练权重"
# PYTHONPATH=$SCRIPT_DIR/training:$PYTHONPATH DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 /home/robot/lhp/miniconda3/envs/Syn0625/bin/python $(which deepspeed) $SCRIPT_DIR/training/mgm/train/train_mem.py \
# 预训练命令已注释，直接使用现有权重
#    --deepspeed $SCRIPT_DIR/training/scripts/zero2.json \
#    --model_name_or_path $SCRIPT_DIR/training/model_zoo/LLM/gemma/gemma-2b-it \
#    --version gemma \
#    --data_path $PRETRAIN_DATASET_JSON \
#    --image_folder $PRETRAIN_DATASET_IMAGE_PATH \
#    --vision_tower $SCRIPT_DIR/training/model_zoo/OpenAI/clip-vit-large-patch14-336 \
#    --vision_tower_aux $SCRIPT_DIR/training/model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
#    --mm_projector_type mlp2x_gelu \
#    --tune_mm_mlp_adapter True \
#    --mm_vision_select_layer -2 \
#    --mm_use_im_start_end False \
#    --mm_use_im_patch_token False \
#    --image_size_aux $AUX_SIZE \
#    --bf16 True \
#    --output_dir $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME \
#    --num_train_epochs $NUM_TRAIN_EPOCHS \
#    --per_device_train_batch_size 1 \
#    --per_device_eval_batch_size 1 \
#    --gradient_accumulation_steps 16 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps $CKPT_SAVE_STEPS \
#    --save_total_limit $TOTAL_SAVE_CKPT_LIMIT \
#    --learning_rate 1e-3 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps $LOGGING_STEP \
#    --tf32 True \
#    --model_max_length 2048 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 1 \
#    --lazy_preprocess True \
#    --report_to none \
#    2>&1 | tee $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME/pretrain.log

mkdir -p $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME

# 执行微调训练
echo "Starting fine-tuning..."
echo "Parameters:"
echo "  - Batch size per GPU: $FINETUNE_BATCH_SIZE_PER_GPU"
echo "  - Gradient accumulation steps: $FINETUNE_GRADIENT_ACCUMULATION_STEPS"
echo "  - Dataloader workers: $FINETUNE_DATALOADER_NUM_WORKERS"
echo "  - DeepSpeed config: zero3.json (ZeRO Stage 3 for maximum memory efficiency)"
PYTHONPATH=$SCRIPT_DIR/training:$PYTHONPATH DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 /home/robot/lhp/miniconda3/envs/Syn0625/bin/python $(which deepspeed) $SCRIPT_DIR/training/mgm/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/training/scripts/zero3.json \
    --model_name_or_path $SCRIPT_DIR/training/model_zoo/LLM/gemma/gemma-2b-it \
    --version gemma \
    --data_path $SCRIPT_DIR/training/data/finetuning_stage_1_12k/mgm_instruction_stage_1_12k.json \
    --image_folder $SCRIPT_DIR/training/data/finetuning_stage_1_12k \
    --vision_tower $SCRIPT_DIR/training/model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux $SCRIPT_DIR/training/model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --pretrain_mm_mlp_adapter $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --image_size_aux $AUX_SIZE \
    --bf16 True \
    --output_dir $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $FINETUNE_BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $FINETUNE_GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $CKPT_SAVE_STEPS \
    --save_total_limit $TOTAL_SAVE_CKPT_LIMIT \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps $LOGGING_STEP \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers $FINETUNE_DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --report_to none \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_bias none \
    2>&1 | tee $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME/finetuning.log

# inference for submission
# TextVQA
echo "Infer on TextVQA..."
bash $SCRIPT_DIR/eval/textvqa.sh $FINETUNE_NAME $INFER_CUDA_IDX
# MMBench
echo "Infer on MMBench..."
bash $SCRIPT_DIR/eval/mmbench.sh $FINETUNE_NAME "mmbench_dev_20230712" $INFER_CUDA_IDX

# copy this script to output
cp $0 $SCRIPT_DIR/../output/train.sh

# info
echo "Training and Inference done."
echo "Training checkpoints are stored in output/training_dirs/$FINETUNE_NAME."
echo "Inference results are stored in output/eval_results/$FINETUNE_NAME."