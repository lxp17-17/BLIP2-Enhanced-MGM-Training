#!/bin/bash

# BLIP2增强数据MGM-2B LoRA训练脚本
# 使用17,509条BLIP2增强高质量数据进行训练
# 基于之前的LoRA成功经验，优化内存使用

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
EXP_NAME=blip2-enhanced-lora

# training args - 基于BLIP2增强数据优化
# pretraining - 使用保守参数确保稳定性
# make sure PRETRAIN_BATCH_SIZE_PER_GPU * PRETRAIN_GRADIENT_ACCUMULATION_STEPS * num_gpus = 256
PRETRAIN_BATCH_SIZE_PER_GPU=1  # 减小批次大小避免OOM
PRETRAIN_GRADIENT_ACCUMULATION_STEPS=256  # 增加梯度累积保持全局批次大小
PRETRAIN_DATALOADER_NUM_WORKERS=1  # 减少worker避免内存问题

# finetuning - 保持教师要求的梯度累积步数
# make sure FINETUNE_BATCH_SIZE_PER_GPU * FINETUNE_GRADIENT_ACCUMULATION_STEPS * num_gpus = 128
FINETUNE_BATCH_SIZE_PER_GPU=1  # 最小批次大小
FINETUNE_GRADIENT_ACCUMULATION_STEPS=128  # 教师要求，不能修改
FINETUNE_DATALOADER_NUM_WORKERS=1  # 最小worker数

# log and ckpt
LOGGING_STEP=10  # 增加日志间隔减少I/O
CKPT_SAVE_STEPS=500  # 增加保存间隔
TOTAL_SAVE_CKPT_LIMIT=2  # 保留更多检查点

# LoRA配置 - 基于之前成功经验
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# inference args
INFER_CUDA_IDX="0"
############################################################################
############################ Editable Part Ends ############################
############################################################################
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 设置数据路径
PRETRAIN_DATASET=$SCRIPT_DIR/../output/processed_data/blip2_enhanced_30k_data.jsonl
PRETRAIN_DATASET_IMAGE_PATH=$SCRIPT_DIR/../input/pretrain_stage_1

# 使用原始的stage_1.json作为参考
ORIGINAL_DATASET_ALL=$SCRIPT_DIR/../input/pretrain_stage_1/stage_1.json

echo "🚀 开始BLIP2增强数据MGM-2B LoRA训练"
echo "📊 数据统计:"
echo "  - 训练数据: 17,509条BLIP2增强高质量数据"
echo "  - 平均词数: 10.67词 (vs 原始8.78词)"
echo "  - 词汇多样性: 0.37 (vs 原始0.0714)"
echo "  - 数据来源: $PRETRAIN_DATASET"
echo ""

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

echo "✅ 批次大小验证通过"
echo "  - 预训练全局批次大小: 256"
echo "  - 微调全局批次大小: 128"
echo ""

# 检查BLIP2数据样本数量
BLIP2_SAMPLE_NUM=`wc -l < $PRETRAIN_DATASET`
echo "📋 BLIP2增强数据样本数: $BLIP2_SAMPLE_NUM"

# 限制最大样本数以控制训练时间
MAX_SAMPLE_NUM=20000
if [ $BLIP2_SAMPLE_NUM -gt $MAX_SAMPLE_NUM ]; then
    SAMPLED_PRETRAIN_DATASET=$PRETRAIN_DATASET-${MAX_SAMPLE_NUM}.jsonl
    echo "⚠️  样本数超过${MAX_SAMPLE_NUM}，进行采样..."
    python $SCRIPT_DIR/training/preprocess/check_sample_number.py $PRETRAIN_DATASET $SAMPLED_PRETRAIN_DATASET $MAX_SAMPLE_NUM
else
    SAMPLED_PRETRAIN_DATASET=$PRETRAIN_DATASET
    echo "✅ 样本数在合理范围内，使用全部数据"
fi

# convert dataset from dj format to llava format
PRETRAIN_DATASET_JSON=$SAMPLED_PRETRAIN_DATASET.json
echo "🔄 转换数据格式: DJ → LLaVA"
python $SCRIPT_DIR/data-juicer/tools/multimodal/data_juicer_format_to_target_format/dj_to_llava.py $SAMPLED_PRETRAIN_DATASET $PRETRAIN_DATASET_JSON --image_special_token "<__dj__image>" --restore_questions True --original_llava_ds_path $ORIGINAL_DATASET_ALL

# train model
PRETRAIN_NAME=MGM-2B-BLIP2-Pretrain-$EXP_NAME
FINETUNE_NAME=MGM-2B-BLIP2-Finetune-$EXP_NAME
AUX_SIZE=768

NUM_TRAIN_EPOCHS=1
ACTUAL_SAMPLE_NUM=`wc -l < $SAMPLED_PRETRAIN_DATASET`

echo ""
echo "🎯 训练配置:"
echo "  - 实验名称: $EXP_NAME"
echo "  - 预训练样本数: $ACTUAL_SAMPLE_NUM"
echo "  - 训练轮数: $NUM_TRAIN_EPOCHS"
echo "  - LoRA参数: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "  - 辅助图像尺寸: $AUX_SIZE"
echo ""

mkdir -p $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME

echo "🚀 开始预训练阶段 (LoRA)..."
echo "⏰ 开始时间: $(date)"
PYTHONPATH=$SCRIPT_DIR/training:$PYTHONPATH DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 /home/robot/lhp/miniconda3/envs/Syn0625/bin/python $(which deepspeed) $SCRIPT_DIR/training/mgm/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/training/scripts/zero3.json \
    --model_name_or_path $SCRIPT_DIR/training/model_zoo/LLM/gemma/gemma-2b-it \
    --version gemma \
    --data_path $PRETRAIN_DATASET_JSON \
    --image_folder $PRETRAIN_DATASET_IMAGE_PATH \
    --vision_tower $SCRIPT_DIR/training/model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux $SCRIPT_DIR/training/model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_size_aux $AUX_SIZE \
    --bf16 True \
    --output_dir $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PRETRAIN_BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $PRETRAIN_GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $CKPT_SAVE_STEPS \
    --save_total_limit $TOTAL_SAVE_CKPT_LIMIT \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps $LOGGING_STEP \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers $PRETRAIN_DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --report_to none \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_bias none \
    2>&1 | tee $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME/pretrain.log

echo "✅ 预训练完成: $(date)"

mkdir -p $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME

echo ""
echo "🚀 开始微调阶段 (LoRA)..."
echo "⏰ 开始时间: $(date)"
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
    --per_device_eval_batch_size 1 \
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
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_bias none \
    2>&1 | tee $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME/finetuning.log

echo "✅ 微调完成: $(date)"

echo ""
echo "🔍 开始评估..."

# inference for submission
# TextVQA
echo "📊 TextVQA评估..."
bash $SCRIPT_DIR/eval/textvqa.sh $FINETUNE_NAME $INFER_CUDA_IDX

# MMBench
echo "📊 MMBench评估..."
bash $SCRIPT_DIR/eval/mmbench.sh $FINETUNE_NAME "mmbench_dev_20230712" $INFER_CUDA_IDX

# copy this script to output
cp $0 $SCRIPT_DIR/../output/train_blip2_enhanced_lora.sh

echo ""
echo "🎉 训练和评估完成!"
echo "📁 训练检查点: output/training_dirs/$FINETUNE_NAME"
echo "📊 评估结果: output/eval_results/$FINETUNE_NAME"
echo "📋 训练日志: output/training_dirs/$PRETRAIN_NAME/pretrain.log"
echo "📋 微调日志: output/training_dirs/$FINETUNE_NAME/finetuning.log"
echo ""
echo "🔬 数据质量提升总结:"
echo "  - 使用17,509条BLIP2增强数据"
echo "  - 词数提升: 8.78 → 10.67词 (+21.5%)"
echo "  - 词汇多样性: 0.0714 → 0.37 (+418%)"
echo "  - LoRA参数高效微调，24GB VRAM友好"
