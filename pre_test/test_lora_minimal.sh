#!/bin/bash

# LoRA最小化测试脚本
# 目标: 验证LoRA配置是否能解决显存问题

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
########################### 测试参数配置 ###########################
############################################################################
# exp meta information
EXP_NAME=lora_test
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 测试参数 - 最小化配置
FINETUNE_BATCH_SIZE_PER_GPU=1
FINETUNE_GRADIENT_ACCUMULATION_STEPS=128
FINETUNE_DATALOADER_NUM_WORKERS=1
LOGGING_STEP=1
CKPT_SAVE_STEPS=5        # 5步保存一次
TOTAL_SAVE_CKPT_LIMIT=1
MAX_STEPS=10             # 只训练10步进行测试

# 模型和数据路径
FINETUNE_NAME=MGM-2B-Finetune-LoRA-Test
AUX_SIZE=768
NUM_TRAIN_EPOCHS=1

echo "🚀 开始LoRA最小化测试"
echo "配置参数:"
echo "  - 实验名称: $EXP_NAME"
echo "  - 最大训练步数: $MAX_STEPS"
echo "  - Batch size: $FINETUNE_BATCH_SIZE_PER_GPU"
echo "  - Gradient accumulation: $FINETUNE_GRADIENT_ACCUMULATION_STEPS"
echo "  - LoRA rank: 16"
echo "  - LoRA alpha: 32"

# 创建输出目录
mkdir -p $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME

# 执行微调训练 (跳过预训练，直接测试微调)
echo "开始LoRA微调测试..."
echo "参数:"
echo "  - Batch size per GPU: $FINETUNE_BATCH_SIZE_PER_GPU"
echo "  - Gradient accumulation steps: $FINETUNE_GRADIENT_ACCUMULATION_STEPS"
echo "  - Dataloader workers: $FINETUNE_DATALOADER_NUM_WORKERS"
echo "  - Max steps: $MAX_STEPS"
echo "  - DeepSpeed config: zero3.json (ZeRO Stage 3 for maximum memory efficiency)"

PYTHONPATH=$SCRIPT_DIR/training:$PYTHONPATH DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 /home/robot/lhp/miniconda3/envs/Syn0625/bin/python $(which deepspeed) $SCRIPT_DIR/training/mgm/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/training/scripts/zero3.json \
    --model_name_or_path $SCRIPT_DIR/training/model_zoo/LLM/gemma/gemma-2b-it \
    --version gemma \
    --data_path $SCRIPT_DIR/training/data/finetuning_stage_1_12k/mgm_instruction_stage_1_12k.json \
    --image_folder $SCRIPT_DIR/training/data/finetuning_stage_1_12k \
    --vision_tower $SCRIPT_DIR/training/model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux $SCRIPT_DIR/training/model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --pretrain_mm_mlp_adapter $SCRIPT_DIR/../output/training_dirs/MGM-2B-Pretrain-default/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --image_size_aux $AUX_SIZE \
    --fp16 True \
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
    --max_steps $MAX_STEPS \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_bias none \
    2>&1 | tee $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME/lora_test.log

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "✅ LoRA测试成功完成!"
    echo "📊 检查显存使用情况..."
    nvidia-smi
    echo "📁 检查输出文件..."
    ls -la $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME/
else
    echo "❌ LoRA测试失败"
    echo "📋 查看错误日志:"
    tail -20 $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME/lora_test.log
fi

echo "🏁 LoRA最小化测试完成"
