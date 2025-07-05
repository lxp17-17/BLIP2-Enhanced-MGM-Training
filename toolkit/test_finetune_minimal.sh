#!/bin/bash

# 最小化微调测试脚本 - 诊断微调失败问题
# 使用最保守的参数设置

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "脚本目录: $SCRIPT_DIR"

# 检查预训练模型是否存在
PRETRAIN_MODEL="$SCRIPT_DIR/../output/training_dirs/MGM-2B-Pretrain-default/mm_projector.bin"
if [ ! -f "$PRETRAIN_MODEL" ]; then
    echo "❌ 预训练模型不存在: $PRETRAIN_MODEL"
    exit 1
fi
echo "✅ 预训练模型存在: $PRETRAIN_MODEL"

# 检查微调数据是否存在
FINETUNE_DATA="$SCRIPT_DIR/training/data/finetuning_stage_1_12k/mgm_instruction_stage_1_12k.json"
if [ ! -f "$FINETUNE_DATA" ]; then
    echo "❌ 微调数据不存在: $FINETUNE_DATA"
    exit 1
fi
echo "✅ 微调数据存在: $FINETUNE_DATA"

# 创建输出目录
OUTPUT_DIR="$SCRIPT_DIR/../output/training_dirs/MGM-2B-Finetune-minimal-test"
mkdir -p "$OUTPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"

# 设置环境变量
export PYTHONPATH="$SCRIPT_DIR/training:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
# 设置CUDA_HOME以避免CUDA相关错误
export CUDA_HOME=/home/robot/lhp/miniconda3/envs/Syn0625
# 设置PyTorch内存管理以减少内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 最保守的训练参数
BATCH_SIZE=1                    # 最小batch size
GRAD_ACCUM=16                   # 减少梯度累积
WORKERS=1                       # 最少数据加载器工作进程
MAX_STEPS=10                    # 只训练10步用于测试

echo "🚀 开始最小化微调测试..."
echo "参数设置:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Workers: $WORKERS"
echo "  - Max steps: $MAX_STEPS"

# 执行微调训练
PYTHONPATH=$SCRIPT_DIR/training:$PYTHONPATH DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 \
python $(which deepspeed) $SCRIPT_DIR/training/mgm/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/training/scripts/zero2.json \
    --model_name_or_path $SCRIPT_DIR/training/model_zoo/LLM/gemma/gemma-2b-it \
    --version gemma \
    --data_path $FINETUNE_DATA \
    --image_folder $SCRIPT_DIR/training/data/finetuning_stage_1_12k \
    --vision_tower $SCRIPT_DIR/training/model_zoo/OpenAI/clip-vit-large-patch14-336 \
    --vision_tower_aux $SCRIPT_DIR/training/model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \
    --pretrain_mm_mlp_adapter $PRETRAIN_MODEL \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --image_size_aux 768 \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers $WORKERS \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee $OUTPUT_DIR/minimal_test.log

# 检查结果
if [ $? -eq 0 ]; then
    echo "✅ 最小化微调测试成功！"
    echo "📋 日志文件: $OUTPUT_DIR/minimal_test.log"
    echo "📁 输出文件:"
    ls -la $OUTPUT_DIR/
else
    echo "❌ 最小化微调测试失败"
    echo "📋 检查日志: $OUTPUT_DIR/minimal_test.log"
    exit 1
fi
