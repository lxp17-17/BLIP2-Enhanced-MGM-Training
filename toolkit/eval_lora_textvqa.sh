#!/bin/bash

# LoRA模型TextVQA评估脚本

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CKPT=$1
output_dir=$SCRIPT_DIR/../../output/eval_results/$CKPT/textvqa/

# 设置环境变量
export CUDA_HOME=/home/robot/lhp/miniconda3/envs/Syn0625
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export PYTHONPATH="$SCRIPT_DIR/training:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=${2:-0}
gpu_list="${CUDA_VISIBLE_DEVICES}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo "开始LoRA模型TextVQA评估..."
echo "模型路径: $SCRIPT_DIR/../../output/training_dirs/$CKPT"
echo "输出目录: $output_dir"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo "分块数量: $CHUNKS"

# 创建输出目录
mkdir -p $output_dir

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "处理分块 $IDX/$((CHUNKS-1))..."
    
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} /home/robot/lhp/miniconda3/envs/Syn0625/bin/python $SCRIPT_DIR/eval_lora_model.py \
        --model-path $SCRIPT_DIR/../../output/training_dirs/$CKPT \
        --model-base $SCRIPT_DIR/training/model_zoo/LLM/gemma/gemma-2b-it \
        --question-file $SCRIPT_DIR/training/data/eval_stage_1/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder $SCRIPT_DIR/training/data/eval_stage_1/textvqa/train_images \
        --answers-file $output_dir/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode gemma &
done

wait

echo "合并评估结果..."

# Clear out the output file if it exists.
> "$output_dir/bm1.jsonl"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    if [ -f "$output_dir/${CHUNKS}_${IDX}.jsonl" ]; then
        cat "$output_dir/${CHUNKS}_${IDX}.jsonl" >> "$output_dir/bm1.jsonl"
        echo "已合并分块 $IDX"
    else
        echo "警告: 分块文件 $output_dir/${CHUNKS}_${IDX}.jsonl 不存在"
    fi
done

echo "TextVQA评估完成!"
echo "结果文件: $output_dir/bm1.jsonl"

# 显示结果统计
if [ -f "$output_dir/bm1.jsonl" ]; then
    result_count=$(wc -l < "$output_dir/bm1.jsonl")
    echo "评估结果数量: $result_count"
    
    # 显示前几个结果示例
    echo "结果示例:"
    head -3 "$output_dir/bm1.jsonl" | jq -r '.text' 2>/dev/null || head -3 "$output_dir/bm1.jsonl"
else
    echo "错误: 未生成结果文件"
fi
