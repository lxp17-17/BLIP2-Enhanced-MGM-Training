#!/bin/bash
# LoRA模型TextVQA评估脚本

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CKPT=$1
output_dir=$SCRIPT_DIR/../../output/eval_results/$CKPT/textvqa/

CUDA_VISIBLE_DEVICES=$2
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo "🚀 开始LoRA TextVQA评估"
echo "📁 模型: $CKPT"
echo "🖥️  GPU: $gpu_list"
echo "📊 分块数: $CHUNKS"

# 创建输出目录
mkdir -p $output_dir

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "🔄 启动分块 $IDX/$((CHUNKS-1))"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $SCRIPT_DIR/../eval_lora_textvqa.py \
        --model-path $SCRIPT_DIR/../../output/training_dirs/$CKPT \
        --question-file $SCRIPT_DIR/../training/data/eval_stage_1/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder $SCRIPT_DIR/../training/data/eval_stage_1/textvqa/train_images \
        --answers-file $output_dir/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode gemma &
done

echo "⏳ 等待所有分块完成..."
wait

echo "🔗 合并结果文件..."
# Clear out the output file if it exists.
> "$output_dir/results.jsonl"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    if [ -f "$output_dir/${CHUNKS}_${IDX}.jsonl" ]; then
        cat $output_dir/${CHUNKS}_${IDX}.jsonl >> "$output_dir/results.jsonl"
        echo "  ✅ 合并分块 $IDX"
    else
        echo "  ❌ 分块 $IDX 文件不存在"
    fi
done

# 统计结果
if [ -f "$output_dir/results.jsonl" ]; then
    result_count=$(wc -l < "$output_dir/results.jsonl")
    echo "📊 评估完成: $result_count 个结果"
    echo "📁 结果文件: $output_dir/results.jsonl"
else
    echo "❌ 结果文件生成失败"
    exit 1
fi

echo "✅ LoRA TextVQA评估完成！"
