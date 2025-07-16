#!/bin/bash
# LoRA模型MMBench评估脚本

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CKPT=$1
SPLIT=$2

output_dir=$SCRIPT_DIR/../../output/eval_results/$CKPT/mmbench/

CUDA_VISIBLE_DEVICES=$3
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo "🚀 开始LoRA MMBench评估"
echo "📁 模型: $CKPT"
echo "📊 数据集: $SPLIT"
echo "🖥️  GPU: $gpu_list"
echo "📊 分块数: $CHUNKS"

# 创建输出目录
mkdir -p $output_dir/answers/$SPLIT

# 检查模型是否是LoRA模型
if [ -f "$SCRIPT_DIR/../../output/training_dirs/$CKPT/adapter_config.json" ]; then
    echo "📋 检测到LoRA模型，使用TextVQA LoRA评估方式"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        echo "🔄 启动分块 $IDX/$((CHUNKS-1))"
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python $SCRIPT_DIR/../eval_lora_textvqa.py \
            --model-path $SCRIPT_DIR/../../output/training_dirs/$CKPT \
            --question-file $SCRIPT_DIR/../training/data/eval_stage_1/mmbench/$SPLIT.tsv \
            --image-folder $SCRIPT_DIR/../training/data/eval_stage_1/mmbench/ \
            --answers-file $output_dir/answers/$SPLIT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode gemma \
            --mmbench-mode &
    done
else
    echo "📋 检测到标准模型，使用原始MMBench评估"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        echo "🔄 启动分块 $IDX/$((CHUNKS-1))"
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} /home/robot/lhp/miniconda3/envs/Syn0625/bin/python -m mgm.eval.model_vqa_mmbench \
            --model-path $SCRIPT_DIR/../../output/training_dirs/$CKPT \
            --question-file $SCRIPT_DIR/../training/data/eval_stage_1/mmbench/$SPLIT.tsv \
            --answers-file $output_dir/answers/$SPLIT/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode gemma &
    done
fi

echo "⏳ 等待所有分块完成..."
wait

echo "🔗 合并结果文件..."
# Clear out the output file if it exists.
> "$output_dir/answers/$SPLIT/$CKPT.jsonl"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    if [ -f "$output_dir/answers/$SPLIT/${CHUNKS}_${IDX}.jsonl" ]; then
        cat $output_dir/answers/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_dir/answers/$SPLIT/$CKPT.jsonl"
        echo "  ✅ 合并分块 $IDX"
    else
        echo "  ❌ 分块 $IDX 文件不存在"
    fi
done

# 统计结果
if [ -f "$output_dir/answers/$SPLIT/$CKPT.jsonl" ]; then
    result_count=$(wc -l < "$output_dir/answers/$SPLIT/$CKPT.jsonl")
    echo "📊 评估完成: $result_count 个结果"
    echo "📁 结果文件: $output_dir/answers/$SPLIT/$CKPT.jsonl"
else
    echo "❌ 结果文件生成失败"
    exit 1
fi

echo "🔄 转换结果格式用于提交..."
mkdir -p $output_dir/answers_upload/$SPLIT

python $SCRIPT_DIR/../training/scripts/convert_mmbench_for_submission.py \
    --annotation-file $SCRIPT_DIR/../training/data/eval_stage_1/mmbench/$SPLIT.tsv \
    --result-dir $output_dir/answers/$SPLIT \
    --upload-dir $output_dir/answers_upload/$SPLIT \
    --experiment $CKPT

if [ -d "$output_dir/answers_upload/$SPLIT" ] && [ "$(ls -A $output_dir/answers_upload/$SPLIT)" ]; then
    echo "📤 提交格式文件生成完成: $output_dir/answers_upload/$SPLIT"
else
    echo "⚠️  提交格式文件生成可能有问题"
fi

echo "✅ LoRA MMBench评估完成！"
