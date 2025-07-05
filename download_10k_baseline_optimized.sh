#!/bin/bash
# 优化版10k基线数据下载脚本 - 跳过已存在文件，避免重复下载
# 包含：10k预训练数据 + 12k微调数据 + 基础模型 + 评测数据

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 下载函数：检查文件是否存在，不存在才下载
download_if_not_exists() {
    local url=$1
    local filename=$(basename "$url")
    
    if [ -f "$filename" ]; then
        echo "✓ $filename 已存在，跳过下载"
    else
        echo "⬇ 正在下载 $filename..."
        wget "$url"
    fi
}

# 解压函数：检查目录是否存在，不存在才解压
extract_if_not_exists() {
    local archive=$1
    local target_dir=$2
    
    if [ -d "$target_dir" ]; then
        echo "✓ $target_dir 已存在，跳过解压"
        rm -f "$archive"  # 删除压缩包
    else
        echo "📦 正在解压 $archive..."
        tar zxvf "$archive" && rm -f "$archive"
    fi
}

# [步骤1] 下载微调数据集 (12k样本)
echo "[1] 检查微调数据集 (12k样本)..."
mkdir -p ${SCRIPT_DIR}/toolkit/training/data
cd ${SCRIPT_DIR}/toolkit/training/data

download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/finetuning_stage_1_12k.tar.gz"
extract_if_not_exists "finetuning_stage_1_12k.tar.gz" "finetuning_stage_1_12k"

cd finetuning_stage_1_12k 2>/dev/null || mkdir -p finetuning_stage_1_12k && cd finetuning_stage_1_12k
download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/mgm_instruction_stage_1_12k.json"

# [步骤2] 下载种子数据集 (10k样本)
echo "[2] 检查种子数据集 (10k样本)..."
mkdir -p ${SCRIPT_DIR}/input
cd ${SCRIPT_DIR}/input

download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/pretrain_stage_1_10k.tar.gz"
extract_if_not_exists "pretrain_stage_1_10k.tar.gz" "pretrain_stage_1_10k"

cd pretrain_stage_1_10k 2>/dev/null || mkdir -p pretrain_stage_1_10k && cd pretrain_stage_1_10k
download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/mgm_pretrain_stage_1_10k.jsonl"
download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/stage_1.json"

# [步骤3] 检查基础模型
echo "[3] 检查基础模型..."

# 检查Gemma-2B模型
mkdir -p ${SCRIPT_DIR}/toolkit/training/model_zoo/LLM/gemma
cd ${SCRIPT_DIR}/toolkit/training/model_zoo/LLM/gemma
if [ -d "gemma-2b-it" ]; then
    echo "✓ Gemma-2B模型已存在，跳过下载"
else
    download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/gemma-2b-it.tar.gz"
    extract_if_not_exists "gemma-2b-it.tar.gz" "gemma-2b-it"
fi

# 检查视觉编码器模型
mkdir -p ${SCRIPT_DIR}/toolkit/training/model_zoo/OpenAI
cd ${SCRIPT_DIR}/toolkit/training/model_zoo/OpenAI

# CLIP视觉编码器
if [ -d "clip-vit-large-patch14-336" ]; then
    echo "✓ CLIP视觉编码器已存在，跳过下载"
else
    download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/clip-vit-large-patch14-336.tar.gz"
    extract_if_not_exists "clip-vit-large-patch14-336.tar.gz" "clip-vit-large-patch14-336"
fi

# OpenCLIP ConvNeXt视觉编码器
if [ -d "openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup" ]; then
    echo "✓ OpenCLIP ConvNeXt视觉编码器已存在，跳过下载"
else
    download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz"
    extract_if_not_exists "openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz" "openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup"
fi

# [步骤4] 检查评测数据集
echo "[4] 检查评测数据集..."
mkdir -p ${SCRIPT_DIR}/toolkit/training/data
cd ${SCRIPT_DIR}/toolkit/training/data

if [ -d "eval_stage_1" ]; then
    echo "✓ 评测数据集已存在，跳过下载"
else
    download_if_not_exists "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/eval_stage_1.tar.gz"
    extract_if_not_exists "eval_stage_1.tar.gz" "eval_stage_1"
fi

echo "✅ 检查完成！所有必需文件已就绪。"
echo "📊 下载摘要："
echo "   - 如果文件已存在会显示 ✓ 跳过"
echo "   - 如果文件缺失会显示 ⬇ 下载"
echo "   - 大型模型文件只有在目录不存在时才会下载"