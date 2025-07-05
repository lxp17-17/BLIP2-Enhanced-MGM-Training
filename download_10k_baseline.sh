#!/bin/bash
# 10k基线数据下载脚本 - 快速体验比赛流程用的小数据集
# 包含：10k预训练数据 + 12k微调数据 + 基础模型 + 评测数据

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# [步骤1] 下载微调数据集 (12k样本，用于第二阶段指令微调)
echo "[1] 正在下载微调数据集 (12k样本)..."
mkdir -p ${SCRIPT_DIR}/toolkit/training/data  # 创建训练数据目录
cd ${SCRIPT_DIR}/toolkit/training/data
# 下载12k微调数据压缩包
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/finetuning_stage_1_12k.tar.gz
# 解压并删除压缩包
tar zxvf finetuning_stage_1_12k.tar.gz && rm -rf finetuning_stage_1_12k.tar.gz
cd finetuning_stage_1_12k
# 下载微调指令数据的JSONL文件
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/mgm_instruction_stage_1_12k.json

# [步骤2] 下载种子数据集 (10k样本，这是你要处理的原始数据)
echo "[2] 正在下载种子数据集 (10k样本)..."
mkdir -p ${SCRIPT_DIR}/input  # 创建输入数据目录
cd ${SCRIPT_DIR}/input
# 下载10k种子数据压缩包
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/pretrain_stage_1_10k.tar.gz
# 解压并删除压缩包
tar zxvf pretrain_stage_1_10k.tar.gz && rm -rf pretrain_stage_1_10k.tar.gz
cd pretrain_stage_1_10k
# 下载预训练数据的JSONL文件和配置文件
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/mgm_pretrain_stage_1_10k.jsonl
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/stage_1.json

# [步骤3] 下载基础模型 (语言模型和视觉编码器)
echo "[3] 正在下载基础模型..."
# 下载Gemma-2B语言模型
mkdir -p ${SCRIPT_DIR}/toolkit/training/model_zoo/LLM/gemma
cd ${SCRIPT_DIR}/toolkit/training/model_zoo/LLM/gemma
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/gemma-2b-it.tar.gz
tar zxvf gemma-2b-it.tar.gz && rm -rf gemma-2b-it.tar.gz

# 下载视觉编码器模型
mkdir -p ${SCRIPT_DIR}/toolkit/training/model_zoo/OpenAI
cd ${SCRIPT_DIR}/toolkit/training/model_zoo/OpenAI
# CLIP视觉编码器（低分辨率，336x336）
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/clip-vit-large-patch14-336.tar.gz
tar zxvf clip-vit-large-patch14-336.tar.gz && rm -rf clip-vit-large-patch14-336.tar.gz
# OpenCLIP ConvNeXt视觉编码器（高分辨率）
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz
tar zxvf openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz && rm -rf openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz

# [步骤4] 下载评测数据集 (TextVQA和MMBench)
echo "[4] 正在下载评测数据集..."
mkdir -p ${SCRIPT_DIR}/toolkit/training/data
cd ${SCRIPT_DIR}/toolkit/training/data
# 下载评测数据压缩包
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/eval_stage_1.tar.gz
tar zxvf eval_stage_1.tar.gz && rm -rf eval_stage_1.tar.gz

echo "Done"