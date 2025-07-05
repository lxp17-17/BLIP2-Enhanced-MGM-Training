#!/bin/bash
# 下载脚本 - 获取DJ合成挑战赛第一阶段所需的数据集和模型

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 第一步：下载微调数据集
echo "[1] Downloading finetuning datasets..."
# 创建训练数据目录
mkdir -p ${SCRIPT_DIR}/toolkit/training/data
cd ${SCRIPT_DIR}/toolkit/training/data
# 下载微调阶段数据包
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/finetuning_stage_1.tar.gz
# 解压并删除压缩包
tar zxvf finetuning_stage_1.tar.gz && rm -rf finetuning_stage_1.tar.gz
cd finetuning_stage_1
# 下载MGM指令数据集（JSON格式）
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/mgm_instruction_stage_1.json

# 第二步：下载种子数据集（用于预训练）
echo "[2] Downloading seed datasets..."
# 创建输入数据目录
mkdir -p ${SCRIPT_DIR}/input
cd ${SCRIPT_DIR}/input
# 下载预训练阶段种子数据包
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/pretrain_stage_1.tar.gz
# 解压并删除压缩包
tar zxvf pretrain_stage_1.tar.gz && rm -rf pretrain_stage_1.tar.gz
cd pretrain_stage_1
# 下载MGM预训练数据集（JSONL格式）
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/mgm_pretrain_stage_1.jsonl
# 下载第一阶段配置文件
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/stage_1.json

# 第三步：下载基础模型
echo "[3] Downloading base models for training..."
# 创建Gemma模型目录并下载
mkdir -p ${SCRIPT_DIR}/toolkit/training/model_zoo/LLM/gemma
cd ${SCRIPT_DIR}/toolkit/training/model_zoo/LLM/gemma
# 下载Gemma-2B-IT基础语言模型
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/gemma-2b-it.tar.gz
tar zxvf gemma-2b-it.tar.gz && rm -rf gemma-2b-it.tar.gz

# 创建OpenAI模型目录并下载视觉编码器
mkdir -p ${SCRIPT_DIR}/toolkit/training/model_zoo/OpenAI
cd ${SCRIPT_DIR}/toolkit/training/model_zoo/OpenAI
# 下载CLIP视觉编码器（ViT-Large-Patch14-336）
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/clip-vit-large-patch14-336.tar.gz
tar zxvf clip-vit-large-patch14-336.tar.gz && rm -rf clip-vit-large-patch14-336.tar.gz
# 下载OpenCLIP ConvNeXt视觉编码器（用于高分辨率图像处理）
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/models/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz
tar zxvf openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz && rm -rf openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup.tar.gz

# 第四步：下载评估数据集
echo "[4] Downloading evaluation datasets"
# 切换到训练数据目录
mkdir -p ${SCRIPT_DIR}/toolkit/training/data
cd ${SCRIPT_DIR}/toolkit/training/data
# 下载评估数据包（TextVQA、MMBench等基准测试集）
wget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/better_synth/data/stage_1/eval_stage_1.tar.gz
tar zxvf eval_stage_1.tar.gz && rm -rf eval_stage_1.tar.gz

echo "Done"