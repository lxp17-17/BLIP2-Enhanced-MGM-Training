#!/bin/bash

# 🎯 BLIP2增强数据MGM-2B LoRA训练脚本
# 📊 使用17,509条BLIP2增强高质量数据进行训练
# 🔧 基于之前的LoRA成功经验，优化内存使用
# 📅 最后成功运行: 2025-07-05 (训练时长: 1小时11分钟)

# 🐍 确保使用正确的Python环境 - 将training目录添加到Python路径
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/training:$PYTHONPATH"
# 🖥️  指定使用GPU 0进行训练
export CUDA_VISIBLE_DEVICES=0
# 🚫 禁用DeepSpeed CUDA扩展编译以避免版本不匹配问题
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
# 🏠 设置CUDA_HOME路径，指向conda环境以避免CUDA相关错误
export CUDA_HOME=/home/robot/lhp/miniconda3/envs/Syn0625
# 🧠 设置PyTorch内存管理策略，使用可扩展段减少内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

############################################################################
########################### 🔧 可编辑配置区域开始 ###########################
############################################################################
# 📝 实验元信息 - 定义实验名称，用于区分不同的训练实验
EXP_NAME=blip2-enhanced-lora

# 🚂 训练参数配置 - 基于BLIP2增强数据优化，针对24GB VRAM限制调整
# 📊 预训练阶段 - 使用保守参数确保训练稳定性
# ⚖️  全局批次大小公式: BATCH_SIZE_PER_GPU × GRADIENT_ACCUMULATION_STEPS × GPU数量 = 256
PRETRAIN_BATCH_SIZE_PER_GPU=1        # 🔻 每GPU批次大小=1，避免显存溢出(OOM)
PRETRAIN_GRADIENT_ACCUMULATION_STEPS=256  # 🔺 梯度累积步数=256，保持全局批次大小256
PRETRAIN_DATALOADER_NUM_WORKERS=1    # 🔻 数据加载器工作进程=1，减少内存占用

# 🎯 微调阶段 - 保持教师要求的梯度累积步数(不可修改)
# ⚖️  全局批次大小公式: BATCH_SIZE_PER_GPU × GRADIENT_ACCUMULATION_STEPS × GPU数量 = 128
FINETUNE_BATCH_SIZE_PER_GPU=1        # 🔻 每GPU批次大小=1，最小批次避免OOM
FINETUNE_GRADIENT_ACCUMULATION_STEPS=128  # 📌 梯度累积步数=128，教师要求不能修改
FINETUNE_DATALOADER_NUM_WORKERS=1    # 🔻 数据加载器工作进程=1，最小worker数

# 📋 日志和检查点配置
LOGGING_STEP=10                      # 📊 每10步记录一次日志，减少I/O频率
CKPT_SAVE_STEPS=500                  # 💾 每500步保存一次检查点，减少存储频率
TOTAL_SAVE_CKPT_LIMIT=2              # 🗂️  最多保留2个检查点，节省磁盘空间

# 🔧 LoRA配置参数 - 基于之前成功经验的最优配置
LORA_R=16                            # 🎛️  LoRA秩=16，控制参数量和性能平衡
LORA_ALPHA=32                        # 🎚️  LoRA缩放因子=32，控制LoRA权重影响
LORA_DROPOUT=0.1                     # 🎲 LoRA dropout=0.1，防止过拟合

# 🔍 推理参数配置
INFER_CUDA_IDX="0"                   # 🖥️  推理时使用GPU 0
############################################################################
############################ 🔧 可编辑配置区域结束 ############################
############################################################################
# 📁 获取脚本所在目录的绝对路径，用于后续相对路径计算
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 📂 设置数据路径 - 指向你生成的BLIP2增强数据
PRETRAIN_DATASET=$SCRIPT_DIR/../output/processed_data/blip2_enhanced_30k_data.jsonl  # 🎯 你的BLIP2增强数据文件
PRETRAIN_DATASET_IMAGE_PATH=$SCRIPT_DIR/../input/pretrain_stage_1                    # 🖼️  图像文件夹路径

# 📋 使用原始的stage_1.json作为数据格式转换的参考模板
ORIGINAL_DATASET_ALL=$SCRIPT_DIR/../input/pretrain_stage_1/stage_1.json

# 🎉 开始训练 - 打印训练配置和数据统计信息
echo "🚀 开始BLIP2增强数据MGM-2B LoRA训练"
echo "📊 数据统计:"
echo "  - 训练数据: 17,509条BLIP2增强高质量数据"          # 📈 实际使用的数据量
echo "  - 平均词数: 10.67词 (vs 原始8.78词)"            # 📝 描述长度提升21.5%
echo "  - 词汇多样性: 0.37 (vs 原始0.0714)"            # 🌈 词汇丰富度提升418%
echo "  - 数据来源: $PRETRAIN_DATASET"                  # 📁 数据文件路径
echo ""

# ✅ 验证全局批次大小 - 确保训练配置符合要求
# 🧮 预训练阶段: 检查 BATCH_SIZE_PER_GPU × GRADIENT_ACCUMULATION_STEPS × GPU数量 = 256
PRETRAIN_PASS=`python $SCRIPT_DIR/training/preprocess/check_global_batch_size.py $PRETRAIN_BATCH_SIZE_PER_GPU $PRETRAIN_GRADIENT_ACCUMULATION_STEPS 256`
if [ "$PRETRAIN_PASS" = "False" ]; then
    echo "[❌ ERROR] 预训练阶段全局批次大小不等于256！请检查配置后重试。"
    exit
fi
# 🧮 微调阶段: 检查 BATCH_SIZE_PER_GPU × GRADIENT_ACCUMULATION_STEPS × GPU数量 = 128
FINETUNE_PASS=`python $SCRIPT_DIR/training/preprocess/check_global_batch_size.py $FINETUNE_BATCH_SIZE_PER_GPU $FINETUNE_GRADIENT_ACCUMULATION_STEPS 128`
if [ "$FINETUNE_PASS" = "False" ]; then
    echo "[❌ ERROR] 微调阶段全局批次大小不等于128！请检查配置后重试。"
    exit
fi

echo "✅ 批次大小验证通过"
echo "  - 预训练全局批次大小: 256 (1×256×1)"           # 📊 显示计算过程
echo "  - 微调全局批次大小: 128 (1×128×1)"             # 📊 显示计算过程
echo ""

# 📊 检查BLIP2增强数据的实际样本数量
BLIP2_SAMPLE_NUM=`wc -l < $PRETRAIN_DATASET`
echo "📋 BLIP2增强数据样本数: $BLIP2_SAMPLE_NUM"

# ⏱️  限制最大样本数以控制训练时间（避免训练过长）
MAX_SAMPLE_NUM=20000
if [ $BLIP2_SAMPLE_NUM -gt $MAX_SAMPLE_NUM ]; then
    # 🎲 如果样本数超过限制，进行随机采样
    SAMPLED_PRETRAIN_DATASET=$PRETRAIN_DATASET-${MAX_SAMPLE_NUM}.jsonl
    echo "⚠️  样本数超过${MAX_SAMPLE_NUM}，进行采样..."
    python $SCRIPT_DIR/training/preprocess/check_sample_number.py $PRETRAIN_DATASET $SAMPLED_PRETRAIN_DATASET $MAX_SAMPLE_NUM
else
    # ✅ 样本数在合理范围内，使用全部数据
    SAMPLED_PRETRAIN_DATASET=$PRETRAIN_DATASET
    echo "✅ 样本数在合理范围内，使用全部数据"
fi

# 🔄 数据格式转换: 将Data-Juicer格式转换为LLaVA格式
PRETRAIN_DATASET_JSON=$SAMPLED_PRETRAIN_DATASET.json
echo "🔄 转换数据格式: DJ → LLaVA"
# 📝 调用转换工具，将BLIP2增强数据转换为MGM训练所需的LLaVA格式
python $SCRIPT_DIR/data-juicer/tools/multimodal/data_juicer_format_to_target_format/dj_to_llava.py \
    $SAMPLED_PRETRAIN_DATASET \                    # 🔤 输入: DJ格式的BLIP2增强数据
    $PRETRAIN_DATASET_JSON \                       # 📄 输出: LLaVA格式的JSON文件
    --image_special_token "<__dj__image>" \        # 🖼️  图像占位符标记
    --restore_questions True \                     # ❓ 恢复问题格式
    --original_llava_ds_path $ORIGINAL_DATASET_ALL # 📋 参考原始数据集格式

# 🏷️  定义训练模型名称 - 用于区分预训练和微调阶段的输出
PRETRAIN_NAME=MGM-2B-BLIP2-Pretrain-$EXP_NAME     # 📂 预训练模型名称
FINETUNE_NAME=MGM-2B-BLIP2-Finetune-$EXP_NAME     # 📂 微调模型名称
AUX_SIZE=768                                       # 🖼️  辅助视觉编码器的图像尺寸

NUM_TRAIN_EPOCHS=1                                 # 🔄 训练轮数=1，避免过拟合
ACTUAL_SAMPLE_NUM=`wc -l < $SAMPLED_PRETRAIN_DATASET` # 📊 统计实际使用的样本数

echo ""
echo "🎯 训练配置:"
echo "  - 实验名称: $EXP_NAME"                      # 📝 实验标识
echo "  - 预训练样本数: $ACTUAL_SAMPLE_NUM"         # 📊 实际训练数据量
echo "  - 训练轮数: $NUM_TRAIN_EPOCHS"              # 🔄 epoch数量
echo "  - LoRA参数: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT" # 🔧 LoRA配置
echo "  - 辅助图像尺寸: $AUX_SIZE"                  # 🖼️  图像处理尺寸
echo ""

# 📁 创建预训练模型输出目录
mkdir -p $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME

echo "🚀 开始预训练阶段 (LoRA)..."
echo "⏰ 开始时间: $(date)"
# 🎯 预训练命令 - 使用DeepSpeed进行分布式训练，启用LoRA参数高效微调
PYTHONPATH=$SCRIPT_DIR/training:$PYTHONPATH DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 \
/home/robot/lhp/miniconda3/envs/Syn0625/bin/python $(which deepspeed) \
$SCRIPT_DIR/training/mgm/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/training/scripts/zero3.json \                                    # 🚀 DeepSpeed ZeRO-3配置文件
    --model_name_or_path $SCRIPT_DIR/training/model_zoo/LLM/gemma/gemma-2b-it \             # 🤖 基础语言模型: Gemma-2B-IT
    --version gemma \                                                                        # 📝 模型版本标识
    --data_path $PRETRAIN_DATASET_JSON \                                                    # 📊 训练数据: 你的BLIP2增强数据
    --image_folder $PRETRAIN_DATASET_IMAGE_PATH \                                           # 🖼️  图像文件夹路径
    --vision_tower $SCRIPT_DIR/training/model_zoo/OpenAI/clip-vit-large-patch14-336 \      # 👁️  主视觉编码器: CLIP ViT-L/14@336px
    --vision_tower_aux $SCRIPT_DIR/training/model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \ # 👁️  辅助视觉编码器: ConvNeXt-L
    --mm_projector_type mlp2x_gelu \                                                        # 🔗 多模态投影器: 2层MLP+GELU激活
    --tune_mm_mlp_adapter True \                                                            # 🎛️  训练多模态适配器
    --mm_vision_select_layer -2 \                                                           # 🎯 选择视觉编码器倒数第2层特征
    --mm_use_im_start_end False \                                                           # 🚫 不使用图像开始/结束标记
    --mm_use_im_patch_token False \                                                         # 🚫 不使用图像patch标记
    --image_size_aux $AUX_SIZE \                                                            # 🖼️  辅助图像尺寸: 768px
    --bf16 True \                                                                           # 🔢 使用BFloat16精度训练
    --output_dir $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME \                      # 📁 输出目录
    --num_train_epochs $NUM_TRAIN_EPOCHS \                                                  # 🔄 训练轮数: 1
    --per_device_train_batch_size $PRETRAIN_BATCH_SIZE_PER_GPU \                           # 📦 每GPU批次大小: 1
    --per_device_eval_batch_size 1 \                                                       # 📦 评估批次大小: 1
    --gradient_accumulation_steps $PRETRAIN_GRADIENT_ACCUMULATION_STEPS \                  # 🔺 梯度累积步数: 256
    --evaluation_strategy "no" \                                                            # 🚫 不进行评估
    --save_strategy "steps" \                                                               # 💾 按步数保存检查点
    --save_steps $CKPT_SAVE_STEPS \                                                         # 💾 每500步保存一次
    --save_total_limit $TOTAL_SAVE_CKPT_LIMIT \                                             # 💾 最多保留2个检查点
    --learning_rate 1e-3 \                                                                  # 📈 学习率: 0.001 (预训练用较大学习率)
    --weight_decay 0. \                                                                     # ⚖️  权重衰减: 0 (不使用L2正则化)
    --warmup_ratio 0.03 \                                                                   # 🔥 预热比例: 3% (逐渐增加学习率)
    --lr_scheduler_type "cosine" \                                                          # 📉 学习率调度: 余弦退火
    --logging_steps $LOGGING_STEP \                                                         # 📊 每10步记录日志
    --tf32 True \                                                                           # 🔢 启用TensorFloat-32加速
    --model_max_length 2048 \                                                               # 📏 最大序列长度: 2048 tokens
    --gradient_checkpointing True \                                                         # 💾 梯度检查点节省显存
    --dataloader_num_workers $PRETRAIN_DATALOADER_NUM_WORKERS \                            # 👷 数据加载器工作进程: 1
    --lazy_preprocess True \                                                                # 🐌 延迟预处理节省内存
    --report_to none \                                                                      # 🚫 不上报到wandb等平台
    --lora_enable True \                                                                    # ✅ 启用LoRA参数高效微调
    --lora_r $LORA_R \                                                                      # 🎛️  LoRA秩: 16
    --lora_alpha $LORA_ALPHA \                                                              # 🎚️  LoRA缩放: 32
    --lora_dropout $LORA_DROPOUT \                                                          # 🎲 LoRA dropout: 0.1
    --lora_bias none \                                                                      # 🚫 LoRA不训练bias
    2>&1 | tee $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME/pretrain.log             # 📋 保存训练日志

echo "✅ 预训练完成: $(date)"

# 📁 创建微调模型输出目录
mkdir -p $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME

echo ""
echo "🚀 开始微调阶段 (LoRA)..."
echo "⏰ 开始时间: $(date)"
# 🎯 微调命令 - 在预训练基础上进行指令微调，使用标准12k指令数据
PYTHONPATH=$SCRIPT_DIR/training:$PYTHONPATH DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 \
/home/robot/lhp/miniconda3/envs/Syn0625/bin/python $(which deepspeed) \
$SCRIPT_DIR/training/mgm/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/training/scripts/zero3.json \                                    # 🚀 DeepSpeed ZeRO-3配置文件
    --model_name_or_path $SCRIPT_DIR/training/model_zoo/LLM/gemma/gemma-2b-it \             # 🤖 基础语言模型: Gemma-2B-IT
    --version gemma \                                                                        # 📝 模型版本标识
    --data_path $SCRIPT_DIR/training/data/finetuning_stage_1_12k/mgm_instruction_stage_1_12k.json \ # 📚 微调数据: 12k指令数据
    --image_folder $SCRIPT_DIR/training/data/finetuning_stage_1_12k \                       # 🖼️  微调图像文件夹
    --vision_tower $SCRIPT_DIR/training/model_zoo/OpenAI/clip-vit-large-patch14-336 \      # 👁️  主视觉编码器: CLIP ViT-L/14@336px
    --vision_tower_aux $SCRIPT_DIR/training/model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup \ # 👁️  辅助视觉编码器: ConvNeXt-L
    --pretrain_mm_mlp_adapter $SCRIPT_DIR/../output/training_dirs/$PRETRAIN_NAME/mm_projector.bin \ # 🔗 加载预训练的多模态投影器
    --mm_projector_type mlp2x_gelu \                                                        # 🔗 多模态投影器: 2层MLP+GELU激活
    --mm_vision_select_layer -2 \                                                           # 🎯 选择视觉编码器倒数第2层特征
    --mm_use_im_start_end False \                                                           # 🚫 不使用图像开始/结束标记
    --mm_use_im_patch_token False \                                                         # 🚫 不使用图像patch标记
    --image_aspect_ratio pad \                                                              # 🖼️  图像宽高比处理: 填充
    --group_by_modality_length True \                                                       # 📊 按模态长度分组提高效率
    --image_size_aux $AUX_SIZE \                                                            # 🖼️  辅助图像尺寸: 768px
    --bf16 True \                                                                           # 🔢 使用BFloat16精度训练
    --output_dir $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME \                      # 📁 输出目录
    --num_train_epochs $NUM_TRAIN_EPOCHS \                                                  # 🔄 训练轮数: 1
    --per_device_train_batch_size $FINETUNE_BATCH_SIZE_PER_GPU \                           # 📦 每GPU批次大小: 1
    --per_device_eval_batch_size 1 \                                                       # 📦 评估批次大小: 1
    --gradient_accumulation_steps $FINETUNE_GRADIENT_ACCUMULATION_STEPS \                  # 🔺 梯度累积步数: 128 (教师要求)
    --evaluation_strategy "no" \                                                            # 🚫 不进行评估
    --save_strategy "steps" \                                                               # 💾 按步数保存检查点
    --save_steps $CKPT_SAVE_STEPS \                                                         # 💾 每500步保存一次
    --save_total_limit $TOTAL_SAVE_CKPT_LIMIT \                                             # 💾 最多保留2个检查点
    --learning_rate 2e-5 \                                                                  # 📈 学习率: 0.00002 (微调用较小学习率)
    --weight_decay 0. \                                                                     # ⚖️  权重衰减: 0 (不使用L2正则化)
    --warmup_ratio 0.03 \                                                                   # 🔥 预热比例: 3% (逐渐增加学习率)
    --lr_scheduler_type "cosine" \                                                          # 📉 学习率调度: 余弦退火
    --logging_steps $LOGGING_STEP \                                                         # 📊 每10步记录日志
    --tf32 True \                                                                           # 🔢 启用TensorFloat-32加速
    --model_max_length 1024 \                                                               # 📏 最大序列长度: 1024 tokens (微调用较短)
    --gradient_checkpointing True \                                                         # 💾 梯度检查点节省显存
    --dataloader_num_workers $FINETUNE_DATALOADER_NUM_WORKERS \                            # 👷 数据加载器工作进程: 1
    --lazy_preprocess True \                                                                # 🐌 延迟预处理节省内存
    --report_to none \                                                                      # 🚫 不上报到wandb等平台
    --lora_enable True \                                                                    # ✅ 启用LoRA参数高效微调
    --lora_r $LORA_R \                                                                      # 🎛️  LoRA秩: 16
    --lora_alpha $LORA_ALPHA \                                                              # 🎚️  LoRA缩放: 32
    --lora_dropout $LORA_DROPOUT \                                                          # 🎲 LoRA dropout: 0.1
    --lora_bias none \                                                                      # 🚫 LoRA不训练bias
    2>&1 | tee $SCRIPT_DIR/../output/training_dirs/$FINETUNE_NAME/finetuning.log           # 📋 保存微调日志

echo "✅ 微调完成: $(date)"

echo ""
echo "🔍 开始评估..."

# 📊 模型评估 - 在标准基准上测试训练好的模型性能
# 📝 TextVQA评估 - 文本视觉问答任务
echo "📊 TextVQA评估..."
bash $SCRIPT_DIR/eval/textvqa.sh $FINETUNE_NAME $INFER_CUDA_IDX

# 📊 MMBench评估 - 多模态理解基准测试
echo "📊 MMBench评估..."
bash $SCRIPT_DIR/eval/mmbench.sh $FINETUNE_NAME "mmbench_dev_20230712" $INFER_CUDA_IDX

# 📋 备份训练脚本到输出目录，便于后续复现
cp $0 $SCRIPT_DIR/../output/train_blip2_enhanced_lora.sh

echo ""
echo "🎉 训练和评估完成!"
echo "📁 训练检查点: output/training_dirs/$FINETUNE_NAME"        # 🎯 你的最终LoRA模型位置
echo "📊 评估结果: output/eval_results/$FINETUNE_NAME"            # 📈 评估结果文件位置
echo "📋 训练日志: output/training_dirs/$PRETRAIN_NAME/pretrain.log"  # 📝 预训练日志
echo "📋 微调日志: output/training_dirs/$FINETUNE_NAME/finetuning.log" # 📝 微调日志
echo ""
echo "🔬 数据质量提升总结:"
echo "  - 使用17,509条BLIP2增强数据"                            # 📊 数据量统计
echo "  - 词数提升: 8.78 → 10.67词 (+21.5%)"                  # 📝 描述长度改善
echo "  - 词汇多样性: 0.0714 → 0.37 (+418%)"                  # 🌈 词汇丰富度大幅提升
echo "  - LoRA参数高效微调，24GB VRAM友好"                      # 💾 内存效率优化
