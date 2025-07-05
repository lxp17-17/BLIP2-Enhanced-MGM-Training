# CUDA环境配置与DeepSpeed优化指南

## 🔧 CUDA环境配置

### 问题背景
在多模态模型训练中，CUDA版本不匹配会导致DeepSpeed编译失败，进而影响训练稳定性。

### 解决方案

#### 1. 检查当前环境
```bash
# 检查PyTorch CUDA版本
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# 检查系统CUDA版本（如果已安装）
nvcc --version

# 检查GPU状态
nvidia-smi
```

#### 2. 安装匹配的CUDA Toolkit
```bash
# 确保在正确的conda环境中
conda info --envs

# 安装CUDA 12.4（匹配PyTorch）
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y

# 验证安装
nvcc --version
```

#### 3. 环境变量设置
```bash
export CUDA_VISIBLE_DEVICES=0
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export PYTHONPATH="$SCRIPT_DIR/training:$PYTHONPATH"
```

## ⚡ DeepSpeed配置优化

### ZeRO-2配置文件
`toolkit/training/scripts/zero2_offload.json`:
```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto", 
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu", 
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto"
    }
}
```

### 关键优化策略

#### 1. 内存优化
- **CPU Offload**: 将优化器和参数offload到CPU
- **Gradient Checkpointing**: 启用梯度检查点
- **Mixed Precision**: 使用bf16减少显存占用

#### 2. 批次大小调优
```bash
# 保守配置（推荐）
BATCH_SIZE_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=128
DATALOADER_NUM_WORKERS=1

# 激进配置（可能OOM）
BATCH_SIZE_PER_GPU=4
GRADIENT_ACCUMULATION_STEPS=32
DATALOADER_NUM_WORKERS=4
```

#### 3. 模型配置优化
```bash
--model_max_length 1024          # 减少序列长度
--gradient_checkpointing True     # 启用梯度检查点
--bf16 True                      # 使用混合精度
--lazy_preprocess True           # 延迟数据预处理
```

## 🚀 训练启动命令

### 标准启动
```bash
PYTHONPATH=$SCRIPT_DIR/training:$PYTHONPATH \
DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 \
python $(which deepspeed) $SCRIPT_DIR/training/mgm/train/train_mem.py \
    --deepspeed $SCRIPT_DIR/training/scripts/zero2_offload.json \
    --model_name_or_path $MODEL_PATH \
    --version gemma \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --pretrain_mm_mlp_adapter $PRETRAIN_MODEL \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --gradient_checkpointing True \
    --bf16 True \
    --model_max_length 1024 \
    --dataloader_num_workers 1 \
    --lazy_preprocess True
```

### 调试启动（最小配置）
```bash
# 用于快速验证的最小配置
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
--max_steps 10
--save_steps 5
--dataloader_num_workers 1
--model_max_length 1024
```

## 📊 资源监控

### 实时监控命令
```bash
# GPU使用情况
watch -n 1 nvidia-smi

# 内存使用情况  
watch -n 1 free -h

# 进程监控
htop
```

### 关键指标
- **GPU显存使用率**: 建议保持在80%以下
- **系统内存使用率**: 建议保持在70%以下
- **进程状态**: 避免出现返回码-9（被杀死）

## ⚠️ 常见问题与解决

### 1. CUDA版本不匹配
```
[WARNING] DeepSpeed Op Builder: Installed CUDA version 11.5 does not match torch 12.4
```
**解决**: 安装匹配的CUDA toolkit

### 2. 进程被杀死（返回码-9）
```
exits with return code = -9
```
**解决**: 减少batch size和worker数量

### 3. DeepSpeed编译失败
```
Building extension module cpu_adam... FAILED
```
**解决**: 设置环境变量 `DS_BUILD_OPS=0`

### 4. 显存不足
```
CUDA out of memory
```
**解决**: 启用CPU offload和gradient checkpointing

## 🎯 最佳实践

### 1. 环境准备
1. 确保conda环境激活
2. 安装匹配的CUDA toolkit
3. 验证PyTorch和CUDA兼容性

### 2. 配置策略
1. 从保守参数开始
2. 逐步调优性能
3. 监控资源使用情况

### 3. 调试流程
1. 先运行最小测试
2. 确认基础功能正常
3. 再进行完整训练

---

**更新日期**: 2025-07-01  
**适用环境**: CUDA 12.4 + PyTorch 2.5.1 + DeepSpeed  
**测试状态**: 已验证
