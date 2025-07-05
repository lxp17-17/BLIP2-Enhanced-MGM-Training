# LoRA实施执行计划

## 📅 计划时间
**开始时间**: 2025-07-01 22:35  
**预计完成**: 2025-07-03 (2-3天)  
**当前阶段**: Phase 2 - 显存优化

## 🎯 执行目标

### 主要目标
- **解决显存不足问题**: 从25GB需求降至16GB以下
- **完成10K基线训练**: 验证LoRA方案可行性
- **掌握LoRA技术**: 为完整训练做准备

### 技术指标
- 显存使用 < 24GB ✅
- 可训练参数减少 > 80%
- 训练过程稳定无OOM
- 模型性能保持合理水平

## 📋 详细执行步骤

### Step 1: LoRA配置设计 (30分钟)

#### 1.1 分析当前训练参数
```bash
# 检查当前训练脚本配置
cd /home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit
cat train_mgm_2b_stage_1_10k_baseline.sh | grep -A 10 -B 10 "FINETUNE"
```

#### 1.2 设计LoRA配置
```python
# 保守配置 (推荐开始)
lora_config = {
    "lora_enable": True,
    "lora_r": 16,           # rank - 控制LoRA矩阵大小
    "lora_alpha": 32,       # scaling factor
    "lora_dropout": 0.1,    # dropout率
    "lora_bias": "none",    # bias处理方式
    "target_modules": ["q_proj", "v_proj", "o_proj"]  # 目标模块
}

# 预期效果: 参数量从3B降至~20M (减少99%)
```

### Step 2: 修改训练脚本 (45分钟)

#### 2.1 备份原始脚本
```bash
cp train_mgm_2b_stage_1_10k_baseline.sh train_mgm_2b_stage_1_10k_baseline_backup.sh
```

#### 2.2 添加LoRA参数
在微调阶段添加LoRA配置参数:
```bash
--lora_enable True \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--lora_bias none \
```

#### 2.3 验证脚本语法
```bash
bash -n train_mgm_2b_stage_1_10k_baseline.sh
```

### Step 3: 验证LoRA集成 (30分钟)

#### 3.1 检查训练代码LoRA支持
```bash
# 确认train.py中的LoRA相关代码
grep -n "lora_enable\|LoraConfig\|get_peft_model" dj_synth_challenge/toolkit/training/mgm/train/train.py
```

#### 3.2 验证PEFT库
```bash
cd dj_synth_challenge/toolkit
python -c "from peft import LoraConfig, get_peft_model; print('PEFT库正常')"
```

### Step 4: 小规模测试 (60分钟)

#### 4.1 创建测试脚本
创建最小化测试版本，限制训练步数:
```bash
# 复制并修改为测试版本
cp train_mgm_2b_stage_1_10k_baseline.sh test_lora_minimal.sh

# 修改测试参数:
# --max_steps 5 (只训练5步)
# --save_steps 2 (2步保存一次)
# --logging_steps 1 (每步记录)
```

#### 4.2 执行测试
```bash
cd dj_synth_challenge/toolkit
bash test_lora_minimal.sh 2>&1 | tee lora_test.log
```

#### 4.3 监控显存使用
```bash
# 另开终端监控
python monitor_memory.py --interval 5
```

### Step 5: 参数调优 (90分钟)

#### 5.1 如果测试成功
- 逐步增加训练步数 (5 → 20 → 50)
- 验证训练稳定性
- 检查loss收敛情况

#### 5.2 如果仍有显存问题
尝试更激进的配置:
```python
# 更小的rank
lora_r = 8

# 更少的目标模块
target_modules = ["q_proj", "v_proj"]

# 启用gradient checkpointing
gradient_checkpointing = True
```

#### 5.3 如果效果不佳
尝试更大的配置:
```python
# 更大的rank
lora_r = 32

# 更多的目标模块
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Step 6: 完整训练验证 (120分钟)

#### 6.1 执行完整10K基线训练
```bash
cd dj_synth_challenge/toolkit
bash train_mgm_2b_stage_1_10k_baseline.sh 2>&1 | tee lora_full_training.log
```

#### 6.2 训练过程监控
- 实时监控显存使用
- 检查loss收敛曲线
- 验证模型保存正常

#### 6.3 训练结果验证
- 检查模型文件生成
- 运行简单推理测试
- 对比训练前后效果

## 🔧 具体实施代码

### 修改训练脚本的具体位置
```bash
# 在train_mgm_2b_stage_1_10k_baseline.sh中找到微调部分
# 大约在第130-150行之间，添加LoRA参数

# 原始参数后添加:
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_bias none \
```

### 创建测试脚本模板
```bash
#!/bin/bash
# test_lora_minimal.sh - LoRA最小化测试

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/home/robot/lhp/miniconda3/envs/Syn0625
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

# 测试参数
MAX_STEPS=5
SAVE_STEPS=2
LOGGING_STEPS=1

# 执行微调测试 (只包含微调部分)
echo "开始LoRA最小化测试..."
# ... (微调命令 + LoRA参数)
```

## 📊 预期结果

### 显存使用对比
```
优化前:
├── 模型权重: ~6GB
├── 激活值: ~8GB
├── 梯度: ~6GB (全参数)
├── 优化器状态: ~6GB (全参数)
└── 总计: ~26GB > 24GB ❌

优化后 (LoRA):
├── 冻结模型权重: ~6GB
├── 激活值: ~8GB (不变)
├── LoRA梯度: ~0.1GB (仅LoRA参数)
├── 优化器状态: ~0.2GB (仅LoRA参数)
└── 总计: ~14.3GB < 24GB ✅
```

### 参数量对比
```
原始方案:
├── 总参数: 3.03B
├── 可训练参数: 3.03B (100%)
└── 显存需求: ~26GB

LoRA方案:
├── 总参数: 3.03B
├── 可训练参数: ~20M (0.7%)
├── 参数减少: 99.3%
└── 显存需求: ~14GB
```

## ⚠️ 风险控制

### 可能遇到的问题
1. **LoRA集成失败**: 检查PEFT库版本兼容性
2. **显存仍不足**: 进一步减少rank或目标模块
3. **训练不收敛**: 调整learning rate或LoRA配置
4. **性能下降**: 增加rank或扩展目标模块

### 备选方案
1. **Plan B**: 降级到1.5B模型
2. **Plan C**: 使用更激进的量化 (int8/int4)
3. **Plan D**: 模型并行 (如果有多GPU)

## 📝 执行记录

### 实时记录模板
```
时间: [timestamp]
步骤: [step_name]
状态: [成功/失败/进行中]
显存使用: [XX GB / 24GB]
问题: [如有]
解决方案: [如有]
下一步: [next_action]
```

## 🎯 成功标准

### 技术成功
- [ ] 显存使用稳定在24GB以下
- [ ] 训练过程无OOM错误
- [ ] LoRA参数正确加载和保存
- [ ] 模型推理功能正常

### 项目成功
- [ ] 10K基线训练完整完成
- [ ] 训练日志和结果完整记录
- [ ] LoRA技术完全掌握
- [ ] 为完整训练做好准备

---

**立即开始执行**: 从Step 1开始，逐步实施LoRA方案！🚀
