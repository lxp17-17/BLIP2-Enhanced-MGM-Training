# DJ合成挑战项目概览

## 项目简介

这是一个**Better Synth**多模态AI训练竞赛项目，专注于使用合成数据训练视觉语言模型。项目采用**数据合成 → 模型训练 → 评估**的完整流程，基于Data-Juicer数据处理工具和MGM（Mini-Gemini）模型架构。

## 核心文件

- **install.sh** - 环境安装脚本，包含data-juicer、MGM训练组件、flash-attn等核心依赖
- **download.sh** - 数据和模型下载脚本，获取种子数据、基础模型、评估数据集
- **README.md** - 详细操作指南和比赛流程说明
- **10k_quick_baseline.ipynb** - 新手快速体验的基线实验
- **Dockerfile** - 容器化部署配置

## 目录结构

### 核心工具包
- **toolkit/data-juicer/** - 数据处理和合成工具包（完整克隆）
  - 提供多模态数据预处理管道
  - 支持数据去重、过滤、增强等操作
  - 包含BLIP2等图像标注工具
  
- **toolkit/training/** - MGM模型训练框架
  - MGM-2B多模态视觉语言模型
  - 支持双视觉编码器（低分辨率+高分辨率）
  - 集成flash-attn优化和DeepSpeed分布式训练
  
- **toolkit/eval/** - 模型评估脚本
  - TextVQA和MMBench基准测试
  - 自动化评估流程

### 数据目录
- **input/** - 输入的种子数据集
- **output/** - 处理后的数据和训练输出
  - `processed_data/` - 合成的训练数据
  - `training_dirs/` - 训练好的模型
  - `eval_results/` - 评估结果
- **solution/** - 参赛者的解决方案代码

## 训练脚本

- **train_mgm_2b_stage_1.sh** - 完整的第一阶段训练
  - 预训练 + 微调两阶段流程
  - 支持自定义参数配置
  
- **train_mgm_2b_stage_1_10k_baseline.sh** - 10k数据快速基线
  - 适合初学者快速体验
  - 较小数据集，训练时间短

## 技术架构

### 数据流程
1. **种子数据** → **Data-Juicer处理** → **合成训练数据**
2. **JSONL格式** + **图像文件**
3. **DJ格式** ↔ **LLaVA格式**转换

### 模型架构
- **基础语言模型**: Gemma-2B-IT
- **视觉编码器**: 
  - CLIP ViT-Large-Patch14-336（低分辨率）
  - OpenCLIP ConvNeXt-Large（高分辨率）
- **多模态融合**: MGM架构

### 训练优化
- **Flash Attention**: 加速注意力计算
- **DeepSpeed**: 分布式训练和内存优化
- **两阶段训练**: 预训练 + 指令微调

## 竞赛流程

1. **环境搭建**: `bash install.sh`
2. **数据下载**: `bash download.sh`
3. **数据合成**: 使用Data-Juicer处理种子数据
4. **模型训练**: 执行训练脚本
5. **模型评估**: 在TextVQA/MMBench上测试
6. **结果提交**: 打包solution和output目录

## 硬件要求

- **GPU**: A100或H100（推荐40GB+显存）
- **系统**: Linux（Ubuntu 20.04+）
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.8+

## 特色功能

- **完整的多模态训练流水线**
- **可扩展的数据处理管道**
- **优化的大模型训练框架**
- **标准化的评估体系**
- **容器化部署支持**

这是一个production-ready的多模态AI竞赛框架，涵盖了从数据处理到模型部署的完整链路。