#!/bin/bash
# 安装脚本 - 用于设置DJ合成挑战项目的环境和依赖

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 安装基础依赖包
# pydantic: 数据验证库，限制版本<2.0.0以避免兼容性问题
# setuptools: Python包构建工具
pip install "pydantic<2.0.0" setuptools==69.5.1 setuptools_scm==8.0.4

# 第一步：安装data-juicer数据处理工具包
echo "[1] Installing toolkit/data-juicer"
cd ${SCRIPT_DIR}/toolkit
# 从GitHub克隆data-juicer项目
git clone https://github.com/modelscope/data-juicer.git
cd data-juicer
# 安装data-juicer及其所有可选依赖
pip install ".[all]"

# 第二步：安装MGM训练相关组件
echo "[2] Installing toolkit/training"
cd ${SCRIPT_DIR}/toolkit/training
# 以开发模式安装本地训练包
pip install -e .
# 安装flash-attn用于加速注意力计算，跳过构建隔离以避免编译问题
pip install flash-attn --no-build-isolation

echo "Done"

# 这些依赖是为了支持这个多模态AI模型训练竞赛项目：

#   data-juicer: 数据处理和合成工具包
#   - 将种子数据集转换为训练格式
#   - 提供图像标注功能
#   - 处理多模态数据预处理流水线

#   MGM训练组件: Mini-Gemini视觉语言模型框架
#   - 竞赛指定的核心模型架构
#   - 支持双视觉编码器（低分辨率+高分辨率）
#   - 结合文本和图像理解生成能力

#   flash-attn: 注意力计算加速库
#   - 优化大型transformer模型的性能
#   - 减少GPU内存使用
#   - 显著提升训练速度，专为A100/H100等高端GPU优化

#   这是一个多模态AI合成数据竞赛，需要这些专业工具来处理数据合成、模型训练和性能优化。