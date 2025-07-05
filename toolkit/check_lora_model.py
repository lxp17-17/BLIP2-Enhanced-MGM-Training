#!/usr/bin/env python3
"""
简化的LoRA模型检查脚本
检查LoRA模型的基本信息和兼容性
"""

import os
import json
import torch
import sys

def check_lora_model(model_path):
    """检查LoRA模型的基本信息"""
    print(f"🔍 检查LoRA模型: {model_path}")
    print("=" * 60)
    
    # 检查必要文件
    required_files = [
        "adapter_config.json",
        "adapter_model.bin", 
        "config.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file}: {size:.1f} MB")
        else:
            missing_files.append(file)
            print(f"❌ {file}: 缺失")
    
    if missing_files:
        print(f"\n⚠️  缺失文件: {missing_files}")
        return False
    
    print("\n📋 LoRA配置信息:")
    # 读取LoRA配置
    try:
        with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
            lora_config = json.load(f)
        
        print(f"  基础模型: {lora_config.get('base_model_name_or_path', 'unknown')}")
        print(f"  LoRA rank: {lora_config.get('r', 'unknown')}")
        print(f"  LoRA alpha: {lora_config.get('lora_alpha', 'unknown')}")
        print(f"  LoRA dropout: {lora_config.get('lora_dropout', 'unknown')}")
        print(f"  目标模块: {lora_config.get('target_modules', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 读取LoRA配置失败: {e}")
        return False
    
    print("\n🔧 模型配置信息:")
    # 读取模型配置
    try:
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            model_config = json.load(f)
        
        print(f"  模型类型: {model_config.get('model_type', 'unknown')}")
        print(f"  架构: {model_config.get('architectures', 'unknown')}")
        print(f"  词汇表大小: {model_config.get('vocab_size', 'unknown')}")
        print(f"  隐藏层大小: {model_config.get('hidden_size', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 读取模型配置失败: {e}")
        return False
    
    print("\n📊 文件大小统计:")
    total_size = 0
    for file in os.listdir(model_path):
        if file.endswith(('.bin', '.json', '.txt')):
            file_path = os.path.join(model_path, file)
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            total_size += size
            print(f"  {file}: {size:.1f} MB")
    
    print(f"\n📦 总大小: {total_size:.1f} MB")
    
    # 检查是否可以加载LoRA权重
    print("\n🧪 权重加载测试:")
    try:
        adapter_path = os.path.join(model_path, "adapter_model.bin")
        weights = torch.load(adapter_path, map_location='cpu')
        print(f"✅ LoRA权重加载成功，包含 {len(weights)} 个参数")
        
        # 显示前几个权重的信息
        for i, (key, tensor) in enumerate(list(weights.items())[:5]):
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
        
        if len(weights) > 5:
            print(f"  ... 还有 {len(weights)-5} 个权重")
            
    except Exception as e:
        print(f"❌ LoRA权重加载失败: {e}")
        return False
    
    print("\n✅ LoRA模型检查完成！")
    print("🎯 模型文件完整，可以用于推理或进一步评估")
    return True

def main():
    if len(sys.argv) != 2:
        print("用法: python check_lora_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        sys.exit(1)
    
    success = check_lora_model(model_path)
    
    if not success:
        print("\n❌ 模型检查失败！")
        sys.exit(1)
    else:
        print("\n🎉 模型检查成功！")

if __name__ == "__main__":
    main()
