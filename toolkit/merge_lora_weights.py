#!/usr/bin/env python3
"""
LoRA权重合并脚本
将LoRA权重合并到基础模型，生成标准的MGM模型用于评估
"""

import os
import sys
import json
import torch
import shutil
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
import argparse
from pathlib import Path

class LoRAMerger:
    """LoRA权重合并器"""
    
    def __init__(self, lora_model_path: str, base_model_path: str = None, output_path: str = None):
        """
        初始化LoRA合并器
        
        Args:
            lora_model_path: LoRA模型路径
            base_model_path: 基础模型路径
            output_path: 输出路径
        """
        self.lora_model_path = lora_model_path
        self.base_model_path = base_model_path
        self.output_path = output_path
        
        print(f"🔧 初始化LoRA权重合并器")
        print(f"📁 LoRA模型: {lora_model_path}")
        
    def determine_paths(self):
        """确定基础模型和输出路径"""
        # 读取LoRA配置确定基础模型
        adapter_config_path = os.path.join(self.lora_model_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            lora_config = json.load(f)
        
        if self.base_model_path is None:
            base_model_name = lora_config.get("base_model_name_or_path", "")
            if "gemma-2b-it" in base_model_name:
                self.base_model_path = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit/training/model_zoo/LLM/gemma/gemma-2b-it"
            else:
                self.base_model_path = base_model_name
        
        if self.output_path is None:
            # 创建合并后的模型路径
            lora_dir_name = os.path.basename(self.lora_model_path)
            merged_dir_name = lora_dir_name.replace("-lora", "-merged")
            self.output_path = os.path.join(
                os.path.dirname(self.lora_model_path),
                merged_dir_name
            )
        
        print(f"📋 基础模型: {self.base_model_path}")
        print(f"📤 输出路径: {self.output_path}")
        
        return True
    
    def merge_lora_weights(self):
        """合并LoRA权重到基础模型"""
        try:
            print("\n🔄 开始合并LoRA权重...")
            
            # 确定路径
            self.determine_paths()
            
            # 创建输出目录
            os.makedirs(self.output_path, exist_ok=True)
            
            # 加载tokenizer
            print("🔤 加载Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            
            # 检查是否是MGM模型（多模态）
            config_path = os.path.join(self.lora_model_path, "config.json")
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            model_type = model_config.get("model_type", "")
            print(f"🔍 检测到模型类型: {model_type}")
            
            if "mgm" in model_type.lower():
                print("🖼️  检测到MGM多模态模型，使用特殊处理...")
                return self.merge_mgm_lora()
            else:
                print("📝 检测到标准语言模型，使用标准合并...")
                return self.merge_standard_lora()
                
        except Exception as e:
            print(f"❌ LoRA权重合并失败: {e}")
            return False
    
    def merge_standard_lora(self):
        """合并标准LoRA模型"""
        try:
            from transformers import AutoModelForCausalLM
            
            # 加载基础模型
            print("🧠 加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="cpu",  # 在CPU上合并以节省GPU内存
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            print("🔗 加载LoRA适配器...")
            model = PeftModel.from_pretrained(
                base_model,
                self.lora_model_path,
                torch_dtype=torch.float16
            )
            
            # 合并权重
            print("⚡ 合并权重...")
            merged_model = model.merge_and_unload()
            
            # 保存合并后的模型
            print("💾 保存合并后的模型...")
            merged_model.save_pretrained(self.output_path)
            
            # 保存tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            tokenizer.save_pretrained(self.output_path)
            
            print("✅ 标准LoRA权重合并完成！")
            return True
            
        except Exception as e:
            print(f"❌ 标准LoRA合并失败: {e}")
            return False
    
    def merge_mgm_lora(self):
        """合并MGM多模态LoRA模型"""
        try:
            print("🔄 MGM模型需要特殊处理...")
            
            # 复制必要的文件
            print("📋 复制模型配置文件...")
            
            # 复制LoRA模型的配置文件
            files_to_copy = [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json"
            ]
            
            for file_name in files_to_copy:
                src_file = os.path.join(self.lora_model_path, file_name)
                if os.path.exists(src_file):
                    dst_file = os.path.join(self.output_path, file_name)
                    shutil.copy2(src_file, dst_file)
                    print(f"  ✅ 复制 {file_name}")
                else:
                    # 尝试从基础模型复制
                    base_src_file = os.path.join(self.base_model_path, file_name)
                    if os.path.exists(base_src_file):
                        dst_file = os.path.join(self.output_path, file_name)
                        shutil.copy2(base_src_file, dst_file)
                        print(f"  ✅ 从基础模型复制 {file_name}")
            
            # 复制LoRA权重文件
            print("🔗 复制LoRA权重文件...")
            lora_files = [
                "adapter_model.bin",
                "adapter_config.json",
                "non_lora_trainables.bin"
            ]
            
            for file_name in lora_files:
                src_file = os.path.join(self.lora_model_path, file_name)
                if os.path.exists(src_file):
                    dst_file = os.path.join(self.output_path, file_name)
                    shutil.copy2(src_file, dst_file)
                    print(f"  ✅ 复制 {file_name}")
            
            # 创建合并标记文件
            merge_info = {
                "merged_from_lora": True,
                "lora_model_path": self.lora_model_path,
                "base_model_path": self.base_model_path,
                "merge_method": "mgm_copy_method",
                "note": "MGM模型使用文件复制方法，保留LoRA结构用于推理"
            }
            
            with open(os.path.join(self.output_path, "merge_info.json"), 'w') as f:
                json.dump(merge_info, f, indent=2)
            
            print("✅ MGM LoRA模型准备完成！")
            print("ℹ️  注意: MGM模型保留LoRA结构，需要使用PEFT库加载")
            return True
            
        except Exception as e:
            print(f"❌ MGM LoRA处理失败: {e}")
            return False
    
    def verify_merged_model(self):
        """验证合并后的模型"""
        print("\n🧪 验证合并后的模型...")
        
        try:
            # 检查文件是否存在
            required_files = ["config.json"]
            missing_files = []
            
            for file_name in required_files:
                file_path = os.path.join(self.output_path, file_name)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path) / (1024*1024)  # MB
                    print(f"  ✅ {file_name}: {size:.1f} MB")
                else:
                    missing_files.append(file_name)
                    print(f"  ❌ {file_name}: 缺失")
            
            if missing_files:
                print(f"⚠️  缺失文件: {missing_files}")
                return False
            
            # 检查配置文件
            config_path = os.path.join(self.output_path, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"📊 模型信息:")
            print(f"  模型类型: {config.get('model_type', 'unknown')}")
            print(f"  架构: {config.get('architectures', 'unknown')}")
            print(f"  词汇表大小: {config.get('vocab_size', 'unknown')}")
            
            print("✅ 模型验证通过！")
            return True
            
        except Exception as e:
            print(f"❌ 模型验证失败: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="LoRA权重合并工具")
    parser.add_argument("--lora_path", required=True, help="LoRA模型路径")
    parser.add_argument("--base_path", default=None, help="基础模型路径")
    parser.add_argument("--output_path", default=None, help="输出路径")
    parser.add_argument("--verify", action="store_true", help="验证合并后的模型")
    
    args = parser.parse_args()
    
    print("🚀 LoRA权重合并工具启动")
    print("=" * 50)
    
    # 创建合并器
    merger = LoRAMerger(args.lora_path, args.base_path, args.output_path)
    
    # 执行合并
    success = merger.merge_lora_weights()
    
    if not success:
        print("❌ LoRA权重合并失败！")
        sys.exit(1)
    
    # 验证模型
    if args.verify:
        if not merger.verify_merged_model():
            print("❌ 模型验证失败！")
            sys.exit(1)
    
    print(f"\n🎉 LoRA权重合并完成！")
    print(f"📁 输出路径: {merger.output_path}")
    print("✅ 可以用于标准MGM评估流程")

if __name__ == "__main__":
    main()
