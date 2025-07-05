#!/usr/bin/env python3
"""
LoRA推理引擎
支持LoRA模型的完整推理功能
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from PIL import Image
import argparse
from typing import List, Dict, Any, Optional

class LoRAInferenceEngine:
    """LoRA模型推理引擎"""
    
    def __init__(self, lora_model_path: str, base_model_path: str = None):
        """
        初始化LoRA推理引擎
        
        Args:
            lora_model_path: LoRA模型路径
            base_model_path: 基础模型路径，如果为None则从LoRA配置中获取
        """
        self.lora_model_path = lora_model_path
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 初始化LoRA推理引擎")
        print(f"📁 LoRA模型路径: {lora_model_path}")
        print(f"🖥️  设备: {self.device}")
        
    def load_model(self):
        """加载LoRA模型"""
        try:
            # 读取LoRA配置
            adapter_config_path = os.path.join(self.lora_model_path, "adapter_config.json")
            with open(adapter_config_path, 'r') as f:
                lora_config = json.load(f)
            
            # 确定基础模型路径
            if self.base_model_path is None:
                base_model_name = lora_config.get("base_model_name_or_path", "")
                if "gemma-2b-it" in base_model_name:
                    self.base_model_path = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit/training/model_zoo/LLM/gemma/gemma-2b-it"
                else:
                    self.base_model_path = base_model_name
            
            print(f"📋 基础模型: {self.base_model_path}")
            print(f"🔧 LoRA配置: rank={lora_config.get('r')}, alpha={lora_config.get('lora_alpha')}")
            
            # 加载tokenizer
            print("🔤 加载Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            print("🧠 加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            print("🔗 加载LoRA适配器...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_model_path,
                torch_dtype=torch.float16
            )
            
            # 设置为评估模式
            self.model.eval()
            
            print("✅ LoRA模型加载成功！")
            return True
            
        except Exception as e:
            print(f"❌ LoRA模型加载失败: {e}")
            return False
    
    def generate_text(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 生成温度
            
        Returns:
            生成的文本
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        try:
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除输入部分
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"❌ 文本生成失败: {e}")
            return ""
    
    def test_basic_inference(self) -> bool:
        """测试基础推理功能"""
        print("\n🧪 测试基础推理功能...")
        
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Describe a beautiful sunset.",
        ]
        
        success_count = 0
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n测试 {i}: {prompt}")
            try:
                response = self.generate_text(prompt, max_length=100, temperature=0.1)
                if response and len(response.strip()) > 0:
                    print(f"✅ 响应: {response[:100]}...")
                    success_count += 1
                else:
                    print("❌ 生成空响应")
            except Exception as e:
                print(f"❌ 生成失败: {e}")
        
        success_rate = success_count / len(test_prompts)
        print(f"\n📊 基础推理测试结果: {success_count}/{len(test_prompts)} ({success_rate*100:.1f}%)")
        
        return success_rate > 0.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "loaded",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lora_parameters": trainable_params,
            "base_model_path": self.base_model_path,
            "lora_model_path": self.lora_model_path,
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0
        }

def main():
    parser = argparse.ArgumentParser(description="LoRA推理引擎测试")
    parser.add_argument("--lora_path", required=True, help="LoRA模型路径")
    parser.add_argument("--base_path", default=None, help="基础模型路径")
    parser.add_argument("--test_inference", action="store_true", help="测试推理功能")
    
    args = parser.parse_args()
    
    print("🚀 LoRA推理引擎启动")
    print("=" * 50)
    
    # 创建推理引擎
    engine = LoRAInferenceEngine(args.lora_path, args.base_path)
    
    # 加载模型
    if not engine.load_model():
        print("❌ 模型加载失败，退出")
        sys.exit(1)
    
    # 显示模型信息
    model_info = engine.get_model_info()
    print(f"\n📊 模型信息:")
    print(f"  总参数: {model_info['total_parameters']:,}")
    print(f"  LoRA参数: {model_info['lora_parameters']:,}")
    print(f"  词汇表大小: {model_info['vocab_size']:,}")
    
    # 测试推理
    if args.test_inference:
        success = engine.test_basic_inference()
        if success:
            print("\n🎉 LoRA推理引擎测试成功！")
        else:
            print("\n❌ LoRA推理引擎测试失败！")
            sys.exit(1)
    
    print("\n✅ LoRA推理引擎就绪")

if __name__ == "__main__":
    main()
