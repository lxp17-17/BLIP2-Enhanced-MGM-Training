#!/usr/bin/env python3

"""
LoRA模型评估测试脚本
用于快速测试LoRA模型是否能正常加载和推理
"""

import argparse
import torch
import os
import json
from PIL import Image

def test_lora_model(model_path, base_model_path, test_image_path, test_question):
    """测试LoRA模型加载和推理"""
    
    print("🚀 开始LoRA模型测试...")
    print(f"模型路径: {model_path}")
    print(f"基础模型: {base_model_path}")
    print(f"测试图片: {test_image_path}")
    print(f"测试问题: {test_question}")
    
    try:
        # 设置环境
        from mgm.model.builder import load_pretrained_model
        from mgm.utils import disable_torch_init
        from mgm.conversation import conv_templates
        from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mgm.mm_utils import tokenizer_image_token, get_model_name_from_path
        
        disable_torch_init()
        
        # 检查是否是LoRA模型
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print("✅ 检测到LoRA模型")
            
            # 读取LoRA配置
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            print(f"LoRA配置: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")
            
            # 使用PEFT加载
            from peft import PeftModel
            
            # 加载基础模型
            print("📥 加载基础模型...")
            model_name = get_model_name_from_path(base_model_path)
            tokenizer, base_model, image_processor, context_len = load_pretrained_model(
                base_model_path, None, model_name
            )
            print("✅ 基础模型加载成功")
            
            # 加载LoRA适配器
            print("📥 加载LoRA适配器...")
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✅ LoRA适配器加载成功")
            
            # 合并权重（可选）
            if hasattr(model, 'merge_and_unload'):
                print("🔄 合并LoRA权重...")
                model = model.merge_and_unload()
                print("✅ 权重合并完成")
            
        else:
            print("📥 标准模型加载...")
            model_name = get_model_name_from_path(model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, None, model_name
            )
        
        print("✅ 模型加载完成")
        
        # 测试推理
        if test_image_path and os.path.exists(test_image_path):
            print("🖼️  加载测试图片...")
            image = Image.open(test_image_path).convert('RGB')
            
            # 准备输入
            qs = test_question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            conv = conv_templates["gemma"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # 处理图像
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            print("🤖 开始推理...")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=512,
                    use_cache=True
                )
            
            # 解码输出
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            
            print("✅ 推理完成!")
            print(f"🎯 问题: {test_question}")
            print(f"🤖 回答: {outputs}")
            
            return True, outputs
        else:
            print("⚠️  跳过推理测试（无测试图片）")
            return True, "模型加载成功，但未进行推理测试"
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='LoRA模型测试脚本')
    parser.add_argument('--model-path', required=True, help='LoRA模型路径')
    parser.add_argument('--base-model', required=True, help='基础模型路径')
    parser.add_argument('--test-image', help='测试图片路径')
    parser.add_argument('--test-question', default='What is in this image?', help='测试问题')
    
    args = parser.parse_args()
    
    success, result = test_lora_model(
        args.model_path,
        args.base_model, 
        args.test_image,
        args.test_question
    )
    
    if success:
        print("\n🎉 测试成功!")
        print(f"结果: {result}")
    else:
        print(f"\n💥 测试失败: {result}")
        exit(1)

if __name__ == "__main__":
    main()
