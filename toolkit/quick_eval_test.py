#!/usr/bin/env python3

"""
快速LoRA模型评估测试
测试少量样本验证模型是否正常工作
"""

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import argparse

def quick_eval_test(model_path, num_samples=5):
    """快速评估测试"""
    
    print("🚀 开始快速LoRA模型评估测试...")
    print(f"模型路径: {model_path}")
    print(f"测试样本数: {num_samples}")
    
    try:
        # 设置环境
        from mgm.model.builder import load_pretrained_model
        from mgm.utils import disable_torch_init
        from mgm.conversation import conv_templates
        from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mgm.mm_utils import tokenizer_image_token, get_model_name_from_path
        
        disable_torch_init()
        
        # 检查LoRA模型
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            print("❌ 未找到LoRA配置文件")
            return False
            
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path")
        print(f"✅ LoRA配置: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}")
        print(f"📍 基础模型: {base_model_path}")
        
        # 加载模型 - 使用基础模型路径但加载LoRA权重
        print("📥 加载模型...")
        model_name = get_model_name_from_path(model_path)
        
        # 直接使用LoRA模型路径，让MGM的builder处理
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, base_model_path, model_name
        )
        print("✅ 模型加载成功")
        
        # 加载测试数据
        test_data_path = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit/training/data/eval_stage_1/textvqa/llava_textvqa_val_v051_ocr.jsonl"
        image_folder = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit/training/data/eval_stage_1/textvqa/train_images"
        
        print(f"📂 加载测试数据: {test_data_path}")
        
        with open(test_data_path, 'r') as f:
            test_data = [json.loads(line) for line in f]
        
        # 只测试前几个样本
        test_samples = test_data[:num_samples]
        print(f"🎯 开始测试 {len(test_samples)} 个样本...")
        
        results = []
        success_count = 0
        
        for i, sample in enumerate(tqdm(test_samples, desc="评估进度")):
            try:
                question_id = sample["question_id"]
                image_file = sample["image"]
                question = sample["text"]
                
                # 加载图像
                image_path = os.path.join(image_folder, image_file)
                if not os.path.exists(image_path):
                    print(f"⚠️  图像不存在: {image_path}")
                    continue
                
                image = Image.open(image_path).convert('RGB')
                
                # 准备输入
                qs = question
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
                
                # 推理
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=128,
                        use_cache=True
                    )
                
                # 解码输出
                input_token_len = input_ids.shape[1]
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                
                result = {
                    "question_id": question_id,
                    "question": question,
                    "answer": outputs,
                    "image": image_file
                }
                results.append(result)
                success_count += 1
                
                print(f"✅ 样本 {i+1}/{len(test_samples)}")
                print(f"   问题: {question[:100]}...")
                print(f"   回答: {outputs}")
                print()
                
            except Exception as e:
                print(f"❌ 样本 {i+1} 失败: {e}")
                continue
        
        # 保存结果
        output_file = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/output/结果分析/quick_eval_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"🎉 评估完成!")
        print(f"✅ 成功样本: {success_count}/{len(test_samples)}")
        print(f"📁 结果保存至: {output_file}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='快速LoRA模型评估测试')
    parser.add_argument('--model-path', required=True, help='LoRA模型路径')
    parser.add_argument('--num-samples', type=int, default=5, help='测试样本数量')
    
    args = parser.parse_args()
    
    success = quick_eval_test(args.model_path, args.num_samples)
    
    if success:
        print("\n🎉 快速评估测试成功!")
    else:
        print("\n💥 快速评估测试失败!")
        exit(1)

if __name__ == "__main__":
    main()
