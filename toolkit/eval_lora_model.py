#!/usr/bin/env python3

"""
LoRA模型评估脚本
专门用于评估LoRA微调后的MGM模型
"""

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    length = len(lst)
    return [lst[i*length//n:(i+1)*length//n] for i in range(n)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    
    # 对于LoRA模型，需要特殊的加载方式
    model_path = args.model_path
    model_base = args.model_base
    model_name = get_model_name_from_path(model_path)
    
    print(f"Loading LoRA model from: {model_path}")
    print(f"Model base: {model_base}")
    print(f"Model name: {model_name}")
    
    try:
        # 检查是否是LoRA模型
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            print("Detected LoRA model, using PEFT loading...")
            from peft import PeftModel

            # 从adapter_config.json中获取base model路径
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            model_base = adapter_config.get("base_model_name_or_path")
            print(f"Base model path from adapter config: {model_base}")

            # 加载基础模型
            print(f"Loading base model: {model_base}")
            tokenizer, base_model, image_processor, context_len = load_pretrained_model(
                model_base, None, get_model_name_from_path(model_base), load_8bit=args.load_8bit
            )

            # 加载LoRA适配器
            print(f"Loading LoRA adapter from: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)

            # 检查是否需要合并权重
            if hasattr(model, 'merge_and_unload'):
                print("Merging LoRA weights...")
                model = model.merge_and_unload()

            print("LoRA model loaded successfully!")

        else:
            print("Standard model detected, using normal loading...")
            # 标准模型加载
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, model_base, model_name, load_8bit=args.load_8bit
            )

    except Exception as e:
        print(f"LoRA loading failed: {e}")
        print("Falling back to standard loading...")
        # 回退到标准加载方式
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit=args.load_8bit
        )

    # 加载问题数据
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    
    answers_file = open(args.answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        
        # 处理图像
        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
            
        image = Image.open(image_path).convert('RGB')
        
        # 处理对话
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 处理图像
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # 生成回答
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        answers_file.write(json.dumps({
            "question_id": idx,
            "prompt": qs,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
        answers_file.flush()

    answers_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="gemma")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load_8bit", action="store_true")
    args = parser.parse_args()

    eval_model(args)
