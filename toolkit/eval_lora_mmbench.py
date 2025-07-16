#!/usr/bin/env python3
"""
支持LoRA的MMBench评估脚本
基于原始MMBench评估，适配LoRA模型加载
"""

import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
from PIL import Image
import math

# 导入MGM相关模块
import sys
sys.path.append('/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit/training')

from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.utils import disable_torch_init
from mgm.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

# LoRA相关导入
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig

all_options = ['A', 'B', 'C', 'D']

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_lora_model(model_path):
    """
    加载LoRA模型用于推理
    """
    print(f"🔄 加载LoRA模型: {model_path}")

    try:
        # 检查是否是合并后的LoRA模型
        merge_info_path = os.path.join(model_path, "merge_info.json")
        if os.path.exists(merge_info_path):
            print("📋 检测到合并后的LoRA模型")
            with open(merge_info_path, 'r') as f:
                merge_info = json.load(f)

            # 对于合并后的模型，直接删除adapter相关文件来避免PEFT版本问题
            adapter_files = ['adapter_config.json', 'adapter_model.bin']
            temp_moved = []

            for adapter_file in adapter_files:
                adapter_path = os.path.join(model_path, adapter_file)
                if os.path.exists(adapter_path):
                    temp_path = adapter_path + '.temp_moved'
                    os.rename(adapter_path, temp_path)
                    temp_moved.append((adapter_path, temp_path))
                    print(f"🔄 临时移动文件: {adapter_file}")

            try:
                # 使用标准方式加载合并后的模型
                from mgm.model.builder import load_pretrained_model
                model_name = get_model_name_from_path(model_path)
                tokenizer, model, image_processor, context_len = load_pretrained_model(
                    model_path, None, model_name
                )
                print(f"✅ 成功加载合并后的LoRA模型")
                return tokenizer, model, image_processor, context_len
            finally:
                # 恢复移动的文件
                for original_path, temp_path in temp_moved:
                    if os.path.exists(temp_path):
                        os.rename(temp_path, original_path)
                        print(f"🔄 恢复文件: {os.path.basename(original_path)}")

        else:
            # 原始LoRA模型，需要加载base model + LoRA
            print("📋 检测到原始LoRA模型，需要合并权重")

            # 读取LoRA配置
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path

            print(f"🔧 Base模型路径: {base_model_path}")

            # 加载base模型
            from mgm.model.builder import load_pretrained_model
            model_name = get_model_name_from_path(base_model_path)
            tokenizer, base_model, image_processor, context_len = load_pretrained_model(
                base_model_path, None, model_name
            )

            # 加载LoRA权重
            print("🔄 加载LoRA权重...")
            model = PeftModel.from_pretrained(base_model, model_path)

            # 合并权重以提高推理效率
            print("🔄 合并LoRA权重...")
            model = model.merge_and_unload()

            print(f"✅ 成功加载并合并LoRA模型")
            return tokenizer, model, image_processor, context_len

    except Exception as e:
        print(f"❌ LoRA模型加载失败: {e}")
        print("🔄 尝试使用标准方式加载...")

        # 回退到标准加载方式
        from mgm.model.builder import load_pretrained_model
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name
        )
        return tokenizer, model, image_processor, context_len

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if pd.isna(option_value) or option_value == '':
            break
        parsed_options.append(option_value)
    return parsed_options

def is_none(value):
    return pd.isna(value) or value == ''

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    # 使用LoRA加载函数
    tokenizer, model, image_processor, context_len = load_lora_model(model_path)
    model_name = get_model_name_from_path(model_path)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    print(f"🚀 开始MMBench评估")
    print(f"📊 问题数量: {len(questions)}")
    print(f"🎯 对话模式: {args.conv_mode}")
    print(f"🌡️  温度: {args.temperature}")

    for index, row in tqdm(questions.iterrows(), total=len(questions), desc="MMBench评估"):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            
            if hasattr(model, "update_prompt"):
                model.update_prompt([[cur_prompt]])
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            images = process_images([image], image_processor, model.config).to(dtype=torch.float16, device='cuda', non_blocking=True)

            images_aux = None
            if hasattr(model.config, 'image_aux_size') and model.config.image_aux_size is not None:
                images_aux = process_images([image], image_processor, model.config, aux_size=model.config.image_aux_size).to(dtype=torch.float16, device='cuda', non_blocking=True)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            terminators = None
            if tokenizer.eos_token is not None:
                terminators = [tokenizer.eos_token_id]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    images_aux=images_aux,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    
    ans_file.close()
    print(f"✅ MMBench评估完成！结果保存至: {answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")

    args = parser.parse_args()

    eval_model(args)
