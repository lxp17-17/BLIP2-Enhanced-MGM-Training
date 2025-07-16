#!/usr/bin/env python3
"""
简化的MMBench评估测试脚本
用于测试MMBench评估是否可以正常工作
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

all_options = ['A', 'B', 'C', 'D']

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

def test_mmbench_data_loading():
    """测试MMBench数据加载"""
    print("🔍 测试MMBench数据加载...")
    
    data_path = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit/training/data/eval_stage_1/mmbench/mmbench_dev_20230712.tsv"
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return False
    
    try:
        questions = pd.read_table(data_path)
        print(f"✅ 成功加载 {len(questions)} 个问题")
        
        # 显示前几个问题的信息
        for i in range(min(3, len(questions))):
            row = questions.iloc[i]
            options = get_options(row, all_options)
            print(f"  问题 {i+1}: {len(options)} 个选项")
            print(f"    问题: {row['question'][:50]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_image_loading():
    """测试图像加载"""
    print("🖼️  测试图像加载...")
    
    data_path = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit/training/data/eval_stage_1/mmbench/mmbench_dev_20230712.tsv"
    
    try:
        questions = pd.read_table(data_path)
        
        # 测试加载前几张图像
        for i in range(min(3, len(questions))):
            row = questions.iloc[i]
            try:
                image = load_image_from_base64(row['image'])
                print(f"  ✅ 图像 {i+1}: {image.size}")
            except Exception as e:
                print(f"  ❌ 图像 {i+1} 加载失败: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ 图像测试失败: {e}")
        return False

def generate_mock_results():
    """生成模拟MMBench结果用于测试"""
    print("🎭 生成模拟MMBench结果...")
    
    data_path = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/toolkit/training/data/eval_stage_1/mmbench/mmbench_dev_20230712.tsv"
    output_dir = "/home/robot/lhp/projects/0625TCSyn/dj_synth_challenge/output/eval_results/test-mmbench/mmbench/answers/mmbench_dev_20230712"
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test-mmbench.jsonl")
    
    try:
        questions = pd.read_table(data_path)
        
        # 只处理前10个问题用于测试
        test_questions = questions.head(10)
        
        with open(output_file, 'w') as f:
            for index, row in test_questions.iterrows():
                options = get_options(row, all_options)
                cur_option_char = all_options[:len(options)]
                
                idx = row['index']
                question = row['question']
                hint = row['hint']
                
                if not is_none(hint):
                    question = hint + '\n' + question
                for option_char, option in zip(all_options[:len(options)], options):
                    question = question + '\n' + option_char + '. ' + option
                
                # 生成模拟答案（随机选择一个选项）
                import random
                mock_answer = random.choice(cur_option_char)
                
                ans_id = shortuuid.uuid()
                result = {
                    "question_id": idx,
                    "round_id": 0,
                    "prompt": question,
                    "text": mock_answer,
                    "options": options,
                    "option_char": cur_option_char,
                    "answer_id": ans_id,
                    "model_id": "test-mmbench",
                    "metadata": {}
                }
                
                f.write(json.dumps(result) + "\n")
        
        print(f"✅ 生成了 {len(test_questions)} 个模拟结果")
        print(f"📁 输出文件: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟结果生成失败: {e}")
        return False

def main():
    print("🎯 MMBench评估测试")
    print("=" * 50)
    
    # 测试数据加载
    if not test_mmbench_data_loading():
        print("❌ 数据加载测试失败")
        return False
    
    # 测试图像加载
    if not test_image_loading():
        print("❌ 图像加载测试失败")
        return False
    
    # 生成模拟结果
    if not generate_mock_results():
        print("❌ 模拟结果生成失败")
        return False
    
    print("\n🎉 MMBench评估测试完成！")
    print("✅ 所有测试通过")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
