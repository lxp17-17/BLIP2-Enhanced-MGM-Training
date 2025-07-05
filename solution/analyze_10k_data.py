#!/usr/bin/env python3
"""
10k基线数据分析脚本
分析数据格式、内容特征，为数据处理策略提供参考
"""

import json
import os
from collections import Counter

def analyze_10k_data():
    """分析10k数据集"""
    data_path = "input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl"
    
    print("=== 10K基线数据分析 ===\n")
    
    # 读取所有数据
    data_samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_samples.append(json.loads(line.strip()))
    
    print(f"📊 数据量: {len(data_samples)} 条记录")
    
    # 分析文本长度
    text_lengths = [len(sample['text']) for sample in data_samples]
    print(f"📝 文本长度统计:")
    print(f"   - 平均长度: {sum(text_lengths)/len(text_lengths):.1f} 字符")
    print(f"   - 最短: {min(text_lengths)} 字符") 
    print(f"   - 最长: {max(text_lengths)} 字符")
    
    # 分析文本内容特征
    print(f"\n📋 文本内容分析:")
    
    # 统计是否包含特殊标记
    dj_image_count = sum(1 for sample in data_samples if '<__dj__image>' in sample['text'])
    eoc_count = sum(1 for sample in data_samples if '<|__dj__eoc|>' in sample['text'])
    
    print(f"   - 包含 <__dj__image> 标记: {dj_image_count} ({dj_image_count/len(data_samples)*100:.1f}%)")
    print(f"   - 包含 <|__dj__eoc|> 标记: {eoc_count} ({eoc_count/len(data_samples)*100:.1f}%)")
    
    # 分析图像信息
    print(f"\n🖼️ 图像信息分析:")
    
    image_counts = [len(sample['images']) for sample in data_samples]
    print(f"   - 每条记录图像数量: 最少{min(image_counts)}, 最多{max(image_counts)}, 平均{sum(image_counts)/len(image_counts):.1f}")
    
    # 检查图像文件是否存在
    existing_images = 0
    missing_images = 0
    
    for sample in data_samples[:100]:  # 检查前100个样本
        for img_path in sample['images']:
            full_path = f"input/pretrain_stage_1_10k/{img_path}"
            if os.path.exists(full_path):
                existing_images += 1
            else:
                missing_images += 1
    
    print(f"   - 图像文件检查 (前100样本): 存在{existing_images}, 缺失{missing_images}")
    
    # 分析文本内容模式
    print(f"\n📈 文本内容模式:")
    
    # 移除特殊标记后的纯文本
    clean_texts = []
    for sample in data_samples:
        text = sample['text']
        text = text.replace('<__dj__image>', '').replace('<|__dj__eoc|>', '').strip()
        clean_texts.append(text)
    
    # 统计常见词汇
    all_words = []
    for text in clean_texts:
        words = text.lower().split()
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    print(f"   - 总词汇数: {len(word_freq)}")
    print(f"   - 最常见的10个词:")
    for word, count in word_freq.most_common(10):
        print(f"     {word}: {count}次")
    
    # 展示几个代表性样本
    print(f"\n📝 代表性样本:")
    for i, sample in enumerate(data_samples[:5]):
        clean_text = sample['text'].replace('<__dj__image>', '').replace('<|__dj__eoc|>', '').strip()
        print(f"   {i+1}. ID: {sample['id']}")
        print(f"      文本: {clean_text}")
        print(f"      图像: {sample['images'][0]}")
        print()
    
    # 数据质量评估
    print(f"🔍 数据质量评估:")
    
    # 检查异常短文本
    short_texts = [sample for sample in data_samples if len(sample['text'].replace('<__dj__image>', '').replace('<|__dj__eoc|>', '').strip()) < 10]
    print(f"   - 异常短文本 (<10字符): {len(short_texts)} 条 ({len(short_texts)/len(data_samples)*100:.1f}%)")
    
    # 检查可能的重复
    text_hashes = set()
    duplicates = 0
    for sample in data_samples:
        text_hash = hash(sample['text'])
        if text_hash in text_hashes:
            duplicates += 1
        text_hashes.add(text_hash)
    
    print(f"   - 可能的重复文本: {duplicates} 条 ({duplicates/len(data_samples)*100:.1f}%)")
    
    print(f"\n✅ 数据分析完成!")
    print(f"\n💡 建议的数据处理策略:")
    print(f"   1. 数据质量很好，格式统一")
    print(f"   2. 可以尝试改进描述质量（当前描述较简单）")
    print(f"   3. 可以考虑数据增强技术")
    print(f"   4. 特殊标记格式已标准化，便于处理")

if __name__ == "__main__":
    analyze_10k_data()