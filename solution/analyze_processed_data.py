#!/usr/bin/env python3
"""
处理后数据质量分析脚本
对比处理前后的数据质量改善情况
"""

import json
import numpy as np
from collections import Counter
import re

def analyze_processed_data():
    """分析处理后的数据质量"""
    print("🔍 分析处理后的数据质量...")
    
    # 读取处理后的数据
    processed_data = []
    with open('output/processed_data/basic_enhanced_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                processed_data.append(json.loads(line.strip()))
    
    print(f"📊 处理后数据量: {len(processed_data):,}")
    
    # 分析文本特征
    texts = []
    for item in processed_data:
        text = item.get('text', '')
        # 清理特殊标记
        clean_text = text.replace('<__dj__image>', '').replace('<|__dj__eoc|>', '').strip()
        texts.append(clean_text)
    
    # 计算文本长度统计
    word_lengths = [len(text.split()) for text in texts if text]
    char_lengths = [len(text) for text in texts if text]
    
    print(f"\n📝 文本质量分析:")
    print(f"  平均词数: {np.mean(word_lengths):.2f}")
    print(f"  中位数词数: {np.median(word_lengths):.2f}")
    print(f"  平均字符数: {np.mean(char_lengths):.2f}")
    print(f"  词数范围: {min(word_lengths)} - {max(word_lengths)}")
    
    # 质量分布
    empty_texts = len([t for t in texts if not t])
    short_texts = len([l for l in word_lengths if l <= 3])
    long_texts = len([l for l in word_lengths if l >= 50])
    
    print(f"\n📋 质量分布:")
    print(f"  空文本: {empty_texts} ({empty_texts/len(texts)*100:.2f}%)")
    print(f"  过短文本(≤3词): {short_texts} ({short_texts/len(word_lengths)*100:.2f}%)")
    print(f"  过长文本(≥50词): {long_texts} ({long_texts/len(word_lengths)*100:.2f}%)")
    
    # 词汇多样性分析
    all_words = []
    for text in texts[:10000]:  # 采样分析
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    vocabulary_diversity = len(word_freq) / len(all_words) if all_words else 0
    
    print(f"\n📈 词汇多样性 (基于1万样本):")
    print(f"  总词汇数: {len(all_words):,}")
    print(f"  唯一词汇数: {len(word_freq):,}")
    print(f"  词汇多样性: {vocabulary_diversity:.4f}")
    
    # 显示一些样本
    print(f"\n📝 处理后样本示例:")
    for i, item in enumerate(processed_data[:5]):
        clean_text = item['text'].replace('<__dj__image>', '').replace('<|__dj__eoc|>', '').strip()
        print(f"  {i+1}. {clean_text}")
    
    # 对比原始数据
    print(f"\n📊 与原始数据对比:")
    print(f"  原始数据: 400,000 → 处理后: {len(processed_data):,}")
    print(f"  数据保留率: {len(processed_data)/400000*100:.1f}%")
    print(f"  原始平均词数: 8.78 → 处理后: {np.mean(word_lengths):.2f}")
    print(f"  原始词汇多样性: 0.0714 → 处理后: {vocabulary_diversity:.4f}")
    
    improvement = (np.mean(word_lengths) - 8.78) / 8.78 * 100
    diversity_improvement = (vocabulary_diversity - 0.0714) / 0.0714 * 100
    
    print(f"\n🎯 改善效果:")
    print(f"  文本长度改善: {improvement:+.1f}%")
    print(f"  词汇多样性改善: {diversity_improvement:+.1f}%")
    
    return {
        'total_samples': len(processed_data),
        'avg_word_length': float(np.mean(word_lengths)),
        'vocabulary_diversity': vocabulary_diversity,
        'retention_rate': len(processed_data)/400000*100,
        'improvement_word_length': improvement,
        'improvement_diversity': diversity_improvement
    }

if __name__ == "__main__":
    results = analyze_processed_data()
    print(f"\n✅ 数据质量分析完成！")
    print(f"📋 建议下一步: 使用处理后的数据进行MGM完整训练")
