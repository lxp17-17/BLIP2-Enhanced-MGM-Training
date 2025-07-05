#!/usr/bin/env python3
"""
完整种子数据集深度探索脚本 - 40万数据分析
全面分析40万完整种子数据集的特征，生成详细的图文分析报告
支持采样分析、可视化图表生成和统计报告
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from PIL import Image
import hashlib
import random
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
sns.set_palette("husl")

class FullDatasetExplorer:
    def __init__(self, sample_size=50000):
        """
        初始化完整数据集探索器
        Args:
            sample_size: 采样大小，用于图像分析等耗时操作
        """
        self.data = []
        self.sample_data = []
        self.sample_size = sample_size
        self.analysis_results = {}
        self.output_dir = Path("output/full_dataset_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置图表保存目录
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 初始化完整数据集探索器")
        print(f"📊 采样大小: {sample_size:,}")
        print(f"📁 输出目录: {self.output_dir}")

    def load_full_dataset(self):
        """加载完整的40万数据集"""
        print("🔍 加载完整种子数据集...")
        
        data_file = "input/pretrain_stage_1/mgm_pretrain_stage_1.jsonl"
        
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            return False
        
        print(f"📊 开始加载数据文件: {data_file}")
        start_time = time.time()
        
        # 使用进度条加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="加载数据"), 1):
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        item['_line_number'] = line_num
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                        continue
        
        load_time = time.time() - start_time
        print(f"✅ 成功加载 {len(self.data):,} 条数据，耗时: {load_time:.1f}秒")
        
        # 创建采样数据用于图像分析
        if len(self.data) > self.sample_size:
            print(f"🎲 创建 {self.sample_size:,} 样本的随机采样...")
            self.sample_data = random.sample(self.data, self.sample_size)
        else:
            self.sample_data = self.data.copy()
            
        print(f"📈 采样数据大小: {len(self.sample_data):,}")
        return True

    def analyze_data_structure(self):
        """分析数据结构"""
        print("\n=== 数据结构分析 ===")
        
        if not self.data:
            return
        
        # 分析字段分布
        field_counts = defaultdict(int)
        field_types = defaultdict(set)
        
        # 使用采样数据进行字段分析
        sample_for_structure = self.data[:10000]  # 使用前1万条进行结构分析
        
        for item in tqdm(sample_for_structure, desc="分析数据结构"):
            for key, value in item.items():
                field_counts[key] += 1
                field_types[key].add(type(value).__name__)
        
        print("📋 字段统计:")
        for field, count in sorted(field_counts.items()):
            coverage = count / len(sample_for_structure) * 100
            types = ', '.join(field_types[field])
            print(f"  {field}: {count:,}/{len(sample_for_structure):,} ({coverage:.1f}%) - 类型: {types}")
        
        # 检查必要字段
        required_fields = ['text', 'images', 'id']
        missing_fields = []
        
        for field in required_fields:
            if field not in field_counts or field_counts[field] < len(sample_for_structure):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"⚠️ 缺失必要字段: {missing_fields}")
        else:
            print("✅ 所有必要字段完整")
        
        self.analysis_results['data_structure'] = {
            'total_samples': len(self.data),
            'analyzed_samples': len(sample_for_structure),
            'field_counts': dict(field_counts),
            'field_types': {k: list(v) for k, v in field_types.items()},
            'missing_fields': missing_fields
        }

    def analyze_text_features_comprehensive(self):
        """全面分析文本特征"""
        print("\n=== 文本特征全面分析 ===")
        
        texts = []
        original_texts = []
        
        print("📝 提取文本数据...")
        for item in tqdm(self.data, desc="提取文本"):
            original_text = item.get('text', '')
            original_texts.append(original_text)
            
            # 清理Data-Juicer的特殊标记
            clean_text = original_text.replace('<__dj__image>', '').replace('<|__dj__eoc|>', '').strip()
            texts.append(clean_text)
        
        # 基础统计
        print("📊 计算基础统计...")
        word_lengths = [len(text.split()) for text in texts if text]
        char_lengths = [len(text) for text in texts if text]
        sentence_counts = [len(re.split(r'[.!?]+', text)) for text in texts if text]
        
        print(f"📊 基础统计 (总样本: {len(texts):,}):")
        print(f"  平均词数: {np.mean(word_lengths):.2f} (中位数: {np.median(word_lengths):.2f})")
        print(f"  词数范围: {min(word_lengths)} - {max(word_lengths)}")
        print(f"  平均字符数: {np.mean(char_lengths):.2f}")
        print(f"  平均句子数: {np.mean(sentence_counts):.2f}")
        
        # 生成词长分布图
        self.create_text_length_charts(word_lengths, char_lengths)
        
        # 质量分析
        empty_texts = len([t for t in texts if not t])
        short_texts = len([l for l in word_lengths if l <= 3])
        long_texts = len([l for l in word_lengths if l >= 50])
        
        print(f"\n📋 质量分析:")
        print(f"  空文本: {empty_texts:,} ({empty_texts/len(texts)*100:.1f}%)")
        print(f"  过短文本(≤3词): {short_texts:,} ({short_texts/len(word_lengths)*100:.1f}%)")
        print(f"  过长文本(≥50词): {long_texts:,} ({long_texts/len(word_lengths)*100:.1f}%)")
        
        # 词频分析 (使用采样数据)
        print("📈 进行词频分析...")
        sample_texts = random.sample(texts, min(50000, len(texts)))
        all_words = []
        for text in tqdm(sample_texts, desc="词频分析"):
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        print(f"\n📈 词频分析 (基于 {len(sample_texts):,} 样本):")
        print(f"  总词汇: {len(all_words):,}, 唯一词汇: {len(word_freq):,}")
        print("  最常见的15个词:")
        for word, count in word_freq.most_common(15):
            print(f"    {word}: {count:,} ({count/len(all_words)*100:.2f}%)")
        
        # 生成词频图表
        self.create_word_frequency_chart(word_freq)
        
        # 保存分析结果
        self.analysis_results['text_features'] = {
            'total_texts': len(texts),
            'avg_word_length': float(np.mean(word_lengths)),
            'median_word_length': float(np.median(word_lengths)),
            'std_word_length': float(np.std(word_lengths)),
            'avg_char_length': float(np.mean(char_lengths)),
            'avg_sentence_count': float(np.mean(sentence_counts)),
            'empty_texts': empty_texts,
            'short_texts': short_texts,
            'long_texts': long_texts,
            'unique_words': len(word_freq),
            'total_words_analyzed': len(all_words),
            'top_words': word_freq.most_common(50),
            'word_length_percentiles': {
                '25th': float(np.percentile(word_lengths, 25)),
                '50th': float(np.percentile(word_lengths, 50)),
                '75th': float(np.percentile(word_lengths, 75)),
                '90th': float(np.percentile(word_lengths, 90)),
                '95th': float(np.percentile(word_lengths, 95))
            }
        }

    def create_text_length_charts(self, word_lengths, char_lengths):
        """创建文本长度分布图表"""
        print("📊 生成文本长度分布图表...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 词数分布直方图
        ax1.hist(word_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Word Count Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(word_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(word_lengths):.1f}')
        ax1.legend()
        
        # 词数分布箱线图
        ax2.boxplot(word_lengths, vert=True)
        ax2.set_title('Word Count Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Words')
        
        # 字符数分布直方图
        ax3.hist(char_lengths, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Character Count Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Characters')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(char_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(char_lengths):.1f}')
        ax3.legend()
        
        # 词数vs字符数散点图
        sample_indices = random.sample(range(len(word_lengths)), min(10000, len(word_lengths)))
        sample_words = [word_lengths[i] for i in sample_indices]
        sample_chars = [char_lengths[i] for i in sample_indices]
        
        ax4.scatter(sample_words, sample_chars, alpha=0.5, s=1)
        ax4.set_title('Words vs Characters Correlation', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Words')
        ax4.set_ylabel('Number of Characters')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'text_length_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 文本长度分析图表已保存: {self.charts_dir / 'text_length_analysis.png'}")

    def create_word_frequency_chart(self, word_freq):
        """创建词频分析图表"""
        print("📊 生成词频分析图表...")
        
        # 获取前30个最常见词汇
        top_words = word_freq.most_common(30)
        words, counts = zip(*top_words)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 词频条形图
        bars = ax1.bar(range(len(words)), counts, color='coral', alpha=0.8)
        ax1.set_title('Top 30 Most Frequent Words', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Words')
        ax1.set_ylabel('Frequency')
        ax1.set_xticks(range(len(words)))
        ax1.set_xticklabels(words, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # 词频分布对数图
        all_counts = list(word_freq.values())
        ax2.hist(all_counts, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.set_title('Word Frequency Distribution (Log Scale)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Number of Words')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'word_frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 词频分析图表已保存: {self.charts_dir / 'word_frequency_analysis.png'}")

    def analyze_image_features_sampled(self):
        """基于采样数据分析图像特征"""
        print(f"\n=== 图像特征分析 (基于 {len(self.sample_data):,} 样本) ===")

        image_paths = []
        missing_images = 0
        image_sizes = []
        image_formats = []

        print("🖼️ 分析图像特征...")
        for item in tqdm(self.sample_data, desc="分析图像"):
            if 'images' in item and item['images']:
                img_path = item['images'][0]
                image_paths.append(img_path)

                # 构建完整的图像路径
                full_img_path = os.path.join('input/pretrain_stage_1', img_path)

                # 检查文件是否存在
                if os.path.exists(full_img_path):
                    try:
                        # 获取图像信息
                        with Image.open(full_img_path) as img:
                            width, height = img.size
                            image_sizes.append((width, height))
                            image_formats.append(img.format)
                    except Exception as e:
                        missing_images += 1
                else:
                    missing_images += 1

        print(f"📊 图像基础统计:")
        print(f"  图像总数: {len(image_paths):,}")
        print(f"  缺失图像: {missing_images:,}")
        print(f"  图像完整率: {(len(image_paths)-missing_images)/len(image_paths)*100:.1f}%")

        if image_sizes:
            # 尺寸分析
            widths = [size[0] for size in image_sizes]
            heights = [size[1] for size in image_sizes]
            aspect_ratios = [w/h for w, h in image_sizes]

            print(f"\n📐 尺寸分析:")
            print(f"  平均宽度: {np.mean(widths):.1f}px (范围: {min(widths)}-{max(widths)})")
            print(f"  平均高度: {np.mean(heights):.1f}px (范围: {min(heights)}-{max(heights)})")
            print(f"  平均宽高比: {np.mean(aspect_ratios):.2f}")

            # 生成图像分析图表
            self.create_image_analysis_charts(widths, heights, aspect_ratios, image_formats)

            # 尺寸分布
            small_images = len([w for w in widths if w < 224])
            medium_images = len([w for w in widths if 224 <= w < 512])
            large_images = len([w for w in widths if w >= 512])

            print(f"\n📊 尺寸分布:")
            print(f"  小图像(<224px): {small_images:,} ({small_images/len(widths)*100:.1f}%)")
            print(f"  中等图像(224-512px): {medium_images:,} ({medium_images/len(widths)*100:.1f}%)")
            print(f"  大图像(≥512px): {large_images:,} ({large_images/len(widths)*100:.1f}%)")

        # 格式分析
        if image_formats:
            format_counts = Counter(image_formats)
            print(f"\n🎨 格式分析:")
            for fmt, count in format_counts.most_common():
                print(f"  {fmt}: {count:,} ({count/len(image_formats)*100:.1f}%)")

        # 保存分析结果
        self.analysis_results['image_features'] = {
            'total_images_sampled': len(image_paths),
            'missing_images': missing_images,
            'completion_rate': (len(image_paths)-missing_images)/len(image_paths)*100 if image_paths else 0,
            'avg_width': float(np.mean(widths)) if widths else 0,
            'avg_height': float(np.mean(heights)) if heights else 0,
            'avg_aspect_ratio': float(np.mean(aspect_ratios)) if aspect_ratios else 0,
            'size_distribution': {
                'small': small_images if image_sizes else 0,
                'medium': medium_images if image_sizes else 0,
                'large': large_images if image_sizes else 0
            } if image_sizes else {},
            'format_distribution': dict(format_counts) if image_formats else {},
            'sample_size': len(self.sample_data)
        }

    def create_image_analysis_charts(self, widths, heights, aspect_ratios, image_formats):
        """创建图像分析图表"""
        print("📊 生成图像分析图表...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 图像宽度分布
        ax1.hist(widths, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title('Image Width Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(widths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(widths):.0f}px')
        ax1.legend()

        # 图像高度分布
        ax2.hist(heights, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('Image Height Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(heights), color='red', linestyle='--',
                   label=f'Mean: {np.mean(heights):.0f}px')
        ax2.legend()

        # 宽高比分布
        ax3.hist(aspect_ratios, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax3.set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Aspect Ratio (Width/Height)')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(aspect_ratios), color='red', linestyle='--',
                   label=f'Mean: {np.mean(aspect_ratios):.2f}')
        ax3.legend()

        # 图像格式分布
        format_counts = Counter(image_formats)
        formats, counts = zip(*format_counts.most_common())
        ax4.pie(counts, labels=formats, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Image Format Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.charts_dir / 'image_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 图像分析图表已保存: {self.charts_dir / 'image_analysis.png'}")

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📝 生成综合分析报告...")

        import time
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')

        # 获取分析结果
        data_structure = self.analysis_results.get('data_structure', {})
        text_features = self.analysis_results.get('text_features', {})
        image_features = self.analysis_results.get('image_features', {})

        report_content = f"""# 完整种子数据集深度分析报告

## 报告信息
- **生成时间**: {current_time}
- **数据集**: 完整40万种子数据集 (pretrain_stage_1)
- **总样本数**: {data_structure.get('total_samples', 0):,}
- **图像分析采样**: {image_features.get('sample_size', 0):,} 样本

## 🎯 执行摘要

### 数据规模
- **总数据量**: {data_structure.get('total_samples', 0):,} 条记录
- **数据完整性**: {'✅ 完整' if not data_structure.get('missing_fields') else '⚠️ 有缺失'}
- **预估处理时间**: {self._estimate_processing_time(data_structure.get('total_samples', 0))}

### 关键发现
1. **文本特征**: 平均 {text_features.get('avg_word_length', 0):.1f} 词/样本
2. **图像质量**: {image_features.get('completion_rate', 0):.1f}% 完整率
3. **词汇多样性**: {text_features.get('unique_words', 0):,} 唯一词汇

## 📊 详细分析

### 1. 数据结构概览

#### 基础统计
- **总样本数**: {data_structure.get('total_samples', 0):,}
- **字段完整性**: {'✅ 完整' if not data_structure.get('missing_fields') else '⚠️ 有缺失'}

#### 字段分布
{self._format_field_distribution(data_structure.get('field_counts', {}))}

### 2. 文本特征深度分析

#### 长度统计
- **平均词数**: {text_features.get('avg_word_length', 0):.2f} ± {text_features.get('std_word_length', 0):.2f}
- **中位数词数**: {text_features.get('median_word_length', 0):.1f}
- **平均字符数**: {text_features.get('avg_char_length', 0):.1f}

#### 分布特征
- **25th百分位**: {text_features.get('word_length_percentiles', {}).get('25th', 0):.1f} 词
- **75th百分位**: {text_features.get('word_length_percentiles', {}).get('75th', 0):.1f} 词
- **90th百分位**: {text_features.get('word_length_percentiles', {}).get('90th', 0):.1f} 词

#### 质量分布
- **空文本**: {text_features.get('empty_texts', 0):,} ({text_features.get('empty_texts', 0)/data_structure.get('total_samples', 1)*100:.2f}%)
- **过短文本(≤3词)**: {text_features.get('short_texts', 0):,} ({text_features.get('short_texts', 0)/data_structure.get('total_samples', 1)*100:.2f}%)
- **过长文本(≥50词)**: {text_features.get('long_texts', 0):,} ({text_features.get('long_texts', 0)/data_structure.get('total_samples', 1)*100:.2f}%)

#### 词汇统计
- **总词汇数**: {text_features.get('total_words_analyzed', 0):,}
- **唯一词汇数**: {text_features.get('unique_words', 0):,}
- **词汇多样性**: {text_features.get('unique_words', 0)/max(text_features.get('total_words_analyzed', 1), 1):.4f}

### 3. 图像特征分析 (基于 {image_features.get('sample_size', 0):,} 样本)

#### 基础统计
- **图像完整率**: {image_features.get('completion_rate', 0):.1f}%
- **平均尺寸**: {image_features.get('avg_width', 0):.0f}×{image_features.get('avg_height', 0):.0f}px
- **平均宽高比**: {image_features.get('avg_aspect_ratio', 0):.2f}

#### 尺寸分布
{self._format_size_distribution(image_features.get('size_distribution', {}))}

#### 格式分布
{self._format_format_distribution(image_features.get('format_distribution', {}))}

## 🎯 数据合成建议

### 核心策略
基于分析结果，推荐以下Data-Juicer处理策略：

1. **文本增强优先级**: 🔥🔥🔥 (平均词数较少，需要大幅增强)
2. **图像质量控制**: 🔥🔥 (质量良好，适度过滤即可)
3. **多模态对齐**: 🔥🔥🔥 (确保图文匹配度)

### 推荐配置参数
```yaml
# 基于40万数据分析的推荐配置
text_length_filter:
  min_len: {max(5, int(text_features.get('avg_word_length', 10) * 0.5))}
  max_len: {min(200, int(text_features.get('avg_word_length', 10) * 5))}

image_shape_filter:
  min_width: {max(224, int(image_features.get('avg_width', 300) * 0.6))}
  min_height: {max(224, int(image_features.get('avg_height', 300) * 0.6))}

image_text_similarity_filter:
  min_score: 0.2  # 适应当前文本简单的特点
```

### 处理优先级
1. **立即执行**: 文本丰富化 (BLIP2多模型描述生成)
2. **重点关注**: 词汇多样性提升
3. **质量控制**: 多模态对齐验证

## 📈 预期效果

### 数据增强目标
- **文本长度**: {text_features.get('avg_word_length', 0):.1f} → 15-20 词 (提升 {(18/max(text_features.get('avg_word_length', 1), 1) - 1)*100:.0f}%)
- **词汇多样性**: 当前 {text_features.get('unique_words', 0)/max(text_features.get('total_words_analyzed', 1), 1):.3f} → 目标 0.200+
- **样本质量**: 预期提升 30-50%

### 处理时间预估
- **预处理阶段**: ~2-4小时
- **BLIP2描述生成**: ~8-12小时
- **质量控制**: ~1-2小时
- **总计**: ~12-18小时

## 📋 下一步行动计划

### Phase 1: 立即执行 (今天)
1. ✅ 完成数据探索分析
2. 🔄 优化Data-Juicer配置
3. 🚀 启动数据处理流程

### Phase 2: 监控与优化 (明天)
1. 📊 监控处理进度
2. 🔍 中期质量检查
3. ⚙️ 参数调优

### Phase 3: 验证与训练 (后天)
1. ✅ 验证合成数据质量
2. 🚀 启动完整MGM训练
3. 📈 性能对比分析

---

## 📊 可视化图表

本报告包含以下可视化分析图表：
- 📈 `charts/text_length_analysis.png` - 文本长度分布分析
- 📊 `charts/word_frequency_analysis.png` - 词频分析
- 🖼️ `charts/image_analysis.png` - 图像特征分析

---
*报告由完整数据集探索器自动生成 - {current_time}*
"""

        # 保存报告
        report_path = self.output_dir / "comprehensive_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 保存JSON格式的分析结果
        json_path = self.output_dir / "analysis_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)

        print(f"✅ 综合分析报告已保存到: {report_path}")
        print(f"✅ 分析结果JSON已保存到: {json_path}")
        print(f"📊 图表目录: {self.charts_dir}")

    def _estimate_processing_time(self, total_samples):
        """估算处理时间"""
        # 基于经验估算：每1000样本约需1-2分钟
        minutes = (total_samples / 1000) * 1.5
        if minutes < 60:
            return f"约 {minutes:.0f} 分钟"
        else:
            hours = minutes / 60
            return f"约 {hours:.1f} 小时"

    def _format_field_distribution(self, field_counts):
        """格式化字段分布"""
        lines = []
        for field, count in sorted(field_counts.items()):
            lines.append(f"- **{field}**: {count:,}")
        return '\n'.join(lines)

    def _format_size_distribution(self, size_dist):
        """格式化尺寸分布"""
        total = sum(size_dist.values()) if size_dist else 1
        lines = []
        for size_type, count in size_dist.items():
            percentage = count / total * 100
            lines.append(f"- **{size_type}**: {count:,} ({percentage:.1f}%)")
        return '\n'.join(lines)

    def _format_format_distribution(self, format_dist):
        """格式化格式分布"""
        total = sum(format_dist.values()) if format_dist else 1
        lines = []
        for fmt, count in format_dist.items():
            percentage = count / total * 100
            lines.append(f"- **{fmt}**: {count:,} ({percentage:.1f}%)")
        return '\n'.join(lines)

    def run_full_exploration(self):
        """运行完整的数据探索流程"""
        print("🎯 开始完整种子数据集深度探索...")

        start_time = time.time()

        # 步骤1: 加载数据
        if not self.load_full_dataset():
            return False

        # 步骤2: 数据结构分析
        self.analyze_data_structure()

        # 步骤3: 文本特征分析
        self.analyze_text_features_comprehensive()

        # 步骤4: 图像特征分析
        self.analyze_image_features_sampled()

        # 步骤5: 生成报告
        self.generate_comprehensive_report()

        total_time = time.time() - start_time
        print(f"\n🎉 完整数据探索分析完成！总耗时: {total_time:.1f}秒")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📊 图表目录: {self.charts_dir}")

        return True

def main():
    """主函数"""
    print("🚀 启动完整种子数据集探索器...")
    print("📊 目标: 分析40万完整种子数据集")

    # 创建探索器实例 (使用5万样本进行图像分析)
    explorer = FullDatasetExplorer(sample_size=50000)

    success = explorer.run_full_exploration()

    if success:
        print("\n✅ 数据探索成功完成！")
        print("📋 下一步建议:")
        print("1. 查看分析报告: output/full_dataset_analysis/comprehensive_analysis_report.md")
        print("2. 查看可视化图表: output/full_dataset_analysis/charts/")
        print("3. 基于分析结果优化Data-Juicer配置")
        print("4. 启动大规模数据合成流程")
    else:
        print("\n❌ 数据探索失败，请检查数据文件和环境")

if __name__ == "__main__":
    main()
