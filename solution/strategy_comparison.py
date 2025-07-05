#!/usr/bin/env python3
"""
数据处理策略对比脚本
对比不同Data-Juicer策略的效果，为最终方案选择提供依据
"""

import json
import os
import time
from pathlib import Path
import subprocess
import yaml

class DataProcessingComparator:
    def __init__(self):
        self.strategies = {
            'baseline_blip2': 'solution/image_captioning.yaml',
            'advanced_processing': 'solution/advanced_data_processing.yaml',
            'quality_focused': None,  # 将动态生成
            'diversity_focused': None,  # 将动态生成
        }
        self.results = {}
        
    def create_quality_focused_strategy(self):
        """创建质量优先策略"""
        config = {
            'project_name': 'quality-focused-strategy',
            'dataset_path': 'input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl',
            'export_path': 'output/processed_data/quality_focused.jsonl',
            'np': 6,
            'text_keys': 'text',
            'image_key': 'images',
            'image_special_token': '<__dj__image>',
            'eoc_special_token': '<|__dj__eoc|>',
            'open_monitor': True,
            
            'process': [
                # 严格质量过滤
                {'text_length_filter': {
                    'min_len': 15,
                    'max_len': 200
                }},
                {'alphanumeric_filter': {
                    'tokenization': False,
                    'min_ratio': 0.7,
                    'max_ratio': 1.0
                }},
                {'character_repetition_filter': {
                    'rep_len': 3,
                    'max_ratio': 0.05
                }},
                {'special_characters_filter': {
                    'min_ratio': 0.0,
                    'max_ratio': 0.2
                }},
                {'flagged_words_filter': {
                    'lang': 'en',
                    'tokenization': False,
                    'max_ratio': 0.0
                }},
                
                # 图像质量过滤
                {'image_size_filter': {
                    'min_size': '50KB',
                    'max_size': '5MB',
                    'any_or_all': 'any'
                }},
                {'image_shape_filter': {
                    'min_width': 336,
                    'min_height': 336,
                    'max_width': 1024,
                    'max_height': 1024,
                    'any_or_all': 'any'
                }},
                {'image_aspect_ratio_filter': {
                    'min_ratio': 0.5,
                    'max_ratio': 2.0,
                    'any_or_all': 'any'
                }},
                
                # 多模态对齐
                {'image_text_similarity_filter': {
                    'hf_clip': 'openai/clip-vit-large-patch14',
                    'min_score': 0.3,
                    'max_score': 1.0
                }},
                
                # 高质量重新标注
                {'image_captioning_mapper': {
                    'hf_img2seq': 'Salesforce/blip2-flan-t5-xl',
                    'keep_original_sample': False
                }},
                
                # 去重
                {'document_minhash_deduplicator': {
                    'tokenization': 'space',
                    'window_size': 5,
                    'num_permutations': 128,
                    'jaccard_threshold': 0.85
                }}
            ]
        }
        
        config_path = 'solution/quality_focused.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.strategies['quality_focused'] = config_path
        return config_path
    
    def create_diversity_focused_strategy(self):
        """创建多样性优先策略"""
        config = {
            'project_name': 'diversity-focused-strategy',
            'dataset_path': 'input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl',
            'export_path': 'output/processed_data/diversity_focused.jsonl',
            'np': 6,
            'text_keys': 'text',
            'image_key': 'images',
            'image_special_token': '<__dj__image>',
            'eoc_special_token': '<|__dj__eoc|>',
            'open_monitor': True,
            
            'process': [
                # 宽松质量过滤
                {'text_length_filter': {
                    'min_len': 5,
                    'max_len': 300
                }},
                {'alphanumeric_filter': {
                    'tokenization': False,
                    'min_ratio': 0.4,
                    'max_ratio': 1.0
                }},
                
                # 基础图像过滤
                {'image_size_filter': {
                    'min_size': '10KB',
                    'max_size': '10MB',
                    'any_or_all': 'any'
                }},
                {'image_shape_filter': {
                    'min_width': 224,
                    'min_height': 224,
                    'any_or_all': 'any'
                }},
                
                # 多种描述生成
                {'image_captioning_mapper': {
                    'hf_img2seq': 'Salesforce/blip2-opt-2.7b',
                    'keep_original_sample': True,
                    'caption_key': 'blip2_caption'
                }},
                {'image_tagging_mapper': {
                    'hf_tagger': 'microsoft/DiT-base-finetuned-ade20k',
                    'keep_original_sample': True,
                    'tag_key': 'image_tags'
                }},
                
                # 文本增强
                {'sentence_augmentation_mapper': {
                    'aug_num': 3,
                    'keep_original_sample': True
                }},
                
                # 实体和关键词提取
                {'extract_keyword_mapper': {
                    'lang': 'en',
                    'top_k': 8,
                    'keep_original_sample': True,
                    'keyword_key': 'keywords'
                }},
                
                # 轻度去重
                {'document_minhash_deduplicator': {
                    'tokenization': 'space',
                    'window_size': 3,
                    'num_permutations': 64,
                    'jaccard_threshold': 0.7
                }}
            ]
        }
        
        config_path = 'solution/diversity_focused.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.strategies['diversity_focused'] = config_path
        return config_path
    
    def run_strategy(self, strategy_name, config_path):
        """运行单个策略"""
        print(f"\n🚀 运行策略: {strategy_name}")
        print(f"配置文件: {config_path}")
        
        start_time = time.time()
        
        try:
            # 运行Data-Juicer处理
            cmd = f"python toolkit/data-juicer/tools/process_data.py --config {config_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ 策略 {strategy_name} 执行成功")
                print(f"⏱️ 处理时间: {processing_time:.2f} 秒")
                
                # 分析输出结果
                output_stats = self.analyze_output(strategy_name, config_path)
                
                self.results[strategy_name] = {
                    'success': True,
                    'processing_time': processing_time,
                    'config_path': config_path,
                    **output_stats
                }
            else:
                print(f"❌ 策略 {strategy_name} 执行失败")
                print(f"错误信息: {result.stderr}")
                
                self.results[strategy_name] = {
                    'success': False,
                    'error': result.stderr,
                    'processing_time': processing_time
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 策略 {strategy_name} 执行超时")
            self.results[strategy_name] = {
                'success': False,
                'error': 'Timeout after 1 hour',
                'processing_time': 3600
            }
        except Exception as e:
            print(f"💥 策略 {strategy_name} 执行异常: {str(e)}")
            self.results[strategy_name] = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def analyze_output(self, strategy_name, config_path):
        """分析输出数据的统计信息"""
        try:
            # 从配置文件读取输出路径
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            output_path = config.get('export_path', '')
            
            if not os.path.exists(output_path):
                return {'output_samples': 0, 'data_quality': 'unknown'}
            
            # 统计输出样本数
            sample_count = 0
            text_lengths = []
            
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line.strip())
                            sample_count += 1
                            
                            text = item.get('text', '')
                            clean_text = text.replace('<__dj__image>', '').replace('<|__dj__eoc|>', '').strip()
                            text_lengths.append(len(clean_text.split()))
                            
                        except json.JSONDecodeError:
                            continue
            
            # 计算质量指标
            avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            
            return {
                'output_samples': sample_count,
                'avg_text_length': avg_text_length,
                'retention_rate': sample_count / 10000 * 100,  # 假设输入10K样本
                'data_quality': 'good' if avg_text_length > 10 else 'poor'
            }
            
        except Exception as e:
            print(f"分析输出时出错: {str(e)}")
            return {'output_samples': 0, 'data_quality': 'unknown'}
    
    def compare_strategies(self):
        """对比所有策略"""
        print("🔍 开始策略对比实验...")
        
        # 创建动态策略配置
        self.create_quality_focused_strategy()
        self.create_diversity_focused_strategy()
        
        # 运行所有策略
        for strategy_name, config_path in self.strategies.items():
            if config_path and os.path.exists(config_path):
                self.run_strategy(strategy_name, config_path)
            else:
                print(f"⚠️ 跳过策略 {strategy_name}: 配置文件不存在")
        
        # 生成对比报告
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """生成对比报告"""
        print("\n📊 生成策略对比报告...")
        
        report_lines = [
            "# Data-Juicer策略对比报告",
            "",
            "## 策略概览",
            "",
            "| 策略名称 | 执行状态 | 处理时间(秒) | 输出样本数 | 保留率(%) | 平均文本长度 | 数据质量 |",
            "|---------|---------|------------|-----------|----------|------------|---------|"
        ]
        
        for strategy_name, result in self.results.items():
            if result['success']:
                row = f"| {strategy_name} | ✅ 成功 | {result['processing_time']:.1f} | {result.get('output_samples', 'N/A')} | {result.get('retention_rate', 'N/A'):.1f} | {result.get('avg_text_length', 'N/A'):.1f} | {result.get('data_quality', 'N/A')} |"
            else:
                row = f"| {strategy_name} | ❌ 失败 | {result['processing_time']:.1f} | - | - | - | - |"
            
            report_lines.append(row)
        
        report_lines.extend([
            "",
            "## 推荐策略",
            "",
            "基于对比结果，推荐策略选择：",
            "",
            "### 1. 质量优先场景",
            "- 推荐: `quality_focused`",
            "- 适用: 对数据质量要求极高，可接受样本数量减少",
            "",
            "### 2. 平衡场景",
            "- 推荐: `advanced_processing`", 
            "- 适用: 质量和数量并重的标准训练",
            "",
            "### 3. 多样性优先场景",
            "- 推荐: `diversity_focused`",
            "- 适用: 需要大量多样化数据，对质量要求相对宽松",
            "",
            "## 下一步建议",
            "",
            "1. 选择最佳策略进行完整数据集处理",
            "2. 使用处理后的数据训练MGM模型",
            "3. 在TextVQA和MMBench上评估性能",
            "4. 与baseline对比，验证改进效果"
        ])
        
        # 保存报告
        os.makedirs('output/analysis', exist_ok=True)
        report_path = 'output/analysis/strategy_comparison_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ 对比报告已保存到: {report_path}")
        
        # 打印简要结果
        print("\n📈 策略对比结果:")
        for strategy_name, result in self.results.items():
            status = "✅ 成功" if result['success'] else "❌ 失败"
            samples = result.get('output_samples', 'N/A')
            print(f"  {strategy_name}: {status}, 输出样本: {samples}")

if __name__ == "__main__":
    comparator = DataProcessingComparator()
    comparator.compare_strategies()
