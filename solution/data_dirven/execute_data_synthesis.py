#!/usr/bin/env python3
"""
数据合成执行脚本 - Phase 1: 数据合成计划
基于LoRA训练成功的基础上，执行高质量数据合成
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/data_synthesis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataSynthesisExecutor:
    def __init__(self):
        self.project_root = Path.cwd()
        self.data_juicer_path = self.project_root / "toolkit" / "data-juicer"
        self.config_path = self.project_root / "solution" / "basic_data_synthesis.yaml"
        self.output_dir = self.project_root / "output" / "processed_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_prerequisites(self):
        """检查执行前提条件"""
        logger.info("🔍 检查执行前提条件...")

        # 检查完整种子数据 (40万数据)
        seed_data = "input/pretrain_stage_1/mgm_pretrain_stage_1.jsonl"
        if not os.path.exists(seed_data):
            logger.error(f"❌ 完整种子数据文件不存在: {seed_data}")
            return False
        logger.info(f"✅ 完整种子数据文件存在: {seed_data}")

        # 检查数据大小
        file_size = os.path.getsize(seed_data) / (1024**3)  # GB
        logger.info(f"📊 数据文件大小: {file_size:.2f}GB")
        
        # 检查Data-Juicer
        if not self.data_juicer_path.exists():
            logger.error(f"❌ Data-Juicer路径不存在: {self.data_juicer_path}")
            return False
        logger.info(f"✅ Data-Juicer路径存在: {self.data_juicer_path}")
        
        # 检查配置文件
        if not self.config_path.exists():
            logger.error(f"❌ 配置文件不存在: {self.config_path}")
            return False
        logger.info(f"✅ 配置文件存在: {self.config_path}")
        
        # 检查系统资源
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"📊 系统内存: {memory_gb:.1f}GB (可用: {available_gb:.1f}GB)")

        if available_gb < 40:
            logger.warning("⚠️ 可用内存可能不足，处理40万数据建议至少40GB可用内存")

        # 预估处理时间
        logger.info("⏱️ 预估处理时间: 10-12小时 (40万数据)")
        logger.info("🎯 目标: 文本长度 8.78→15-20词, 词汇多样性 0.0714→0.200+")

        return True
    
    def execute_data_processing(self):
        """执行数据处理"""
        logger.info("🚀 开始执行Data-Juicer数据处理...")
        
        # 构建命令 - 使用conda环境的Python
        conda_python = "/home/robot/lhp/miniconda3/envs/Syn0625/bin/python"
        cmd = [
            conda_python,  # 使用conda环境Python
            "-m", "data_juicer.tools.process_data",
            "--config", str(self.config_path),
            "--work_dir", str(self.output_dir)
        ]
        
        logger.info(f"📝 执行命令: {' '.join(cmd)}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 切换到data-juicer目录
            os.chdir(self.data_juicer_path)
            
            # 执行处理 (40万数据需要更长时间)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=43200  # 12小时超时 (40万数据)
            )
            
            # 记录结束时间
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ 数据处理成功完成，耗时: {duration:.1f}秒")
                logger.info("📊 处理输出:")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ 数据处理失败，返回码: {result.returncode}")
                logger.error("错误输出:")
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ 数据处理超时（12小时）")
            return False
        except Exception as e:
            logger.error(f"❌ 数据处理异常: {e}")
            return False
        finally:
            # 切换回原目录
            os.chdir(self.project_root)
    
    def validate_output(self):
        """验证输出数据质量"""
        logger.info("🔍 验证输出数据质量...")
        
        output_file = self.output_dir / "basic_enhanced_data.jsonl"
        if not output_file.exists():
            logger.error(f"❌ 输出文件不存在: {output_file}")
            return False
        
        # 统计输出数据
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                sample_count = len(lines)
            
            logger.info(f"📊 输出样本数: {sample_count}")
            
            # 分析前几个样本
            if sample_count > 0:
                with open(output_file, 'r', encoding='utf-8') as f:
                    sample = json.loads(f.readline())
                    
                logger.info("📝 样本字段:")
                for key in sample.keys():
                    logger.info(f"  - {key}")
                
                # 检查文本长度改善
                if 'enhanced_text' in sample:
                    enhanced_length = len(sample['enhanced_text'].split())
                    original_length = len(sample.get('text', '').split())
                    logger.info(f"📈 文本长度改善: {original_length} → {enhanced_length} 词")
                
            return sample_count > 0
            
        except Exception as e:
            logger.error(f"❌ 验证输出数据时出错: {e}")
            return False
    
    def generate_summary_report(self):
        """生成总结报告"""
        logger.info("📝 生成数据合成总结报告...")
        
        report_path = self.output_dir / "synthesis_report.md"
        
        # 收集统计信息
        input_file = "input/pretrain_stage_1/mgm_pretrain_stage_1.jsonl"
        output_file = self.output_dir / "basic_enhanced_data.jsonl"
        
        input_count = 0
        output_count = 0
        
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                input_count = len(f.readlines())
        
        if output_file.exists():
            with open(output_file, 'r') as f:
                output_count = len(f.readlines())
        
        # 生成报告
        report_content = f"""# 数据合成执行报告

## 执行信息
- **执行时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **配置文件**: {self.config_path.name}
- **处理策略**: 基于数据探索结果的定制化策略

## 数据统计
- **输入样本数**: {input_count:,}
- **输出样本数**: {output_count:,}
- **数据增长率**: {(output_count/input_count*100-100):.1f}% (如果>0表示数据增强成功)

## 处理策略
1. **文本丰富化**: 使用BLIP2多模型生成详细描述
2. **词汇多样化**: 多角度图像标签生成
3. **内容增强**: 文本融合和问答生成
4. **质量控制**: 多层过滤确保数据质量

## 下一步计划
1. 验证合成数据质量
2. 准备完整MGM训练
3. 对比训练效果

---
*报告由数据合成执行器自动生成*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✅ 报告已保存到: {report_path}")
    
    def run_full_synthesis(self):
        """运行完整的数据合成流程"""
        logger.info("🎯 开始Phase 1: 数据合成计划执行")
        
        # 步骤1: 检查前提条件
        if not self.check_prerequisites():
            logger.error("❌ 前提条件检查失败，终止执行")
            return False
        
        # 步骤2: 执行数据处理
        if not self.execute_data_processing():
            logger.error("❌ 数据处理失败，终止执行")
            return False
        
        # 步骤3: 验证输出
        if not self.validate_output():
            logger.error("❌ 输出验证失败")
            return False
        
        # 步骤4: 生成报告
        self.generate_summary_report()
        
        logger.info("🎉 Phase 1: 数据合成计划执行完成！")
        logger.info("📋 下一步: 准备完整MGM训练")
        
        return True

def main():
    """主函数"""
    print("🚀 启动数据合成执行器...")
    
    executor = DataSynthesisExecutor()
    success = executor.run_full_synthesis()
    
    if success:
        print("\n✅ 数据合成成功完成！")
        print("📁 输出目录: output/processed_data/")
        print("📝 查看报告: output/processed_data/synthesis_report.md")
        print("📊 查看日志: output/data_synthesis.log")
    else:
        print("\n❌ 数据合成失败，请检查日志")
        sys.exit(1)

if __name__ == "__main__":
    main()
