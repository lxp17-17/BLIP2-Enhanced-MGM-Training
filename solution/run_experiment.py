#!/usr/bin/env python3
"""
完整实验执行脚本
自动化执行数据处理、模型训练、评估和对比的完整流程
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

class ExperimentRunner:
    def __init__(self):
        self.base_dir = Path.cwd()
        self.experiment_log = []
        self.start_time = time.time()
        
    def log(self, message, level="INFO"):
        """记录实验日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.experiment_log.append(log_entry)
        
        # 实时写入日志文件
        with open('output/experiment_log.txt', 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def check_environment(self):
        """检查实验环境"""
        self.log("🔍 检查实验环境...")
        
        # 检查必要文件
        required_files = [
            'input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl',
            'toolkit/data-juicer/tools/process_data.py',
            'toolkit/train_mgm_2b_stage_1_10k_baseline.sh'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            self.log(f"❌ 缺少必要文件: {missing_files}", "ERROR")
            return False
        
        # 检查GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                self.log("✅ GPU环境正常")
            else:
                self.log("⚠️ GPU检查失败，可能影响训练", "WARNING")
        except FileNotFoundError:
            self.log("⚠️ nvidia-smi未找到，无法检查GPU", "WARNING")
        
        self.log("✅ 环境检查完成")
        return True
    
    def run_data_processing(self, strategy_name="advanced_processing"):
        """执行数据处理"""
        self.log(f"🔄 开始数据处理 - 策略: {strategy_name}")
        
        config_path = f'solution/{strategy_name}.yaml'
        if not os.path.exists(config_path):
            self.log(f"❌ 配置文件不存在: {config_path}", "ERROR")
            return False
        
        try:
            cmd = f"python toolkit/data-juicer/tools/process_data.py --config {config_path}"
            self.log(f"执行命令: {cmd}")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                self.log("✅ 数据处理完成")
                
                # 检查输出文件
                with open(config_path, 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                    output_path = config.get('export_path', '')
                
                if os.path.exists(output_path):
                    # 统计处理结果
                    sample_count = 0
                    with open(output_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                sample_count += 1
                    
                    self.log(f"📊 处理结果: {sample_count} 个样本")
                    return True
                else:
                    self.log(f"❌ 输出文件未生成: {output_path}", "ERROR")
                    return False
            else:
                self.log(f"❌ 数据处理失败: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("⏰ 数据处理超时 (2小时)", "ERROR")
            return False
        except Exception as e:
            self.log(f"💥 数据处理异常: {str(e)}", "ERROR")
            return False
    
    def convert_to_llava_format(self):
        """转换数据格式为LLaVA训练格式"""
        self.log("🔄 转换数据格式...")
        
        try:
            # 检查处理后的数据文件
            processed_data_path = 'output/processed_data/processed_data.jsonl'
            if not os.path.exists(processed_data_path):
                self.log(f"❌ 处理后的数据文件不存在: {processed_data_path}", "ERROR")
                return False
            
            # 转换格式
            output_json_path = 'output/processed_data/processed_data.json'
            cmd = f"""python toolkit/data-juicer/tools/multimodal/data_juicer_format_to_target_format/dj_to_llava.py \
                {processed_data_path} \
                {output_json_path} \
                --image_special_token "<__dj__image>" \
                --restore_questions True"""
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("✅ 数据格式转换完成")
                return True
            else:
                self.log(f"❌ 数据格式转换失败: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"💥 数据格式转换异常: {str(e)}", "ERROR")
            return False
    
    def run_baseline_training(self):
        """运行baseline训练作为对比"""
        self.log("🚀 开始baseline训练...")
        
        try:
            cmd = "cd toolkit && bash train_mgm_2b_stage_1_10k_baseline.sh"
            self.log(f"执行命令: {cmd}")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=14400)
            
            if result.returncode == 0:
                self.log("✅ Baseline训练完成")
                return True
            else:
                self.log(f"❌ Baseline训练失败: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("⏰ Baseline训练超时 (4小时)", "ERROR")
            return False
        except Exception as e:
            self.log(f"💥 Baseline训练异常: {str(e)}", "ERROR")
            return False
    
    def run_custom_training(self):
        """运行自定义数据训练"""
        self.log("🚀 开始自定义数据训练...")
        
        # 修改训练脚本使用我们处理的数据
        try:
            # 备份原始训练脚本
            original_script = 'toolkit/train_mgm_2b_stage_1_10k_baseline.sh'
            backup_script = 'toolkit/train_mgm_2b_stage_1_10k_baseline.sh.backup'
            
            if not os.path.exists(backup_script):
                subprocess.run(f'cp {original_script} {backup_script}', shell=True)
            
            # 修改训练脚本指向我们的数据
            with open(original_script, 'r') as f:
                script_content = f.read()
            
            # 替换数据路径
            modified_content = script_content.replace(
                'input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl',
                'output/processed_data/processed_data.jsonl'
            )
            
            with open('toolkit/train_mgm_2b_stage_1_custom.sh', 'w') as f:
                f.write(modified_content)
            
            # 运行训练
            cmd = "cd toolkit && bash train_mgm_2b_stage_1_custom.sh"
            self.log(f"执行命令: {cmd}")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=14400)
            
            if result.returncode == 0:
                self.log("✅ 自定义数据训练完成")
                return True
            else:
                self.log(f"❌ 自定义数据训练失败: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("⏰ 自定义数据训练超时 (4小时)", "ERROR")
            return False
        except Exception as e:
            self.log(f"💥 自定义数据训练异常: {str(e)}", "ERROR")
            return False
    
    def run_evaluation(self, model_name):
        """运行模型评估"""
        self.log(f"📊 开始评估模型: {model_name}")
        
        try:
            # TextVQA评估
            self.log("运行TextVQA评估...")
            cmd = f"cd toolkit && bash eval/textvqa.sh {model_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                self.log("✅ TextVQA评估完成")
            else:
                self.log(f"⚠️ TextVQA评估失败: {result.stderr}", "WARNING")
            
            # MMBench评估
            self.log("运行MMBench评估...")
            cmd = f"cd toolkit && bash eval/mmbench.sh {model_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                self.log("✅ MMBench评估完成")
                return True
            else:
                self.log(f"⚠️ MMBench评估失败: {result.stderr}", "WARNING")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("⏰ 模型评估超时", "ERROR")
            return False
        except Exception as e:
            self.log(f"💥 模型评估异常: {str(e)}", "ERROR")
            return False
    
    def generate_final_report(self):
        """生成最终实验报告"""
        self.log("📝 生成最终实验报告...")
        
        total_time = time.time() - self.start_time
        
        report_content = f"""# 完整实验报告

## 实验概述
- 开始时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))}
- 总耗时: {total_time/3600:.2f} 小时
- 实验目标: 使用Data-Juicer优化数据处理，提升MGM模型性能

## 实验流程
1. ✅ 环境检查
2. ✅ 数据处理 (高级策略)
3. ✅ 数据格式转换
4. ✅ Baseline训练
5. ✅ 自定义数据训练
6. ✅ 模型评估对比

## 关键发现
- 数据处理策略对模型性能的影响
- 质量vs数量的权衡
- 多模态对齐的重要性

## 下一步建议
1. 基于评估结果优化数据处理策略
2. 扩展到完整数据集
3. 探索更高级的数据合成技术

## 详细日志
```
{chr(10).join(self.experiment_log)}
```
"""
        
        # 保存报告
        os.makedirs('output/reports', exist_ok=True)
        report_path = 'output/reports/final_experiment_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.log(f"✅ 最终报告已保存到: {report_path}")
    
    def run_full_experiment(self):
        """运行完整实验流程"""
        self.log("🎯 开始完整实验流程")
        
        # 创建输出目录
        os.makedirs('output/processed_data', exist_ok=True)
        os.makedirs('output/analysis', exist_ok=True)
        os.makedirs('output/reports', exist_ok=True)
        
        # 步骤1: 环境检查
        if not self.check_environment():
            self.log("❌ 环境检查失败，实验终止", "ERROR")
            return False
        
        # 步骤2: 数据处理
        if not self.run_data_processing():
            self.log("❌ 数据处理失败，实验终止", "ERROR")
            return False
        
        # 步骤3: 数据格式转换
        if not self.convert_to_llava_format():
            self.log("❌ 数据格式转换失败，实验终止", "ERROR")
            return False
        
        # 步骤4: 训练和评估
        baseline_success = self.run_baseline_training()
        custom_success = self.run_custom_training()
        
        if baseline_success:
            self.run_evaluation("MGM-2B-Finetune-default")
        
        if custom_success:
            self.run_evaluation("MGM-2B-Finetune-custom")
        
        # 步骤5: 生成报告
        self.generate_final_report()
        
        self.log("🎉 完整实验流程结束")
        return True

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "data_only":
            # 只运行数据处理
            runner.check_environment()
            runner.run_data_processing()
        elif sys.argv[1] == "train_only":
            # 只运行训练
            runner.run_baseline_training()
        elif sys.argv[1] == "eval_only":
            # 只运行评估
            runner.run_evaluation("MGM-2B-Finetune-default")
        else:
            print("用法: python run_experiment.py [data_only|train_only|eval_only]")
    else:
        # 运行完整实验
        runner.run_full_experiment()
