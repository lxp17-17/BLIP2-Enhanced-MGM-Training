#!/usr/bin/env python3
"""
MMBench评估运行脚本
支持LoRA模型和标准模型的MMBench评估，并生成对比报告
"""

import os
import sys
import subprocess
import json
import argparse
from datetime import datetime

def run_command(cmd, cwd=None):
    """执行命令并返回结果"""
    print(f"🔄 执行命令: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 命令执行失败: {result.stderr}")
            return False, result.stderr
        else:
            print(f"✅ 命令执行成功")
            return True, result.stdout
    except Exception as e:
        print(f"❌ 命令执行异常: {e}")
        return False, str(e)

def check_model_exists(model_path):
    """检查模型是否存在"""
    if os.path.exists(model_path):
        print(f"✅ 模型存在: {model_path}")
        return True
    else:
        print(f"❌ 模型不存在: {model_path}")
        return False

def check_dataset_exists(dataset_path):
    """检查数据集是否存在"""
    if os.path.exists(dataset_path):
        print(f"✅ 数据集存在: {dataset_path}")
        return True
    else:
        print(f"❌ 数据集不存在: {dataset_path}")
        return False

def run_mmbench_evaluation(model_name, split="mmbench_dev_20230712", gpu_id="0", is_lora=True):
    """运行MMBench评估"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../output/training_dirs", model_name)
    dataset_path = os.path.join(script_dir, "training/data/eval_stage_1/mmbench", f"{split}.tsv")
    
    # 检查模型和数据集
    if not check_model_exists(model_path):
        return False, f"模型不存在: {model_path}"
    
    if not check_dataset_exists(dataset_path):
        return False, f"数据集不存在: {dataset_path}"
    
    # 选择评估脚本
    if is_lora:
        eval_script = os.path.join(script_dir, "eval/mmbench_lora.sh")
        print(f"🎯 使用LoRA评估脚本: {eval_script}")
    else:
        eval_script = os.path.join(script_dir, "eval/mmbench.sh")
        print(f"🎯 使用标准评估脚本: {eval_script}")
    
    if not os.path.exists(eval_script):
        return False, f"评估脚本不存在: {eval_script}"
    
    # 构建评估命令
    cmd = f"bash {eval_script} {model_name} {split} {gpu_id}"
    
    print(f"🚀 开始MMBench评估")
    print(f"📁 模型: {model_name}")
    print(f"📊 数据集: {split}")
    print(f"🖥️  GPU: {gpu_id}")
    print(f"🔧 LoRA模式: {is_lora}")
    
    # 执行评估
    success, output = run_command(cmd, cwd=script_dir)
    
    if success:
        # 检查结果文件
        result_file = os.path.join(script_dir, "../output/eval_results", model_name, "mmbench/answers", split, f"{model_name}.jsonl")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                result_count = sum(1 for _ in f)
            print(f"✅ MMBench评估完成: {result_count} 个结果")
            return True, result_file
        else:
            print(f"⚠️  评估完成但结果文件不存在: {result_file}")
            return False, "结果文件不存在"
    else:
        return False, output

def analyze_mmbench_results(result_file):
    """分析MMBench评估结果"""
    if not os.path.exists(result_file):
        return None
    
    print(f"📊 分析MMBench结果: {result_file}")
    
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    total_questions = len(results)
    
    # 简单统计（这里可以根据需要扩展更详细的分析）
    analysis = {
        "total_questions": total_questions,
        "model_responses": len([r for r in results if r.get('text', '').strip()]),
        "avg_response_length": sum(len(r.get('text', '')) for r in results) / total_questions if total_questions > 0 else 0,
        "sample_responses": results[:3] if len(results) >= 3 else results
    }
    
    print(f"📈 分析结果:")
    print(f"  - 总问题数: {analysis['total_questions']}")
    print(f"  - 有效响应: {analysis['model_responses']}")
    print(f"  - 平均响应长度: {analysis['avg_response_length']:.1f} 字符")
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="MMBench评估运行脚本")
    parser.add_argument("--models", nargs="+", required=True, help="要评估的模型名称列表")
    parser.add_argument("--split", default="mmbench_dev_20230712", help="数据集分割")
    parser.add_argument("--gpu", default="0", help="GPU ID")
    parser.add_argument("--lora", action="store_true", help="使用LoRA评估模式")
    parser.add_argument("--output-report", help="输出对比报告路径")
    
    args = parser.parse_args()
    
    print("🎯 MMBench评估任务开始")
    print(f"📋 模型列表: {args.models}")
    print(f"📊 数据集: {args.split}")
    print(f"🖥️  GPU: {args.gpu}")
    print(f"🔧 LoRA模式: {args.lora}")
    
    evaluation_results = {}
    
    # 逐个评估模型
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"🔄 评估模型: {model_name}")
        print(f"{'='*60}")
        
        success, result = run_mmbench_evaluation(
            model_name=model_name,
            split=args.split,
            gpu_id=args.gpu,
            is_lora=args.lora
        )
        
        if success:
            # 分析结果
            analysis = analyze_mmbench_results(result)
            evaluation_results[model_name] = {
                "success": True,
                "result_file": result,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
        else:
            evaluation_results[model_name] = {
                "success": False,
                "error": result,
                "timestamp": datetime.now().isoformat()
            }
    
    # 生成对比报告
    print(f"\n{'='*60}")
    print("📊 生成评估报告")
    print(f"{'='*60}")
    
    report = {
        "evaluation_config": {
            "models": args.models,
            "split": args.split,
            "gpu": args.gpu,
            "lora_mode": args.lora,
            "timestamp": datetime.now().isoformat()
        },
        "results": evaluation_results
    }
    
    # 保存报告
    if args.output_report:
        report_path = args.output_report
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(script_dir, "../output/eval_results", f"mmbench_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 评估报告已保存: {report_path}")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("📈 评估总结")
    print(f"{'='*60}")
    
    successful_models = [name for name, result in evaluation_results.items() if result["success"]]
    failed_models = [name for name, result in evaluation_results.items() if not result["success"]]
    
    print(f"✅ 成功评估: {len(successful_models)} 个模型")
    for model in successful_models:
        analysis = evaluation_results[model]["analysis"]
        if analysis:
            print(f"  - {model}: {analysis['total_questions']} 问题, {analysis['model_responses']} 响应")
    
    if failed_models:
        print(f"❌ 评估失败: {len(failed_models)} 个模型")
        for model in failed_models:
            print(f"  - {model}: {evaluation_results[model]['error']}")
    
    print(f"\n🎉 MMBench评估任务完成！")
    return len(successful_models) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
