#!/usr/bin/env python3
"""
快速测试脚本
用于验证Data-Juicer配置的正确性和基本功能
"""

import os
import json
import yaml
import subprocess
from pathlib import Path

def test_config_syntax():
    """测试配置文件语法"""
    print("🔍 测试配置文件语法...")
    
    config_files = [
        'solution/image_captioning.yaml',
        'solution/advanced_data_processing.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"✅ {config_file} 语法正确")
            except yaml.YAMLError as e:
                print(f"❌ {config_file} 语法错误: {e}")
                return False
        else:
            print(f"⚠️ {config_file} 不存在")
    
    return True

def test_data_availability():
    """测试数据文件可用性"""
    print("\n🔍 测试数据文件可用性...")
    
    data_file = 'input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl'
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行: bash download_10k_baseline.sh")
        return False
    
    # 检查数据格式
    try:
        sample_count = 0
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # 只检查前5行
                    break
                if line.strip():
                    item = json.loads(line.strip())
                    sample_count += 1
                    
                    # 检查必要字段
                    if 'text' not in item:
                        print(f"❌ 第{i+1}行缺少'text'字段")
                        return False
                    if 'images' not in item:
                        print(f"❌ 第{i+1}行缺少'images'字段")
                        return False
        
        print(f"✅ 数据文件格式正确，检查了{sample_count}个样本")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ 数据文件JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 数据文件检查失败: {e}")
        return False

def test_data_juicer_installation():
    """测试Data-Juicer安装"""
    print("\n🔍 测试Data-Juicer安装...")
    
    dj_tool = 'toolkit/data-juicer/tools/process_data.py'
    
    if not os.path.exists(dj_tool):
        print(f"❌ Data-Juicer工具不存在: {dj_tool}")
        print("请先运行: bash install.sh")
        return False
    
    # 测试导入
    try:
        result = subprocess.run([
            'python', dj_tool, '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Data-Juicer工具可正常运行")
            return True
        else:
            print(f"❌ Data-Juicer工具运行失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Data-Juicer工具响应超时")
        return False
    except Exception as e:
        print(f"❌ Data-Juicer工具测试失败: {e}")
        return False

def run_mini_test():
    """运行小规模测试"""
    print("\n🚀 运行小规模测试...")
    
    # 创建测试配置
    test_config = {
        'project_name': 'mini-test',
        'dataset_path': 'input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl',
        'export_path': 'output/mini_test_result.jsonl',
        'np': 2,
        'text_keys': 'text',
        'image_key': 'images',
        'image_special_token': '<__dj__image>',
        'eoc_special_token': '<|__dj__eoc|>',
        'data_probe_ratio': 0.01,  # 只处理1%的数据
        
        'process': [
            # 简单的文本长度过滤
            {
                'text_length_filter': {
                    'min_len': 5,
                    'max_len': 200
                }
            },
            # 基础图像大小过滤
            {
                'image_size_filter': {
                    'min_size': '1KB',
                    'max_size': '50MB',
                    'any_or_all': 'any'
                }
            }
        ]
    }
    
    # 保存测试配置
    test_config_path = 'solution/mini_test.yaml'
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    print(f"📝 创建测试配置: {test_config_path}")
    
    # 运行测试
    try:
        cmd = f"python toolkit/data-juicer/tools/process_data.py --config {test_config_path}"
        print(f"🔄 执行命令: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 小规模测试成功")
            
            # 检查输出
            output_path = test_config['export_path']
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    output_lines = f.readlines()
                print(f"📊 输出样本数: {len(output_lines)}")
                
                # 清理测试文件
                os.remove(output_path)
                os.remove(test_config_path)
                
                return True
            else:
                print(f"❌ 输出文件未生成: {output_path}")
                return False
        else:
            print(f"❌ 小规模测试失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 小规模测试超时")
        return False
    except Exception as e:
        print(f"❌ 小规模测试异常: {e}")
        return False

def test_image_access():
    """测试图像文件访问"""
    print("\n🔍 测试图像文件访问...")
    
    data_file = 'input/pretrain_stage_1_10k/mgm_pretrain_stage_1_10k.jsonl'
    
    if not os.path.exists(data_file):
        print("❌ 数据文件不存在，跳过图像测试")
        return False
    
    try:
        image_count = 0
        missing_count = 0
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:  # 只检查前10个样本
                    break
                    
                if line.strip():
                    item = json.loads(line.strip())
                    if 'images' in item and item['images']:
                        image_path = item['images'][0]
                        image_count += 1
                        
                        if not os.path.exists(image_path):
                            missing_count += 1
        
        if image_count > 0:
            success_rate = (image_count - missing_count) / image_count * 100
            print(f"📊 图像文件检查: {image_count}个图像，{missing_count}个缺失")
            print(f"✅ 图像可访问率: {success_rate:.1f}%")
            return success_rate > 80  # 80%以上认为正常
        else:
            print("❌ 未找到图像文件")
            return False
            
    except Exception as e:
        print(f"❌ 图像文件检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 Data-Juicer配置快速测试")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    tests = [
        ("配置文件语法", test_config_syntax),
        ("数据文件可用性", test_data_availability),
        ("Data-Juicer安装", test_data_juicer_installation),
        ("图像文件访问", test_image_access),
        ("小规模功能测试", run_mini_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print(f"\n{'='*50}")
    print("🏁 测试结果总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 通过率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！可以开始正式实验。")
        print("\n下一步建议:")
        print("1. 运行策略对比: python solution/strategy_comparison.py")
        print("2. 执行完整实验: python solution/run_experiment.py")
    else:
        print("\n⚠️ 部分测试失败，请检查环境配置。")
        print("\n故障排除建议:")
        print("1. 确保已运行: bash install.sh")
        print("2. 确保已运行: bash download_10k_baseline.sh")
        print("3. 检查Python环境和依赖")

if __name__ == "__main__":
    main()
