#!/usr/bin/env python3

import subprocess
import sys
import os
import time

def test_attention_implementation(attn_impl, test_name):
    """测试不同的attention实现"""
    
    print(f"\n{'='*60}")
    print(f"测试 {test_name} (attn_implementation={attn_impl})")
    print(f"{'='*60}")
    
    # 修改train_mem.py使用指定的attention实现
    train_mem_path = "training/mgm/train/train_mem.py"
    
    # 读取原文件
    with open(train_mem_path, 'r') as f:
        content = f.read()
    
    # 替换attention实现
    if attn_impl == "flash_attention_2":
        new_content = content.replace(
            'train(attn_implementation="sdpa")',
            'train(attn_implementation="flash_attention_2")'
        )
    else:
        new_content = content.replace(
            'train(attn_implementation="flash_attention_2")',
            'train(attn_implementation="sdpa")'
        )
    
    # 写入修改后的文件
    with open(train_mem_path, 'w') as f:
        f.write(new_content)
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['CUDA_HOME'] = '/home/robot/lhp/miniconda3/envs/Syn0625'
    env['DS_BUILD_OPS'] = '0'
    env['DS_SKIP_CUDA_CHECK'] = '1'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 构建训练命令 - 使用最小参数快速测试
    cmd = [
        '/home/robot/lhp/miniconda3/envs/Syn0625/bin/python', 
        'training/mgm/train/train_mem.py',
        '--deepspeed', 'training/scripts/zero3.json',
        '--model_name_or_path', 'training/model_zoo/LLM/gemma/gemma-2b-it',
        '--version', 'gemma',
        '--data_path', 'training/data/finetuning_stage_1_12k/mgm_instruction_stage_1_12k.json',
        '--image_folder', 'training/data/finetuning_stage_1_12k',
        '--vision_tower', 'training/model_zoo/OpenAI/clip-vit-large-patch14-336',
        '--vision_tower_aux', 'training/model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup',
        '--pretrain_mm_mlp_adapter', '../output/training_dirs/MGM-2B-Pretrain-default/mm_projector.bin',
        '--mm_projector_type', 'mlp2x_gelu',
        '--mm_vision_select_layer', '-2',
        '--mm_use_im_start_end', 'False',
        '--mm_use_im_patch_token', 'False',
        '--image_aspect_ratio', 'pad',
        '--group_by_modality_length', 'True',
        '--image_size_aux', '768',
        '--fp16', 'True',
        '--output_dir', f'../output/training_dirs/MGM-2B-Test-{attn_impl}',
        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '1',
        '--per_device_eval_batch_size', '1',
        '--gradient_accumulation_steps', '128',
        '--evaluation_strategy', 'no',
        '--save_strategy', 'steps',
        '--save_steps', '50',
        '--save_total_limit', '1',
        '--learning_rate', '2e-5',
        '--weight_decay', '0.',
        '--warmup_ratio', '0.03',
        '--lr_scheduler_type', 'cosine',
        '--logging_steps', '1',
        '--tf32', 'True',
        '--model_max_length', '1024',
        '--gradient_checkpointing', 'True',
        '--dataloader_num_workers', '1',
        '--lazy_preprocess', 'True',
        '--report_to', 'none',
        '--max_steps', '5'  # 只运行5步进行测试
    ]
    
    print(f"运行命令: {' '.join(cmd[:3])} ...")
    print(f"Attention实现: {attn_impl}")
    
    start_time = time.time()
    
    try:
        # 运行训练命令
        result = subprocess.run(
            cmd, 
            env=env,
            capture_output=True, 
            text=True, 
            timeout=300  # 5分钟超时
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} 成功!")
            print(f"⏱️  运行时间: {duration:.2f}秒")
            
            # 检查输出中的显存使用信息
            if "Parameter Offload" in result.stdout:
                print("📊 发现参数offload信息")
            
            return True, duration, result.stdout, result.stderr
            
        else:
            print(f"❌ {test_name} 失败!")
            print(f"返回码: {result.returncode}")
            print(f"⏱️  运行时间: {duration:.2f}秒")
            
            # 显示错误信息的最后几行
            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                print("错误信息 (最后10行):")
                for line in stderr_lines[-10:]:
                    print(f"  {line}")
            
            return False, duration, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} 超时 (5分钟)")
        return False, 300, "", "Timeout"
    
    except Exception as e:
        print(f"💥 {test_name} 异常: {e}")
        return False, 0, "", str(e)

def main():
    """主测试函数"""
    
    print("🚀 FlashAttention vs SDPA 对比测试")
    print("目标: 测试哪种attention实现在MGM-2B上显存使用更优")
    
    results = {}
    
    # 测试1: SDPA (当前使用的)
    success1, time1, stdout1, stderr1 = test_attention_implementation("sdpa", "SDPA (当前)")
    results["SDPA"] = {
        "success": success1,
        "time": time1,
        "stdout": stdout1,
        "stderr": stderr1
    }
    
    # 测试2: FlashAttention 2
    success2, time2, stdout2, stderr2 = test_attention_implementation("flash_attention_2", "FlashAttention 2")
    results["FlashAttention"] = {
        "success": success2,
        "time": time2,
        "stdout": stdout2,
        "stderr": stderr2
    }
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📊 测试结果总结")
    print(f"{'='*60}")
    
    for name, result in results.items():
        status = "✅ 成功" if result["success"] else "❌ 失败"
        print(f"{name:15} | {status:8} | {result['time']:6.2f}秒")
    
    # 推荐
    if results["FlashAttention"]["success"] and results["SDPA"]["success"]:
        if results["FlashAttention"]["time"] < results["SDPA"]["time"]:
            print(f"\n🎯 推荐: FlashAttention (更快 {results['SDPA']['time'] - results['FlashAttention']['time']:.2f}秒)")
        else:
            print(f"\n🎯 推荐: SDPA (更稳定)")
    elif results["FlashAttention"]["success"]:
        print(f"\n🎯 推荐: FlashAttention (SDPA失败)")
    elif results["SDPA"]["success"]:
        print(f"\n🎯 推荐: SDPA (FlashAttention失败)")
    else:
        print(f"\n❌ 两种实现都失败了，需要进一步优化")
    
    # 恢复原始设置 (SDPA)
    test_attention_implementation("sdpa", "恢复原始设置")
    print(f"\n🔄 已恢复为SDPA设置")

if __name__ == "__main__":
    main()
