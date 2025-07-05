#!/usr/bin/env python3

import subprocess
import sys
import os

def run_finetune_test():
    """Run a minimal fine-tuning test with conservative parameters"""
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    cmd = [
        'deepspeed', 'mgm/train/train_mem.py',
        '--deepspeed', './scripts/zero2.json',  # Use zero2.json instead of zero2_offload.json
        '--model_name_or_path', 'model_zoo/LLM/gemma/gemma-2b-it',
        '--version', 'gemma',
        '--data_path', './data/finetuning_stage_1_12k/mgm_instruction_stage_1_12k.json',
        '--image_folder', './data/finetuning_stage_1_12k',
        '--vision_tower', 'model_zoo/OpenAI/clip-vit-large-patch14-336',
        '--vision_tower_aux', 'model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup',
        '--pretrain_mm_mlp_adapter', '../output/training_dirs/MGM-2B-Pretrain-default/mm_projector.bin',
        '--mm_projector_type', 'mlp2x_gelu',
        '--mm_vision_select_layer', '-2',
        '--mm_use_im_start_end', 'False',
        '--mm_use_im_patch_token', 'False',
        '--image_aspect_ratio', 'pad',
        '--group_by_modality_length', 'True',
        '--image_size_aux', '768',
        '--bf16', 'True',
        '--output_dir', '../output/training_dirs/MGM-2B-Finetune-minimal',
        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '1',  # Very small batch size
        '--per_device_eval_batch_size', '1',
        '--gradient_accumulation_steps', '16',  # Smaller accumulation
        '--evaluation_strategy', 'no',
        '--save_strategy', 'steps',
        '--save_steps', '50',  # Save more frequently
        '--save_total_limit', '1',
        '--learning_rate', '2e-5',
        '--weight_decay', '0.',
        '--warmup_ratio', '0.03',
        '--lr_scheduler_type', 'cosine',
        '--logging_steps', '1',
        '--tf32', 'True',
        '--model_max_length', '2048',
        '--gradient_checkpointing', 'True',
        '--dataloader_num_workers', '1',  # Minimal workers
        '--lazy_preprocess', 'True',
        '--report_to', 'none',
        '--max_steps', '10'  # Only run 10 steps for testing
    ]
    
    print("Running minimal fine-tuning test...")
    print("Command:", ' '.join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS: Fine-tuning test completed!")
        print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("ERROR: Fine-tuning test failed!")
        print("Return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

if __name__ == "__main__":
    success = run_finetune_test()
    sys.exit(0 if success else 1)
