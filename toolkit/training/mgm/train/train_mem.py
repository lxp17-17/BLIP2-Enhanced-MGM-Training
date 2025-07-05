import sys
import os

# 添加训练目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(current_dir, '../../')
sys.path.insert(0, training_dir)

from mgm.train.train import train

if __name__ == "__main__":
    train(attn_implementation="sdpa")