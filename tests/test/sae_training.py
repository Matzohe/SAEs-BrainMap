# 这个函数是用来训练全部层的SAEs的

# 可直接运行的命令
# nohup python tests/test/sae_training.py --exp_subj 5 --exp_model_name "clip_vit-b_16" --exp_device "cuda:0" --autoencoder_name "original" --autoencoder_rate 16 > output.log 2>&1 &

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import toml
import argparse
from easydict import EasyDict
from src.SAEs.trainer.MultiLayerVisualTrainer import MultiLayerVisualTrainer
from src.util import get_info_from_shell

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 这段是统一的参数
    parser.add_argument("--exp_model_name", type=str, help="模型名称", default="clip_vit-b_16")
    parser.add_argument("--exp_device", type=str, help="设备", default="cuda:0")
    parser.add_argument("--autoencoder_name", type=str, help="SAE名称", default="original")
    parser.add_argument("--autoencoder_rate", type=int, help="SAE扩大的倍数", default=16)
    parser.add_argument("--exp_layers", type=int, help="模型全部层", default=12)
    parser.add_argument("--autoencoder_tied", type=str, help="SAE是否共享权重", default=True)
    parser.add_argument("--autoencoder_batch_size", type=int, help="SAE训练的批次大小", default=512)
    parser.add_argument("--autoencoder_epoch", type=int, help="SAE训练的轮数", default=5)
    parser.add_argument("--autoencoder_l1_weight", type=float, help="训练中l1权重", default=0.00086)
    parser.add_argument("--autoencoder_lr", type=float, help="训练中学习率", default=5e-5)
    parser.add_argument("--autoencoder_topk", type=int, help="模型的topk特征，当用topk方法时需要", default=512)


    arg_parser = parser.parse_args()
    config_dict = toml.load('config.toml')
    args = EasyDict(config_dict)
    args = get_info_from_shell(arg_parser, args)
    target_layer_list = [i for i in range(args.exp.layers)]
    MultiLayerVisualTrainer(args, target_layer_list)
