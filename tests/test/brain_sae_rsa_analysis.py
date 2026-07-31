# 可直接运行的命令
# nohup python tests/test/brain_sae_rsa_analysis.py --exp_subj 5 --exp_model_name "clip_vit-b_16" --exp_device "cuda:0" --autoencoder_name "original" --autoencoder_rate 16 > output_rsa.log 2>&1 &
# nohup python tests/test/brain_sae_rsa_analysis.py --exp_subj 5 --exp_model_name "dinov2" --exp_device "cuda:1" --autoencoder_name "original" --autoencoder_rate 16 > output_rsa_1.log 2>&1 &
# nohup python tests/test/brain_sae_rsa_analysis.py --exp_subj 5 --exp_model_name "imagenet" --exp_device "cuda:2" --autoencoder_name "original" --autoencoder_rate 16 > output_rsa_2.log 2>&1 &
# nohup python tests/test/brain_sae_rsa_analysis.py --exp_subj 5 --exp_model_name "mae" --exp_device "cuda:3" --autoencoder_name "original" --autoencoder_rate 16 > output_rsa_3.log 2>&1 &

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.sae_brain_correlation.brain_sae_rsa import voxel_dictionary_rsa_selection

import toml
import torch
from easydict import EasyDict
from src.models.load_target_model import load_target_model
import argparse
from src.util import get_info_from_shell

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 这段是统一的参数
    parser.add_argument("--exp_subj", type=int, help="被试号", required=True)
    parser.add_argument("--exp_model_name", type=str, help="模型名称", required=True)
    parser.add_argument("--exp_device", type=str, help="设备", required=True)
    parser.add_argument("--autoencoder_name", type=str, help="SAE名称", required=True)
    parser.add_argument("--autoencoder_rate", type=int, help="SAE扩大的倍数", required=True)
    parser.add_argument("--exp_layers", type=int, help="模型全部层")
    parser.add_argument("--autoencoder_tied", type=str, help="SAE是否共享权重")
    parser.add_argument("--autoencoder_topk", type=int, help="模型的topk特征，当用topk方法时需要")

    arg_parser = parser.parse_args()
    args = EasyDict(toml.load("config.toml"))
    args = get_info_from_shell(arg_parser, args)

    voxel_dictionary_rsa_selection(args, args.exp.layers)