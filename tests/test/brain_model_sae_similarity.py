# 当前测试文件的作用是
# 提取目标模型在NSD测试数据集上的SAE激活以及neuron激活
# 从而获得saes与各个voxels的相关性并进行保存，同时打印全部层最高激活相关性

# 可直接运行的命令
# nohup python tests/test/brain_model_sae_similarity.py --exp_subj 5 --exp_model_name "clip_vit-b_16" --exp_device "cuda:0" --autoencoder_name "original" --autoencoder_rate 16 > output_similarity.log 2>&1 &
# nohup python tests/test/brain_model_sae_similarity.py --exp_subj 5 --exp_model_name "dinov2" --exp_device "cuda:1" --autoencoder_name "original" --autoencoder_rate 16 > output_similarity_1.log 2>&1 &
# nohup python tests/test/brain_model_sae_similarity.py --exp_subj 5 --exp_model_name "imagenet" --exp_device "cuda:2" --autoencoder_name "original" --autoencoder_rate 16 > output_similarity_2.log 2>&1 &
# nohup python tests/test/brain_model_sae_similarity.py --exp_subj 5 --exp_model_name "mae" --exp_device "cuda:3" --autoencoder_name "original" --autoencoder_rate 16 > output_similarity_3.log 2>&1 &

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from tests.sae.sae_brain_similarity.sae_brain_similarity import mean_similarity_analysis, neuron_mean_similarity_analysis

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
    print("current model:", args.exp.model_name)
    print("current subj:", args.exp.subj)
    model, image_preprocess = load_target_model(model_name=args.exp.model_name)
    model = model.to(device=args.exp.device)
    mean_similarity_analysis(args, model, image_preprocess)
    neuron_mean_similarity_analysis(args, model, image_preprocess)

    all_info = []
    all_neuron_info = []
    for layer in range(args.exp.layers):
        brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, layer, args.autoencoder.rate)
        brain_sae_similarity = torch.load(brain_sae_similarity_save_root)
        brain_neuron_similarity = torch.load(args.similarity.brain_neuron_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, layer))
        all_info.append(brain_sae_similarity.unsqueeze(0))
        all_neuron_info.append(brain_neuron_similarity.unsqueeze(0))
    all_info = torch.cat(all_info, dim=0)
    all_info = torch.amax(all_info, dim=(0, 1))
    all_neuron_info = torch.cat(all_neuron_info, dim=0)
    all_neuron_info = torch.amax(all_neuron_info, dim=(0, 1))
    print("SAEs max and mean similarity")
    print(all_info.max())
    print(all_info.mean())
    print("Neurons max and mean similarity")
    print(all_neuron_info.max())
    print(all_neuron_info.mean())
    print()
