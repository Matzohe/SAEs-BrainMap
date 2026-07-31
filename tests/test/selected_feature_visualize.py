# 这个文件是用来可视化特定脑区提取出来的特征，可以调整可视化特征的数量，是否独立以及热力图的颜色

# 可直接运行的命令
# nohup python tests/test/selected_feature_visualize.py --exp_subj 5 --exp_model_name "clip_vit-b_16" --exp_device "cuda:0" --autoencoder_name "original" --autoencoder_rate 16 > output.log 2>&1 &

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import toml
from easydict import EasyDict
from tests.sae.sae_brain_similarity.brain_guide_circuit import all_layer_feature_visualize
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
    # 下面是当前方法的特定参数
    parser.add_argument("--roi_name", type=str, help="脑区名称")
    parser.add_argument("--visualize_topk", type=int, help="可视化的前几特征的数量", default=100)
    parser.add_argument("--heatmap", type=str, help="热图颜色", default="jet")
    parser.add_argument("--save_independently", type=bool, help="是否单独保存每层的可视化结果", default=True)

    arg_parser = parser.parse_args()
    config_dict = toml.load('config.toml')
    args = EasyDict(config_dict)
    args = get_info_from_shell(arg_parser, args)
    target_layers = [i for i in range(args.exp.layers)]
    # 将当前最相关的特征进行可视化
    if arg_parser.roi_name is None:
        for roi_name in ["FFA", "EBA", "RSC", "VWFA", "FOOD"]:
            all_layer_feature_visualize(args=args, 
                                        roi_name=roi_name, 
                                        subj=args.exp.subj, 
                                        all_layers=args.exp.layers,
                                        topk=arg_parser.visualize_topk,
                                        save_independently=arg_parser.save_independently,
                                        heatmap=arg_parser.heatmap,
                                        )
    else:
        all_layer_feature_visualize(args=args, 
                                        roi_name=arg_parser.roi_name, 
                                        subj=args.exp.subj, 
                                        all_layers=args.exp.layers,
                                        topk=arg_parser.visualize_topk,
                                        save_independently=arg_parser.save_independently,
                                        heatmap=arg_parser.heatmap,
                                        )
    

