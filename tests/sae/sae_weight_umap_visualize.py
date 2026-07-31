import torch
from easydict import EasyDict
from typing import Union, List, Optional
from src.visualize.umap import UMAPVisualize
from src.SAEs.sae_loader import load_pretrained_autoencoder
from src.visualize.pca import PCAVisualize


def visualizeSAEsWeight(
        args: EasyDict,
        sae_name_list: Union[str, List[str]],
        target_layer: int,
    ):

    colors = [
        (1.0, 0.0, 0.0),   # 红色
        (0.0, 1.0, 0.0),   # 绿色
        (0.0, 0.0, 1.0),   # 蓝色
        (1.0, 1.0, 0.0),   # 黄色
        (0.0, 1.0, 1.0),   # 青色
        (1.0, 0.0, 1.0),   # 品红
        (0.5, 0.5, 0.5),   # 中灰
        (0.9, 0.3, 0.2),   # 橘红色
        (0.2, 0.6, 0.9),   # 天蓝色
        (0.3, 0.8, 0.4),   # 草绿色
    ]

    all_weight = []
    weight_number_list = []

    if isinstance(sae_name_list, str):
        sae_name_list = [sae_name_list]
    else:
        sae_name_list = sae_name_list

    model_name = args.exp.model_name

    if model_name == "dinov2":
        args.autoencoder.rate = 4
        args.autoencoder.topk = 308
    elif model_name == "clip_vit-b_16":
        args.autoencoder.rate = 16
        args.autoencoder.topk = 512
    else:
        raise NotImplementedError

    for sae_name in sae_name_list:
        args.autoencoder.name = sae_name
        if sae_name == "original":
            args.autoencoder.tied = "True"
        else:
            args.autoencoder.tied = "False"
        sae = load_pretrained_autoencoder(args, layer=target_layer)

        decoder_weight = sae.encoder.weight.detach().cpu()
        if decoder_weight.shape[0] < args.autoencoder.topk * 5:
            decoder_weight = decoder_weight.T

        all_weight.append(decoder_weight)
        weight_number_list.append(decoder_weight.shape[0])

    all_weight = torch.cat(all_weight, dim=0)
    color_list = []
    for i, weight_number in enumerate(weight_number_list):
        color_list.extend([colors[i] for _ in range(weight_number)])

    UMAPVisualize(data=all_weight, n_neighbors=100, min_dist=0.7, show=True, save=True, save_path="{}_umap.png".format(model_name), color_list=color_list, n_components=2)
    PCAVisualize(all_weight)
