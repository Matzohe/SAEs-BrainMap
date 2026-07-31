# 这个函数的目标，是根据特征-脑区相关性，将特征放到大脑皮层上，为每个特征选择最相关的体素
# 然后，将每个体素被选择的次数作为密度，可视化其热力图

import torch
from easydict import EasyDict
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from ..util import check_path
from ..visualize.NSDVisualizer.VisualPipeline.visualize import visualize
from ..dataset.NSD.NSD_utils import load_target_roi_mask

def sae_mapping(
        args: EasyDict, 
        layer: int, 
    ):
    """
    可视化目标层，所有saes特征与大脑最相关的分布，并进行可视化

    Args:
        args (EasyDict): 控制参数
        layer (int): 目标层
    """
    brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, layer, args.autoencoder.rate)
    sae_similarity = torch.load(brain_sae_similarity_save_root)
    target_mapping = np.zeros(shape=(sae_similarity.shape[-1], ))
    _, mapping_index = torch.max(sae_similarity, dim=-1)
    for index in mapping_index:
        target_mapping[index] += 1.
    uniq_value = np.unique(target_mapping)
    uniq_value.sort()
    values = np.linspace(0, 1, len(uniq_value))
    for i in range(len(uniq_value)):
        target_mapping = np.where(target_mapping == uniq_value[i], values[i], target_mapping)

    sae_mapping_discription = "subj{}/{}_{}/{}_{}/sae_mapping/layer{}".format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, layer) 
    visualize(args, sae_mapping_discription, target_mapping)

def all_layer_sae_mapping(
        args: EasyDict, 
        all_layers: int = 12, 
        roi_list: List[str] = ['FFA', 'EBA', 'RSC', 'VWFA', 'FOOD', 'v1', 'v2', 'v3', 'hv4']
    ):
    """
    对于所有层，进行saes特征与大脑最相关的分布计算，同时可视化
    与此同时，将每个脑区的逐层特征数量分布进行统计，看逐层特征选择情况变化。

    Args:
        args (EasyDict): _description_
        all_layers (int, optional): _description_. Defaults to 12.
        roi_list (List[str], optional): _description_. Defaults to ['FFA', 'EBA', 'RSC', 'VWFA', 'FOOD', 'v1', 'v2', 'v3', 'hv4'].
    """
    # 用来保存，每个roi中，每层的特征选择数量
    roi_distribution_list = [[] for _ in range(len(roi_list))]
    all_target_mapping = []
    for layer in range(all_layers):
        brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, layer, args.autoencoder.rate)
        sae_similarity = torch.load(brain_sae_similarity_save_root)
        target_mapping = np.zeros(shape=(sae_similarity.shape[-1], ))
        _, mapping_index = torch.max(sae_similarity, dim=-1)
        for index in mapping_index:
            target_mapping[index] += 1.
        
        for j, roi_name in enumerate(roi_list):
            roi_root = args.NSD.roi_index.format(args.exp.subj, roi_name)
            roi_mask = load_target_roi_mask(roi_root)
            roi_distribution_list[j].append(target_mapping[roi_mask].sum())
        all_target_mapping.append(target_mapping)
    all_target_mapping = np.sum(all_target_mapping, axis=0)
    
    uniq_value = np.unique(all_target_mapping)
    uniq_value.sort()
    values = np.linspace(0, 1, len(uniq_value))
    for i in range(len(uniq_value)):
        all_target_mapping = np.where(all_target_mapping == uniq_value[i], values[i], all_target_mapping)


    # 为每个ROI打印柱状图并进行保存
    for j, roi_name in enumerate(roi_list):
        roi_distribution = np.array(roi_distribution_list[j])
        # roi_distribution = roi_distribution / roi_distribution.max()
        barfig_save_root = "experiments/paper_image/sae_mapping_bar_plot/{}/{}/{}/{}_{}_bar_plot.png".format(args.exp.subj, args.exp.model_name, roi_name, args.autoencoder.name, layer, args.autoencoder.rate)
        plt.figure(figsize=(12,6))
        plt.bar(range(all_layers), roi_distribution)
        plt.title("{} roi distribution".format(roi_name))
        plt.xlabel("layers")
        plt.ylabel("Frequency")
        check_path(barfig_save_root)
        plt.savefig(barfig_save_root)

    sae_mapping_discription = "subj{}/{}_{}/{}_{}/sae_mapping/all_layers_sae_mapping".format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate) 
    visualize(args, sae_mapping_discription, all_target_mapping)
