import torch
import numpy as np
import os
import shutil
from easydict import EasyDict
from ..sae_diffusion_evaluate import diffusionEvaluation
from src.util import check_path

# 将cluster的特征进行提取，然后使用生成式模型来进行可视化，同时附上其最高选择性的图像
# 分为inner和cross layer cluster
def inner_cluster_visualize(
        args: EasyDict, 
        roi_name: str, 
        subj: int, 
        topk: int = 100, 
        target_layer: int = 11,
        save_independantly: bool = False, 
    ):
    """
    从先前的inner cluster中，提取出对应的信息，然后使用diffusion进行可视化，默认是进行group可视化操作

    Args:
        args (EasyDict): 流程控制超参数
        roi_name (str): 指导对应的ROI
        subj (int): 被试id
        topk (int, optional): 每层选择多少最相关的特征. Defaults to 100.
        target_layer (int, optional): 目标层. Defaults to 11.
        save_independantly (bool): 是否独立进行可视化. Defaults to False.
    """
    model_name = args.exp.model_name
    sae_name = args.autoencoder.name
    sae_rate = args.autoencoder.rate
    cluster_save_root = args.similarity.inner_cluster_info_save_root.format(subj, roi_name, model_name, sae_name, sae_rate, target_layer, topk)
    cluster = torch.load(cluster_save_root)
    cluster_info = {}
    cluster_info['subj'] = subj
    cluster_info['roi'] = roi_name
    cluster_info['inner_cluster'] = True
    for cluster_id in cluster.keys():
        cluster_info['cluster_id'] = cluster_id
        feature_index = cluster[cluster_id].int().tolist()
        diffusionEvaluation(args=args, 
                            layer=target_layer,
                            weight_id = feature_index,
                            all_image_number=9,
                            image_per_batch=9,
                            cluster_visualize=True,
                            cluster_info=cluster_info,
                            save_indepentandly=save_independantly,
                        )


def cross_layer_cluster_visualie(
        args: EasyDict,  
        roi_name: str, 
        subj: int, 
        topk: int = 100, 
        all_layers: int = 12, 
        save_independantly: bool = False, 
    ):
    """
    从先前的cross layer cluster中，提取出对应的信息，然后使用diffusion进行可视化，默认是进行group可视化操作

    Args:
        args (EasyDict): 流程控制超参数
        roi_name (str): 指导对应的ROI
        subj (int): 被试id
        topk (int, optional): 每层选择多少最相关的特征. Defaults to 100.
        all_layers (int, optional): 模型总共有多少层. Defaults to 12.
        save_independantly (bool): 是否独立进行可视化. Defaults to False.
    """
    model_name = args.exp.model_name
    sae_name = args.autoencoder.name
    sae_rate = args.autoencoder.rate
    cluster_save_root = args.similarity.cross_layer_cluster_info_save_root.format(subj, roi_name, model_name, sae_name, sae_rate, topk)
    cluster = torch.load(cluster_save_root)
    cluster_info = {}
    cluster_info['subj'] = subj
    cluster_info['roi'] = roi_name
    cluster_info['inner_cluster'] = False
    for cluster_id in cluster.keys():
        cluster_info['cluster_id'] = cluster_id
        feature_index = []
        feature_layer = []
        for each in cluster[cluster_id]:
            layer, feature_id = each.split("_")
            feature_index.append(int(feature_id))
            feature_layer.append(int(layer))
        feature_index = torch.tensor(feature_index, dtype=torch.int32)
        feature_layer = torch.tensor(feature_layer, dtype=torch.int32)
        for layer in range(all_layers):
            mask = feature_layer == layer
            if mask.sum() == 0:
                continue
            current_feature_index = feature_index[mask].int().tolist()
            diffusionEvaluation(args=args, 
                                layer=layer,
                                weight_id = current_feature_index,
                                all_image_number=9,
                                image_per_batch=9,
                                cluster_visualize=True,
                                cluster_info=cluster_info,
                                save_indepentandly=save_independantly,
                            )
            
# 将同一个回路中的信息进行提取于保存，目标在于整理出相关的信息
def inner_circuit_cluster_collection(
        args: EasyDict, 
        roi_name: str, 
        subj: int, 
        topk: int = 100, 
        target_layer: int = 11,
    ):
    """
    将先前回路分析获得的特征挑出来，将同一个簇中的特征放入同一个文件夹中

    Args:
        args (EasyDict): 流程控制超参数
        roi_name (str): 指导对应的ROI
        subj (int): 被试id
        topk (int, optional): 每层选择多少最相关的特征. Defaults to 100.
        target_layer (int, optional): 目标层. Defaults to 11.
    """
    model_name = args.exp.model_name
    sae_name = args.autoencoder.name
    sae_rate = args.autoencoder.rate
    cluster_info_save_root = args.similarity.inner_cluster_info_save_root.format(subj, roi_name, model_name, sae_name, sae_rate, target_layer, topk)
    feature_selectivity_save_root = args.similarity.roi_selected_feature_heatmap_save_root.format(subj, roi_name, model_name, sae_name, sae_rate, target_layer, 0, 0)
    feature_selectivity_save_root = "/".join(feature_selectivity_save_root.split("/")[:-1])
    image_root_list = os.listdir(feature_selectivity_save_root)
    cluster_info = torch.load(cluster_info_save_root, weights_only=False)
    for key, value in cluster_info.items():
        for id in value.tolist():
            name = [s for s in image_root_list if "_{}.png".format(id) in s]
            image_root = feature_selectivity_save_root + "/" + name[0]
            destination_root = args.similarity.inner_cluster_feature_heatmap_save_root.format(subj, model_name, sae_name, sae_rate, roi_name, target_layer, key, id)
            check_path(destination_root)
            shutil.copyfile(image_root, destination_root)


def cross_layer_circuit_cluster_collection(
        args: EasyDict, 
        roi_name: str, 
        subj: int, 
        topk: int = 100, 
        all_layers: int = 12, 
    ):
    """
    将跨层回路分析获得的特征挑出来，将同一个簇中的特征放入同一个文件夹中

    Args:
        args (EasyDict): 流程控制超参数
        roi_name (str): 指导对应的ROI
        subj (int): 被试id
        topk (int, optional): 每层选择多少最相关的特征. Defaults to 100.
        all_layers (int, optional): 模型总共有多少层. Defaults to 12.
    """
    model_name = args.exp.model_name
    sae_name = args.autoencoder.name
    sae_rate = args.autoencoder.rate
    cluster_info_save_root = args.similarity.cross_layer_cluster_info_save_root.format(subj, roi_name, model_name, sae_name, sae_rate, topk)
    all_layer_feature_information = []
    for target_layer in range(all_layers):
        feature_selectivity_save_root = args.similarity.roi_selected_feature_heatmap_save_root.format(subj, roi_name, model_name, sae_name, sae_rate, target_layer, 0, 0)
        feature_selectivity_save_root = "/".join(feature_selectivity_save_root.split("/")[:-1])
        image_root_list = os.listdir(feature_selectivity_save_root)
        all_layer_feature_information.append(image_root_list)
    cluster_info = torch.load(cluster_info_save_root, weights_only=False)
    for key, value in cluster_info.items():
        for info in value:
            target_layer, id = info.split("_tensor(")
            id = id.split(")")[0]
            name = [s for s in all_layer_feature_information[int(target_layer)] if "_{}.png".format(id) in s]
            feature_selectivity_save_root = args.similarity.roi_selected_feature_heatmap_save_root.format(subj, roi_name, model_name, sae_name, sae_rate, target_layer, 0, 0)
            image_root = "/".join(feature_selectivity_save_root.split("/")[:-1]) + "/" + name[0]
            destination_root = args.similarity.cross_layer_cluster_feature_heatmap_save_root.format(subj, roi_name, model_name, sae_name, sae_rate, key, target_layer, id)
            check_path(destination_root)
            shutil.copyfile(image_root, destination_root)
