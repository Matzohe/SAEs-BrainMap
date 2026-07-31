import torch
from easydict import EasyDict
from typing import Tuple, Optional
from ..util import check_path
from ..SAEs.sae_loader import load_pretrained_autoencoder
from ..visualize.NSDVisualizer.VisualPipeline.visualize import visualize

# 将模型最相关的特征进行提取，然后进行聚类
def feature_selection(
        args: EasyDict,
        top_feature: Optional[int] = None,
        save: bool = True
    ) -> Tuple[torch.Tensor, ...]:

    """
    根据相关性矩阵，提取出每个体素最相关的top_feature特征，
    处理方法为，把每层的top feature提取出来，然后拼接后在选取一次 top feature。
    Args:
        args (EasyDict): 控制参数的集合
        top_feature Optional(int): 每个体素最相关的top_feature特征，如果输入为None，那么默认为args.simialarity.top_feature
        save bool: 是否保存结果，默认为True
    Returns:
        voxel_dictionary (torch.Tensor): 根据相关性矩阵，提取出每个体素最相关的topk特征，形状为[top_feature, v, dim]，其中v为总共的体素数量
        top_feature_correlation (torch.Tensor): 每个体素最相关的topk特征的相关性，形状为[top_feature, v]
        top_feature_id (torch.Tensor): 每个体素最相关的topk特征的索引，形状为[top_feature, v]
        top_feature_layer (torch.Tensor): 每个体素最相关的topk特征所在的模型层，形状为[top_feature, v]
    """

    all_layers = args.exp.layers
    if top_feature is None:
        top_feature = args.simialarity.top_feature
    # 保存相关的特征权重
    voxel_dictionary_list = []
    # 用来保存每一层的特征的索引
    feature_index_list = []
    # 用来保存每一层的特征的相关性
    feature_correlation_list = []

    for layer in range(all_layers):
        brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, layer, args.autoencoder.rate)
        brain_sae_similarity = torch.load(brain_sae_similarity_save_root)
        sae = load_pretrained_autoencoder(args, layer=layer)
        sae_weight = sae.encoder.weight
        # 返回每个体素的最高相关特征的相关性和索引
        topk_correlation, topk_index = torch.topk(brain_sae_similarity, k=top_feature, dim=0)
        # 提取当前层最相关的特征，返回一个[top_feature, v, dim]的tensor
        extracted_sae_weight = sae_weight[topk_index]
        feature_correlation_list.append(topk_correlation)
        feature_index_list.append(topk_index)
        voxel_dictionary_list.append(extracted_sae_weight)

    voxel_dictionary = torch.cat(voxel_dictionary_list, dim=0)
    feature_index = torch.cat(feature_index_list, dim=0)
    feature_correlation = torch.cat(feature_correlation_list, dim=0)
    # 从12层提取出来的特征中选择top feature个最相关的特征，这里的top feature index的范围为[0, top_feature * all_layers]
    _, top_feature_index = torch.topk(feature_correlation, k=top_feature, dim=0)
    # 提取出top feature对应的层，范围为[0, all_layers - 1]
    top_feature_layer = top_feature_index // top_feature
    # 根据top feature的索引，提取出top feature
    top_voxel_dictionary = voxel_dictionary[top_feature_index]
    # 根据top feature的索引，提取出top feature的相关性
    top_feature_correlation = feature_correlation[top_feature_index]
    # 根据top feature的索引，提取出top feature在原先的sae权重中的索引
    top_feature_id = feature_index[top_feature_index]
    
    if save:
        voxel_dictionary_save_root = args.similarity.voxel_dictionary_save_root.format(args.exp.model_name, args.exp.subj, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, top_feature)
        voxel_top_correlation_save_root = args.similarity.voxel_top_correlation_save_root.format(args.exp.model_name, args.exp.subj, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, top_feature)
        voxel_top_layer_save_root = args.similarity.voxel_top_layer_save_root.format(args.exp.model_name, args.exp.subj, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, top_feature)
        voxel_top_id_save_root = args.similarity.voxel_top_index_save_root.format(args.exp.model_name, args.exp.subj, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, top_feature)

        check_path(voxel_dictionary_save_root)
        check_path(voxel_top_correlation_save_root)
        check_path(voxel_top_layer_save_root)
        check_path(voxel_top_id_save_root)

        torch.save(top_voxel_dictionary, voxel_dictionary_save_root)
        torch.save(top_feature_correlation, voxel_top_correlation_save_root)
        torch.save(top_feature_layer, voxel_top_layer_save_root)
        torch.save(top_feature_id, voxel_top_id_save_root)
    
    return top_voxel_dictionary, top_feature_correlation, top_feature_id, top_feature_layer


def correlation_analysis(
        feature_correlation: torch.Tensor
    ):
    """
    查看平均和最大相关性
    Args:
        feature_correlation (torch.Tensor): top 特征和体素的相关性
    """
    mean_correlation = feature_correlation.mean()
    max_correlation = feature_correlation.max()
    print("mean correlation:", mean_correlation)
    print("max correlation:", max_correlation)

# 这里融合最高选择性层，然后进行可视化，将逐层的信息投影到大脑皮层上
# 相关信息进行保存，在config中设置相关的保存路径

def top_layer_visualization(
        args: EasyDict,
        top_feature: Optional[int] = None,
        top_feature_layer: Optional[torch.Tensor] = None,

    ):
    """
    将top feature layer进行可视化，首先对top feature layer进行求均值的操作，
    然后进行minmax normalization到0-1，最后进行可视化
    Args:
        args (EasyDict): 控制参数的集合
        top_feature Optional(int): 每个体素最相关的topk特征的个数，如果输入为None，那么默认为args.simialarity.top_feature
        top_feature_layer Optional(torch.Tensor): 每个体素最相关的topk特征所在的模型层，形状为[top_feature, v]，这里可以为None，如果为None，则从保存路径中进行加载
    """
    # 选择top几的特征层数进行可视化
    if top_feature is None:
        top_feature = args.simialarity.top_feature

    if top_feature_layer is None:
        top_feature_layer_save_root = args.similarity.voxel_top_layer_save_root.format(args.exp.model_name, args.exp.subj, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, top_feature)
        try:
            top_feature_layer = torch.load(top_feature_layer_save_root)
        except:
            _, _, _, top_feature_layer = feature_selection(args, top_feature=top_feature, save=True)

    top_feature_layer = top_feature_layer.mean(dim=0)
    top_feature_layer = (top_feature_layer - top_feature_layer.min()) / (top_feature_layer.max() - top_feature_layer.min())
    top_feature_layer = top_feature_layer.numpy()
    
    discription = args.visualize.top_layer_discription.format(args.exp.model_name, args.exp.subj, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, top_feature)
    
    visualize(args, discription, top_feature_layer)


def top_correlation_visualiation(
        args: EasyDict, 
        top_feature: Optional[int] = None,
        top_feature_correlation: Optional[torch.Tensor] = None
    ):
    """
    可视化top feature的相关性
    Args:
        args (EasyDict): 控制参数
        top_feature (Optional[int], optional): _description_. Defaults to None.
        top_feature_correlation (Optional[torch.Tensor], optional): _description_. Defaults to None.
    """

    if top_feature is None:
        top_feature = args.simialarity.top_feature

    if top_feature_correlation is None:
        try:
            top_feature_correlation_save_root = args.similarity.voxel_top_correlation_save_root.format(args.exp.model_name, args.exp.subj, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, top_feature)
            top_feature_correlation = torch.load(top_feature_correlation_save_root)
        except:
            _, top_feature_correlation, _, _ = feature_selection(args, top_feature=top_feature, save=True)

    top_feature_correlation = top_feature_correlation.numpy()

    discription = args.visualize.top_feature_correlation_discription.format(args.exp.model_name, args.exp.subj, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate, top_feature)
    visualize(args, discription, top_feature_correlation)
