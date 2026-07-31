import torch
import numpy as np
import h5py
import os
import umap
import matplotlib.pyplot as plt
from random import shuffle
from typing import Tuple, List, Optional, Dict, Union
from torch.utils.data import DataLoader
from easydict import EasyDict
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from tests.sae.sae_brain_similarity.brain_selected_sae import get_target_roi_correlation, visualize_selected_sae_feature, load_target_roi_mask
from src.dataset.Coco.CocoNSDAnalysis import AnalysisDataset
from src.models.Vision import clip
from src.SAEs.sae_loader import load_pretrained_autoencoder
from src.models.load_target_model import load_target_model
from src.util import check_path

# 可视化不同类别特征时使用的
colors = [
    (228/255, 26/255, 28/255),     # 1 Red
    (55/255, 126/255, 184/255),    # 2 Blue
    (77/255, 175/255, 74/255),     # 3 Green
    (255/255, 127/255, 0/255),     # 4 Orange
    (152/255, 78/255, 163/255),    # 5 Purple
    (255/255, 255/255, 51/255),    # 6 Yellow
    (0/255, 191/255, 196/255),     # 7 Cyan
    (255/255, 105/255, 180/255),   # 8 Pink
    (166/255, 86/255, 40/255),     # 9 Brown
    (0/255, 128/255, 128/255),     # 10 Teal
    (212/255, 175/255, 55/255),    # 11 Gold
    (75/255, 0/255, 130/255),      # 12 Indigo
    (191/255, 255/255, 0/255),     # 13 Lime
    (255/255, 0/255, 255/255),     # 14 Magenta
    (0/255, 154/255, 205/255),     # 15 Deep Sky
    (255/255, 114/255, 86/255),    # 16 Coral
    (102/255, 255/255, 204/255),   # 17 Aquamarine
    (128/255, 0/255, 0/255),       # 18 Maroon
    (34/255, 139/255, 34/255),     # 19 Forest
    (255/255, 140/255, 0/255)      # 20 Dark Orange
]



def all_layer_feature_extraction(
        args: EasyDict, 
        roi_name: str,
        subj: int, 
        all_layers: int = 12, 
        topk: int = 100, 
    ) -> torch.Tensor:

    """
    这个函数是提取出，在指定ROI指导下，选出的所有模型层的最相关的特征
    返回的是一个tensor，形状为(层数，特征编号)，其中特征编号是按照相关性进行降序排序的

    Args:
        args (EasyDict): 模型的全部参数
        roi_name (str): 想要指导的roi名称
        subj (int): 被试名称
        all_layers (int, optional): 模型总共有多少层. Defaults to 12.
        topk (int, optional): 每一层选择多少个特征. Defaults to 100.
    
    Return:
        torch.Tensor: 返回的是一个tensor，形状为(层数，特征编号),其中特征编号是按照相关性进行降序排序的
    """

    all_layer_feature_index = []

    for layer in range(all_layers):
        target_layer_correlation = get_target_roi_correlation(args=args, roi_name=roi_name, target_layer=layer, subj=subj)
        _, most_correlate_feature_index = torch.topk(target_layer_correlation.mean(dim=-1).view(1, -1), k=topk, dim=-1)
        all_layer_feature_index.append(most_correlate_feature_index.view(1, -1))
    
    all_layer_feature_index = torch.cat(all_layer_feature_index, dim=0)

    return all_layer_feature_index


# 这个函数提供了可视化all layer feature extraction提取的特征的渠道
# 因为在brain selected sae中计算完相关性后没有进行保存就直接可视化了
# 这里提供另一个通道
def all_layer_feature_visualize(
        args: EasyDict, 
        roi_name: str,
        subj: int, 
        all_layers: int = 12, 
        topk: int = 100,
        save_independently: bool = True,
        heatmap: str = "jet",
    ):
    feature_index_save_root = args.similarity.roi_selected_feature_index_save_root.format(subj, roi_name, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate, topk)
    if not os.path.exists(feature_index_save_root):
        all_layer_sae_feature_index = all_layer_feature_extraction(args=args, roi_name=roi_name, subj=subj, all_layers=all_layers, topk=topk)
        check_path(feature_index_save_root)
        torch.save(all_layer_sae_feature_index, feature_index_save_root)
    else:
        all_layer_sae_feature_index = torch.load(feature_index_save_root)
    for target_layer in range(all_layers):
        visualize_selected_sae_feature(args, subj=subj, roi_name=roi_name, layer=target_layer, feature_index=all_layer_sae_feature_index[target_layer], roi_level=True, heatmap=heatmap, save_independently=save_independently)


def selected_sae_feature_activation_analysis(
        args: EasyDict,
        roi_name: str,
        subj: int,
        all_layers: int = 12,
        topk: int = 100,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    这个函数用于提取出，和指定ROI最相关的特征在ImageNet Test上的激活，并返回选择出的特征激活，以及选择的特征的index。
    首先提取出和当前roi最相关的所有层的特征index，然后基于此index，进行激活的选择

    Args:
        args (EasyDict): 模型的全部参数
        roi_name (str): 想要指导的roi名称
        subj (int): 被试名称
        all_layers (int, optional): 模型总共有多少层. Defaults to 12.
        topk (int, optional): 每一层选择多少个特征. Defaults to 100.
    Return:
        Tuple[List[torch.Tensor], torch.Tensor]: 返回的是选择的特征的激活，以及选择的特征的index，注意特征激活维度在倒数第二维，最后一维为特征数量
    """
   
    device = args.exp.device

    # 导入提取的模型
    model_name = args.exp.model_name
    target_model, image_preprocess = load_target_model(args.exp.model_name)
    target_model = target_model.to(device=device).eval()

    # 导入逐层训练好的saes
    saes = []
    for layer in range(all_layers):
        sae = load_pretrained_autoencoder(args, layer=layer)
        sae = sae.to(device=device).eval()
        saes.append(sae)

    # 提前保存好的token的保存路径
    ImageNetTestTokenSavePath = args.SAEsEvaluation.imagenet_test_token_save_root
    
    # 导入sae相关的选择好的特征
    feature_index_save_root = args.similarity.roi_selected_feature_index_save_root.format(subj, roi_name, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate, topk)
    if not os.path.exists(feature_index_save_root):
        all_layer_sae_feature_index = all_layer_feature_extraction(args=args, roi_name=roi_name, subj=subj, all_layers=all_layers, topk=topk)
        check_path(feature_index_save_root)
        torch.save(all_layer_sae_feature_index, feature_index_save_root)
    else:
        all_layer_sae_feature_index = torch.load(feature_index_save_root)
    
    # 提取选择出来特征的激活
    selected_feature_activation_save_root = args.similarity.roi_selected_feature_activation_save_root.format(subj, roi_name, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate, topk)
    if os.path.exists(selected_feature_activation_save_root):
        selected_feature_activation = torch.load(selected_feature_activation_save_root)
        return selected_feature_activation, all_layer_sae_feature_index
    else:
        selected_feature_activation = [[] for _ in range(all_layers)]
        with torch.no_grad():
            target_layer = [i for i in range(all_layers)]
            for batch in tqdm(range(98), desc="Top Activation Extraction", total=98):
                with h5py.File(ImageNetTestTokenSavePath.format(model_name, batch), "r") as f:
                    evaluating_data = torch.from_numpy(f['token embedding'][target_layer, :, 1:, :]).to(device=device)  # (all_layer, 1024, 196, 768)
                    f.close()
                for i, layer in enumerate(target_layer):
                    sae = saes[i]
                    feature_index = all_layer_sae_feature_index[i].squeeze(0)
                    activation, _ = sae.encode(evaluating_data[i].squeeze(0))
                    activation = activation.mean(dim=1)
                    activation = activation[:, feature_index]
                    selected_feature_activation[i].append(activation.cpu())
                del evaluating_data

        selected_feature_activation = [torch.cat(i, dim=0) for i in selected_feature_activation]
        check_path(selected_feature_activation_save_root)
        torch.save(selected_feature_activation, selected_feature_activation_save_root)
        return selected_feature_activation, all_layer_sae_feature_index


def jaccard_similarity(
        mat_1: torch.Tensor, 
        mat_2: Optional[torch.Tensor] = None, 
        threshold: Optional[Union[float, torch.Tensor]] = 0.1,
    ) -> torch.Tensor:
    """
    通过jaccard相似度计算两个特征矩阵的相似度，首先将矩阵转换为0-1矩阵，然后计算相似度
    返回的是特征之间的两两相似度，使用jaccard distance进行衡量
    Args:
        mat_1 (torch.Tensor): 矩阵1
        mat_2 Optional(torch.Tensor, None): 矩阵2. Defaults to None, 如果为None的话，将mat_1进行copy为mat_2
        threshold (torch.Tensor or float): 阈值，这里期望输入每个tensor的激活均值，默认阈值为0.1

    Returns:
        torch.Tensor: jaccard相似度
    """
    if mat_2 is None:
        mat_2 = mat_1.clone()
    if isinstance(threshold, torch.Tensor):
        assert threshold.shape[0] == mat_1.shape[0]
        threshold = threshold.unsqueeze(1)
    mat_1 = (mat_1 > threshold).to(torch.float32)
    mat_2 = (mat_2 > threshold).to(torch.float32)
    intersection = mat_1 @ mat_2.T
    ones1 = mat_1.sum(dim=1, keepdim=True)
    ones2 = mat_2.sum(dim=1, keepdim=True).T
    union = (ones1 + ones2 - intersection + 1e-8)
    return 1 - intersection / union

def get_cluster_id(
        emb2d: np.ndarray, 
        feature_index: Optional[Union[torch.Tensor, List[str]]], 
        inner_cluster_visualize: bool = False
    ) -> Union[Dict, Tuple[Dict, np.ndarray]]:
    """
    使用DBSCAN对降维后的的特征进行聚类，返回一个字典文件
    Args:
        emb2d (np.ndarray): 降维后的特征, 
        feature_index (Optional[torch.Tensor, List[str]]): 特征的index，单层情况下是tensor，多层情况下是List
        inner_cluster_visualize (bool, optional): 是否返回聚类id. Defaults to False.
    Return:
        Union[Dict, Tuple[Dict, np.ndarray]]: 返回聚类结果, 如果return_cluster_id为True的话，返回聚类结果和聚类id
    """
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(emb2d)
    cluster_id = dbscan.labels_
    clusters = {}
    for id in np.unique(cluster_id):
        if id == -1:
            continue
        if isinstance(feature_index, torch.Tensor):
            clusters[id] = feature_index[np.where(cluster_id == id)[0].tolist()]
        elif isinstance(feature_index, list):
            if id not in clusters.keys():
                clusters[id] = []
            for each in np.where(cluster_id == id)[0].tolist():
                clusters[id].append(feature_index[each])
        else:
            raise NotImplementedError
    if not inner_cluster_visualize:
        return clusters
    else:
        return clusters, cluster_id

def inner_cluster(
        feature_matrix: torch.Tensor,
        feature_index: torch.Tensor, 
        threshold: Optional[Union[float, torch.Tensor]] = None, 
        n_neighbors: int = 5,
        min_dist: float = 0.1,
        dict_save_path: str = "umap_result.pt", 
        fig_save_path: str = "inner_umap_cluster.png", 
        visualize: bool = True, 
        inner_cluster_visualize: bool = False, 
        cluster_filter: bool = False, 
        cluster_filter_topk: int = 10,         
    ):
    """
    对单层内的特征进行聚类，同时可视化聚类的结果
    每个特征之间的距离使用jaccard来进行衡量
    对于簇内聚类的结果，使用UMAP来进行可视化
    最后将聚类之后的特征index进行保存
    Args:
        feature_matrix (torch.Tensor): 特征矩阵
        feature_index (torch.Tensor): 按照相关性逆序排序的sae特征index
        n_neighbors (int, optional): UMAP的n_neighbors参数. Defaults to 5.
        min_dist (float, optional): UMAP的min_dist参数. Defaults to None.
        fig_save_path (str, optional): 图片保存路径. Defaults to "inner_umap_cluster.png".
        visualize (bool, optional): 是否进行可视化. Defaults to True.
        inner_cluster_visualize (bool, optional): 是否进行簇内类间的可视化. Defaults to False.
    """
    if threshold is None:
        threshold = feature_matrix.mean(dim=-1).view(-1)
    jaccard_distance = jaccard_similarity(feature_matrix, feature_matrix, threshold=threshold)
    print(jaccard_distance.mean())
    reducer = umap.UMAP(
        metric='precomputed',
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
    )

    emb2d = reducer.fit_transform(jaccard_distance)
    cluster_info = get_cluster_id(emb2d, feature_index, inner_cluster_visualize=inner_cluster_visualize)
    if inner_cluster_visualize:
        cluster, cluster_id = cluster_info
        color = []
        for id in cluster_id:
            if id == -1:
                color.append((0.6, 0.6, 0.6))
            else:
                color.append(colors[id % len(colors)])
    else:
        cluster = cluster_info
        color = [(0.6, 0.6, 0.6)]
    
    if cluster_filter:
        selected_feature_name_list = []
        selected_cluster = {}
        for each in feature_index[:cluster_filter_topk]:
                selected_feature_name_list.append(each)
        for index in cluster.keys():
            for each in cluster[index]:
                if each in selected_feature_name_list:
                    selected_cluster[index] = cluster[index]
                    break
        cluster = selected_cluster
    check_path(dict_save_path)
    torch.save(cluster, dict_save_path)

    if visualize:
        check_path(fig_save_path)
        plt.scatter(emb2d[:, 0], emb2d[:, 1], s=2, c=color)
        plt.title('UMAP cluster visualize')
        plt.savefig(fig_save_path)
        plt.close()

def inner_circuit_analysis(
        args: EasyDict, 
        roi_name: str, 
        subj: int, 
        all_layers: int = 12, 
        topk: int = 100, 
        threshold: Optional[Union[float, torch.Tensor]] = None,
        n_neighbors: int = 5,
        min_dist: float = 0.1,
        visualize: bool = False,
        inner_cluster_visualize: bool = False, 
        cluster_filter: bool = False, 
        cluster_filter_topk: int = 10, 
    ):
    """
    提取单层内，同时激活情况强的特征，并进行聚类降维以及可视化

    Args:
        args (EasyDict): 实验参数文件
        roi_name (str): 分析针对哪一个ROI
        subj (int): 被试编码
        all_layers (int, optional): 模型总共有多少层. Defaults to 12.
        topk (int, optional): 每一层选择多少个最相关的特征. Defaults to 100.
        n_neighbors (int, optional): UMAP的n_neighbors参数. Defaults to 5.
        threshold (Optional[float], optional): jaccard相似度阈值. Defaults to None.
        min_dist (float, optional): UMAP的min_dist参数. Defaults to 0.1.
        visualize (bool, optional): 是否可视化. Defaults to False.
        inner_cluster_visualize (bool, optional): 是否进行簇内类间的可视化，即为每个簇赋予颜色，同时在聚类的时候，会返回每个特征的类标签. Defaults to False.
    """
    selected_feature_activation, all_layer_feature_index = selected_sae_feature_activation_analysis(args=args, roi_name=roi_name, subj=subj, all_layers=all_layers, topk=topk)
    image_save_path = args.similarity.inner_cluster_image_save_root
    info_save_path = args.similarity.inner_cluster_info_save_root
    model_name = args.exp.model_name
    sae_name = args.autoencoder.name
    sae_rate = args.autoencoder.rate
    print("start inner cluster analysis")
    for layer in range(all_layers):
        inner_cluster(selected_feature_activation[layer].T, 
                      feature_index = all_layer_feature_index[layer].view(-1), 
                      dict_save_path=info_save_path.format(subj, roi_name, model_name, sae_name, sae_rate, layer, topk), 
                      fig_save_path=image_save_path.format(subj, roi_name, model_name, sae_name, sae_rate, layer, topk), 
                      visualize=visualize, 
                      threshold=threshold,
                      n_neighbors=n_neighbors, 
                      min_dist=min_dist, 
                      inner_cluster_visualize=inner_cluster_visualize,
                      cluster_filter=cluster_filter, 
                      cluster_filter_topk=cluster_filter_topk, 
                      )
    print("finish inner cluster analysis")

def cross_layer_cluster(
        feature_matrix: List[torch.Tensor], 
        feature_index: torch.Tensor, 
        threshold: Optional[Union[float, torch.Tensor]] = None,
        n_neighbors: int = 3,
        min_dist: float = 0.1,
        dict_save_path: str = "cross_layer_umap_result.pt", 
        fig_save_path: str = "cross_layer_umap_cluster.png", 
        visualize: bool = False, 
        inner_cluster_visualize: bool = True, 
        cluster_filter: bool = False, 
        cluster_filter_topk: int = 10, 
    ):
    """
    对于跨层的SAEs，找到跨层之间的激活表征相关的特征
    首先计算多层之间的距离，其次利用距离来进行Umap降维
    最后通过聚类，找到跨层之间的相关性强的特征簇。

    Args:
        feature_matrix (List[torch.Tensor]): 多层的特征激活情况，这里的输入是一个列表，为layers个[feature_activation_dim, feature_num]的tensor
        feature_index (torch.Tensor): 多层的特征index，输入为二维，为[layers, feature_num]
        n_neighbors (int, optional): UMAP的n_neighbors参数. Defaults to 3.
        threshold (Optional[float], optional): jaccard相似度阈值. Defaults to None.
        min_dist (float, optional): UMAP的min_dist参数. Defaults to 0.1.
        dict_save_path (str, optional): 保存聚类结果的路径. Defaults to "cross_layer_umap_result.pt".
        fig_save_path (str, optional): 保存聚类结果图像的路径. Defaults to "cross_layer_umap_cluster.png".
        visualize (bool, optional): 是否可视化. Defaults to True.
        inner_cluster_visualize (bool, optional): 是否可视化簇内的聚类结果，是的话，get_cluster_id会返回所有特征对应的cluster id. Defaults to False.
    """
    all_feature_matrix = torch.cat(feature_matrix, dim=-1).T
    all_layers = feature_index.shape[0]
    # 要为每个index加入一个前缀，将其变为str类型
    str_feature_index = []
    for layer in range(all_layers):
        for each in feature_index[layer]:
            str_feature_index.append(str(layer) + "_" + str(each))
    if threshold is None:
        threshold = all_feature_matrix.mean(dim=-1).view(-1)
    jaccard_distance = jaccard_similarity(all_feature_matrix, all_feature_matrix, threshold=threshold)
    reducer = umap.UMAP(
        metric='precomputed',
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42,
    )

    emb2d = reducer.fit_transform(jaccard_distance)
    if inner_cluster_visualize:
        cluster, cluster_id = get_cluster_id(emb2d, str_feature_index, inner_cluster_visualize=inner_cluster_visualize)
        cross_layer_circuit_estimate(
                all_feature_matrix=all_feature_matrix, 
                feature_index=feature_index, 
                cluster_id = cluster_id, 
                threshold=threshold, 
            )
    else:
        cluster = get_cluster_id(emb2d, str_feature_index, inner_cluster_visualize=inner_cluster_visualize)
    # 在完成聚类之后，需要进行一定的筛选，选择出最相关的那些回路特征。首先要衡量的是，这些特征的co-activated rate是多少
    if cluster_filter:
        selected_feature_name_list = []
        selected_cluster = {}
        for layer in range(all_layers):
            for each in feature_index[layer][:cluster_filter_topk]:
                selected_feature_name_list.append(str(layer) + "_" + str(each))
        for index in cluster.keys():
            for each in cluster[index]:
                if each in selected_feature_name_list:
                    selected_cluster[index] = cluster[index]
                    break
        cluster = selected_cluster
    check_path(dict_save_path)
    torch.save(cluster, dict_save_path)

    if visualize:
        check_path(fig_save_path)
        colors = plt.cm.tab20(np.linspace(0, 1, all_layers))[:, :3]
        colors_list = []
        for i in range(all_layers):
            colors_list.extend([colors[i] for _ in range(feature_index[i].shape[-1])])
        plt.scatter(emb2d[:, 0], emb2d[:, 1], s=2, c=colors_list)
        plt.title('UMAP cluster visualize')
        plt.savefig(fig_save_path)
        plt.close()

def inner_cluster_based_cross_layer_cluster(
        feature_matrix: List[torch.Tensor], 
        feature_index: torch.Tensor, 
        all_layers: int = 12, 

    ):
    """
    这个函数的跨层回路的分析，并不是将所有回路的激活都放在一起
    而是计算两个跨层的feature matrix。

    Args:
        feature_matrix (List[torch.Tensor]): _description_
        feature_index (torch.Tensor): _description_
        all_layers (int, optional): _description_. Defaults to 12.
    """

def cross_layer_circuit_estimate(
        all_feature_matrix: torch.Tensor, 
        feature_index: torch.Tensor, 
        cluster_id: np.ndarray, 
        threshold: Optional[Union[float, torch.Tensor]] = None, 
    ):
    # 这个函数是对初始筛选出来的聚类进行进一步的评估
    # 首先评估这些特征的co-activate的情况，同时看这些特征是不是在一个inner cluster内的
    # 其次是看这些特征和对应脑区的相关性，去除相关性弱的相应聚类
    all_cluster_ids = np.unique(cluster_id)
    for id in all_cluster_ids:
        if id == -1:
            continue
        cluster_feature_activation_matrix = all_feature_matrix[np.where(cluster_id == id)[0].tolist()]
        if isinstance(threshold, torch.Tensor):
            inner_cluster_threshold = threshold[np.where(cluster_id == id)[0].tolist()].view(-1)
            inner_cluster_threshold = inner_cluster_threshold.unsqueeze(-1)
        else:
            inner_cluster_threshold = threshold
        # 现在提取出了一个簇中间的所有特征的activation matrix，现在需要计算这些特征的co-activate情况

        feature_mean_coactivation_rate = ((cluster_feature_activation_matrix > inner_cluster_threshold).sum(dim=0) / cluster_feature_activation_matrix.shape[0]).mean()
        print("cluster", id, "co-activate rate:", feature_mean_coactivation_rate)


def cross_layer_circuit_analysis(
        args: EasyDict, 
        roi_name: str, 
        subj: int, 
        all_layers: int = 12, 
        threshold: Optional[Union[float, torch.Tensor]] = None,
        topk: int = 100, 
        visualize: bool = False, 
        n_neighbors: int = 3,
        min_dist: float = 0.1, 
        cluster_filter: bool = False, 
        cluster_filter_topk: int = 10, 
    ):
    """
    分析跨层的激活情况，并提取跨层的激活之间相关性

    Args:
        args (EasyDict): 实验参数文件
        roi_name (str): 分析针对哪一个ROI
        subj (int): 被试编码
        all_layers (int, optional): 模型总共有多少层. Defaults to 12.
        topk (int, optional): 每一层选择多少个最相关的特征. Defaults to 100.
        visualize (bool, optional): 是否可视化. Defaults to False.
    """

    selected_feature_activation, all_layer_feature_index = selected_sae_feature_activation_analysis(args=args, roi_name=roi_name, subj=subj, all_layers=all_layers, topk=topk)

    image_save_path = args.similarity.cross_layer_cluster_image_save_root
    info_save_path = args.similarity.cross_layer_cluster_info_save_root
    model_name = args.exp.model_name
    sae_name = args.autoencoder.name
    sae_rate = args.autoencoder.rate
    print("start cross layer cluster analysis")
    cross_layer_cluster(feature_matrix=selected_feature_activation, 
                        feature_index=all_layer_feature_index, 
                        dict_save_path=info_save_path.format(subj, roi_name, model_name, sae_name, sae_rate, topk), 
                        fig_save_path=image_save_path.format(subj, roi_name, model_name, sae_name, sae_rate, topk), 
                        visualize=visualize, 
                        n_neighbors=n_neighbors,
                        min_dist=min_dist, 
                        threshold=threshold, 
                        cluster_filter=cluster_filter, 
                        cluster_filter_topk=cluster_filter_topk, 
                        )
    print("finish cross layer cluster analysis")


# 前面的代码，是Brain Guide Circuit Selected + 基于激活的聚类，只能体现大脑的选择性，并不能够说明选择出来的是一个回路
# 下面的思路是，通过特征之间的组合，让其激活范式和对应脑区更相关

def save_model_patch_embedding(
        args: EasyDict, 
    ) -> torch.Tensor:
    """
    这个函数的目标是，保存目标模型在NSD图片数据集上的跨层patch embedding

    Args:
        args (EasyDict): 实验参数

    Returns:
        torch.Tensor: 每一层的patch embedding，其中tensor的第0维是层数，第1维是patch，第2维是图片数量
    """
    activation_info = [[] for _ in range(args.exp.layers)]
    brain_activation_list = []
    model, image_preprocess = load_target_model(args.exp.model_name)
    target_model = model.to(device=args.exp.device)
    with torch.no_grad():
        test_dataset = AnalysisDataset(args=args, image_preprocess=image_preprocess, text_preprocess=clip.tokenize)
        test_dataset.IndividualCondition()
        for image, _, brain_activation in tqdm(DataLoader(test_dataset, batch_size=512)):
            _, info = target_model.encoder_multilayer_information(image.to(device=args.exp.device), target_layer=[i for i in range(args.exp.layers)])
            for layer in range(args.exp.layers):
                activation_info[layer].append(info[layer][1:, :, :].cpu())
            brain_activation_list.append(brain_activation)
        activation_info = [torch.cat(layer_info, dim=1) for layer_info in activation_info]
        brain_activation = torch.cat(brain_activation_list, dim=0)
        if torch.isnan(brain_activation).any():
            nan_mask = torch.isnan(brain_activation).sum(dim=-1) == 0
            brain_activation = brain_activation[nan_mask]
            activation_info = [layer_info[:, nan_mask, :] for layer_info in activation_info]
        del brain_activation
        del brain_activation_list
        patch_embedding_save_root = args.similarity.model_patch_individual_embedding_save_root.format(args.exp.subj, args.exp.model_name)
        activation_info = torch.cat([layer_info.unsqueeze(0) for layer_info in activation_info], dim=0)
        check_path(patch_embedding_save_root)
        torch.save(activation_info, patch_embedding_save_root)
    
    return activation_info

def GreedyAnalysis(
        sae_activation: torch.Tensor, 
        brain_activation: torch.Tensor, 
        selected_feature_list: List[int], 
        inference_device: str, 
    ):
    candidates = [i for i in range(sae_activation.shape[-1]) if i not in selected_feature_list]
    shuffle(candidates)
    sae_selected_activation = sae_activation[:, selected_feature_list].view(1, -1)
    brain_activation = (brain_activation - brain_activation.mean(dim=0, keepdim=True)) / (brain_activation.norm(dim=0, keepdim=True) + 1e-8)
    sae_selected_activation = (sae_selected_activation - sae_selected_activation.mean(dim=1, keepdim=True)) / (sae_selected_activation.norm(dim=1, keepdim=True) + 1e-8)
    correlation = (sae_selected_activation @ brain_activation).mean()
    label = True
    while label:
        for feature in candidates:
            sae_selected_activation_temp = sae_activation[:, selected_feature_list + [feature]].sum(dim=1).view(1, -1)
            sae_selected_activation_temp = (sae_selected_activation_temp - sae_selected_activation_temp.mean(dim=1, keepdim=True)) / (sae_selected_activation_temp.norm(dim=1, keepdim=True) + 1e-8)
            correlation_temp = (sae_selected_activation_temp @ brain_activation).mean()
            if correlation_temp > correlation:
                correlation = correlation_temp
                selected_feature_list.append(feature)
                sae_selected_activation = sae_selected_activation_temp
                break
            label = False
        current_candidates = [i for i in range(sae_activation.shape[-1]) if i not in selected_feature_list]
        candidates = current_candidates
        shuffle(candidates)
    return selected_feature_list, correlation


def greedy_circuit(
        args: EasyDict, 
        roi_name: str, 
        extract_topk: int = 100, 
        inference_device: str = "cuda:0", 
    ):
    """
    通过贪心策略，找到最优的回路

    Args:
        args (EasyDict): 模型控制的参数
        roi_name (str): 分析哪一个脑区
        extract_topk (int, optional): 每一层选择多少个最相关的特征. Defaults to 100.
    """
    # 运行这个代码的时候，在cpu中进行运行
    # 提取大脑激活
    test_dataset = AnalysisDataset(args=args, text_preprocess=clip.tokenize)
    test_dataset.IndividualCondition()
    brain_activation = test_dataset.BrainActivation
    del test_dataset
    if not os.path.exists("{}_info.pt".format(roi_name)):
            

        # 提取的是模型在测试图片上的patch embedding
        model_patch_embedding_path = args.similarity.model_patch_individual_embedding_save_root.format(args.exp.subj, args.exp.model_name)
        if not os.path.exists(model_patch_embedding_path):
            model_patch_embedding = save_model_patch_embedding(args)
        else:
            model_patch_embedding = torch.load(model_patch_embedding_path)
        
        if torch.isnan(brain_activation).any():
            nan_mask = torch.isnan(brain_activation).sum(dim=-1) == 0
            brain_activation = brain_activation[nan_mask]
        subj = args.exp.subj
        roi_root = args.NSD.roi_index.format(subj, roi_name)
        roi_mask = load_target_roi_mask(roi_root)
        brain_activation = brain_activation.T[roi_mask].T

        # 首先提取和大脑激活最相关的SAEs特征，以及对应的index和相关性
        sae_activation_list = []
        saes_feature_index_list = []
        sae_correlated_value_list = []
        with torch.no_grad():
            for layer in range(args.exp.layers):
                roi_correlation = get_target_roi_correlation(args, roi_name=roi_name, subj=subj, target_layer=layer)
                sae = load_pretrained_autoencoder(args, layer=layer).cpu()
                # 计算当前层和这个roi最相关的topk个特征
                correlated_value, most_correlate_voxel_index = torch.topk(roi_correlation.mean(dim=-1).view(1, -1), k=extract_topk, dim=-1)
                sae_activation, _ = sae.encode(model_patch_embedding[layer].squeeze(0))
                sae_activation_list.append(sae_activation[:, :, most_correlate_voxel_index].mean(dim=0))
                saes_feature_index_list.append(most_correlate_voxel_index)
                sae_correlated_value_list.append(correlated_value.view(-1))
                del sae_activation
                del most_correlate_voxel_index
                del correlated_value
                del roi_correlation
                del sae

        saes_activation = torch.cat(sae_activation_list, dim=-1).squeeze(1) # [9000, 12 * extract_topk]
        sae_correlated_value = torch.cat(sae_correlated_value_list, dim=-1) # [12 * extract_topk]
        torch.save([saes_activation, sae_correlated_value], "{}_info.pt".format(roi_name))
    else:
        saes_activation, sae_correlated_value = torch.load("{}_info.pt".format(roi_name))
        saes_activation = saes_activation.squeeze(1)

    # 根据相关性，选取最相关的前10个作为种子，开始贪心搜索
    _, seed_index = torch.topk(sae_correlated_value, k=1200, dim=-1)
    saes_activation = saes_activation.to(device=inference_device)
    brain_activation = brain_activation.to(device=inference_device)
    all_info_list = []
    for seed in seed_index:
        selected_feature_list = [int(seed)]
        selected_circuit, correlation = GreedyAnalysis(sae_activation=saes_activation, brain_activation=brain_activation, selected_feature_list=selected_feature_list, inference_device=inference_device)
        all_info_list.append([selected_circuit, correlation])
    torch.save(all_info_list, "{}_all_circuit.pt".format(roi_name))
    all_info_list.sort(key=lambda x: x[1], reverse=True)
    for i in range(10):
        print("第{}个最优回路，相关性为{}".format(i, all_info_list[i][1]))
        print(all_info_list[i][0])
        print("\n")
