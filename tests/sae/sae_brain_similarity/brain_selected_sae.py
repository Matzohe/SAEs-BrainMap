# 主要根据sae特征和大脑之间的激活相关性去选择相应的SAEs的特征，看大脑和saes之间是否有激活相关性以及选择性相关性
import torch
import numpy as np
from typing import Optional, List
from easydict import EasyDict
from torchvision import transforms
import os
import h5py
from tqdm import tqdm
from PIL import Image
from src.util import check_path
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tests.sae.sae_brain_similarity.sae_brain_similarity import mean_similarity_analysis
from src.visualize.umap import UMAPVisualize
from src.visualize.pca import PCAVisualize
from src.models.load_target_model import load_target_model
from src.dataset.ImageNet.ImageNet import ImageNetTestDataset
from src.SAEs.sae_loader import load_pretrained_autoencoder

def load_target_roi_mask(
        root: str,
    ) -> torch.Tensor:
    """
    倒入对应roi的mask

    Args:
        root (str): The path to the ROI mask file.

    Returns:
        torch.Tensor: The ROI mask tensor.
    """
    roi_index_tensor = torch.from_numpy(np.loadtxt(root, dtype=np.float32, delimiter=",")) > 0
    return roi_index_tensor


def get_target_roi_correlation(
        args: EasyDict,
        roi_name: str,
        target_layer: int,
        subj: int
    ) -> torch.Tensor:
    """
    获取目标roi的相关性，返回一个tensor，包含当前roi中的体素和所有特征的相关性

    Args:
        args (EasyDict): 全部参数
        roi_name (str): 脑区名称
        target_layer (int): 目标层
        subj (int): 被试编号

    Returns:
        torch.Tensor: 当前roi中的体素和所有特征的相关性，列为特征，行为roi
    """
    roi_root = args.NSD.roi_index.format(subj, roi_name)
    roi_mask = load_target_roi_mask(roi_root)
    similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, target_layer, args.autoencoder.rate)
    if not os.path.exists(similarity_save_root):
        target_model, image_preprocess = load_target_model(args.exp.model_name)
        target_model = target_model.to(args.exp.device)
        mean_similarity_analysis(args, target_model=target_model, image_preprocess=image_preprocess)
    similarity = torch.load(similarity_save_root)
    
    return similarity.T[roi_mask].T

def visualize_selected_sae_feature(
        args: EasyDict,
        subj: int,
        roi_name: str,
        layer: int,
        feature_index: torch.Tensor,
        roi_level: bool = False, 
        save_independently: bool = False, 
        heatmap: str = "jet", 
):
    """
    将选择出来的特征选择性进行可视化

    Args:
        args (EasyDict): 全部参数
        subj (int): 被试编号
        roi_name (str): 脑区名称
        layer (int): 目标层
        feature_index (torch.Tensor): 选择的特征
        roi_level (bool, optional): 是否为roi level的可视化，如果是，则选择出来的特征是以整个roi为基础进行选择的，否则是基于体素进行选择的. Defaults to False.
        save_independently (bool, optional): 是否保存每个特征单独的图片，如果是True，则最后的保存路径会进行一定的修改. Defaults to False.
        heatmap (str, optional): 选择性热图的颜色. Defaults to "jet".
    """
    sae = load_pretrained_autoencoder(args, layer=layer)
    device = args.exp.device
    model_name = args.exp.model_name
    target_model, image_preprocess = load_target_model(args.exp.model_name)
    target_model = target_model.to(device=device).eval()
    sae = sae.to(device=device)
    target_layer = [layer]
    saes = [sae]
    ImageNetTestTokenSavePath = args.SAEsEvaluation.imagenet_test_token_save_root
    topk_activation = [[] for _ in range(len(target_layer))]
    dataset  = ImageNetTestDataset(args.dataset.imagenet_test_root, image_preprocess=image_preprocess)
    image_root_list = dataset.root_list
    del dataset
    with torch.no_grad():
        for batch in tqdm(range(98), desc="Top Activation Extraction", total=98):
            with h5py.File(ImageNetTestTokenSavePath.format(model_name, batch), "r") as f:
                evaluating_data = torch.from_numpy(f['token embedding'][target_layer, :, 1:, :]).to(device=device)  # (1024, 196, 768)
                f.close()
            for i, layer in enumerate(target_layer):
                sae = saes[i]
                activation, _ = sae.encode(evaluating_data[i].squeeze(0))
                activation = activation.mean(dim=1)
                activation = activation[:, feature_index]
                topk_activation[i].append(activation.cpu())
            del evaluating_data

    for j, layer in enumerate(target_layer):
        target_topk_activation = torch.cat(topk_activation[j], dim=0)
        image_info = torch.topk(target_topk_activation, k=100, dim=0).indices
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        sae = saes[j]
        sae = sae.to(device=device).eval()
        with torch.no_grad():
            for topk_feature_id in range(len(feature_index)):
                if not save_independently:
                    fig, axes = plt.subplots(5, 8)
                    axes = axes.flatten()
                    for i, id in enumerate(image_info[:20, topk_feature_id]):
                        image_root = image_root_list[id]
                        old_image = Image.open(image_root).convert("RGB")
                        image = image_preprocess(old_image)
                        old_image = transform(old_image)
                        _, image_embedding = target_model.encoder_multilayer_information(image.unsqueeze(0).to(device), target_layer=[k for k in range(args.exp.layers)])
                        image_embedding = image_embedding[layer].squeeze(1)[1:, :]
                        sae_activation, _ = sae.encode(image_embedding)
                        sae_activation = sae_activation[:, feature_index[topk_feature_id]]
                        heatmap_shape = int(math.sqrt(sae_activation.shape[0]))
                        heatmap = sae_activation.view(heatmap_shape, heatmap_shape).detach().cpu().unsqueeze(0).unsqueeze(0)
                        heatmap_resized = F.interpolate(heatmap, size=(224, 224), mode='bilinear')
                        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
                        heatmap_resized = (heatmap_resized * 255).squeeze(0).squeeze(0).numpy().astype(np.uint8)
                        ax = axes[i * 2]
                        ax.imshow(old_image)
                        ax.axis("off")
                        ax = axes[i * 2 + 1]
                        ax.imshow(old_image)
                        ax.imshow(heatmap_resized, cmap='jet', alpha=0.3)
                        ax.axis("off")
                    if not roi_level:
                        image_save_root = args.similarity.voxel_dictionary_feature_selectivity_heatmap_save_root.format(subj, roi_name, model_name, args.autoencoder.name, args.autoencoder.rate, layer, feature_index[topk_feature_id])
                    else:
                        image_save_root = args.similarity.roi_selected_feature_heatmap_save_root.format(subj, roi_name, model_name, args.autoencoder.name, args.autoencoder.rate, layer, topk_feature_id, feature_index[topk_feature_id])
                    check_path(image_save_root)
                    plt.savefig(image_save_root, dpi=300)
                    plt.close()
                elif save_independently:
                    for i, id in enumerate(image_info[:20, topk_feature_id]):
                        image_root = image_root_list[id]
                        old_image = Image.open(image_root).convert("RGB")
                        image = image_preprocess(old_image)
                        old_image = transform(old_image)
                        _, image_embedding = target_model.encoder_multilayer_information(image.unsqueeze(0).to(device), target_layer=[k for k in range(args.exp.layers)])
                        image_embedding = image_embedding[layer].squeeze(1)[1:, :]
                        sae_activation, _ = sae.encode(image_embedding)
                        sae_activation = sae_activation[:, feature_index[topk_feature_id]]
                        heatmap_shape = int(math.sqrt(sae_activation.shape[0]))
                        heatmap = sae_activation.view(heatmap_shape, heatmap_shape).detach().cpu().unsqueeze(0).unsqueeze(0)
                        heatmap_resized = F.interpolate(heatmap, size=(224, 224), mode='bilinear')
                        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
                        heatmap_resized = (heatmap_resized * 255).squeeze(0).squeeze(0).numpy().astype(np.uint8)
                        image_save_root = args.similarity.roi_selected_feature_heatmap_independent_save_root.format(subj, roi_name, model_name, args.autoencoder.name, args.autoencoder.rate, layer, topk_feature_id, feature_index[topk_feature_id], i)
                        original_image_save_root = args.similarity.roi_selected_feature_original_independent_save_root.format(subj, roi_name, model_name, args.autoencoder.name, args.autoencoder.rate, layer, topk_feature_id, feature_index[topk_feature_id], i)
                        check_path(image_save_root)
                        check_path(original_image_save_root)
                        plt.imshow(old_image)
                        plt.axis("off")
                        plt.savefig(original_image_save_root)
                        plt.close()
                        plt.imshow(old_image)
                        plt.imshow(heatmap_resized, cmap='jet', alpha=0.3)
                        plt.axis("off")
                        plt.savefig(image_save_root)
                        plt.close()

def extract_target_roi_voxel_dictionary(
        args: EasyDict,
        roi_name: str,
        target_layer: int,
        subj: int,
    ):
    """
    获取与当前roi最相关的体素字典，同时，将选择出的字典进行降维可视化

    Args:
        args (EasyDict): 全部参数
        roi_name (str): 脑区名称
        target_layer (int): 目标层
        subj (Optional[int], optional): 被试号. Defaults to None.

    """
    roi_correlation = get_target_roi_correlation(args, roi_name, target_layer, subj)
    sae = load_pretrained_autoencoder(args, layer=target_layer)
    sae_weight = sae.decoder.weight.T.data
    # 根据相关性建立起体素字典
    _, weight_mask = torch.max(roi_correlation, dim=0)
    _, most_correlate_voxel_index = torch.max(roi_correlation.mean(dim=-1).view(1, -1), dim=-1)
    print("layer{} ROI{}:{}".format(target_layer, roi_name, most_correlate_voxel_index))
    voxel_dict = sae_weight[weight_mask]
    unique_index = torch.unique(weight_mask)
    unique_weight_dict = sae_weight[unique_index]
    # PCAVisualize(unique_weight_dict, n_components=2)
    # UMAPVisualize(unique_weight_dict, n_neighbors=30, min_dist=0.3, save=True, show=True, save_path="visualize.png", n_components=2, color_list=[[0, 0, 1.0] for _ in range(unique_weight_dict.shape[0])])
    visualize_selected_sae_feature(args, subj=subj, roi_name=roi_name, layer=target_layer, feature_index=unique_index, roi_level=False)
    print("finish visualize")


def extract_roi_correlate_feature(
        args: EasyDict,
        roi_name: str,
        target_layers: List[int],
        subj: int,
        extract_topk: int = 100, 
        visualize_topk: int = 100,
        heatmap: str = "jet", 
        pca_visualize: bool = False, 
        save_independently: bool = False
    ):
    """
    获取指定层与当前roi最相关的数个特征，在指定层中，逐层选择相关的特征，可视化这些特征的同时，结合12层进行降维可视化
    最后目标分析模型特定功能的回路

    Args:
        args (EasyDict): 全部参数
        roi_name (str): 脑区名称
        target_layer (int): 目标层
        subj (Optional[int], optional): 被试号. Defaults to None.
        extract_topk (int, optional): 选择最相关的特征的前多少数量. Defaults to 100.
        visualize_topk (int, optional): 可视化的前几特征的数量. Defaults to 100.
        heatmap (str, optional): 热图颜色. Defaults to "jet".
        pca_visualize (bool, optional): 是否降维可视化. Defaults to False.
        save_independently (bool, optional): 是否单独保存每层的可视化结果. Defaults to False.

    """
    features = []
    for target_layer in target_layers:
        roi_correlation = get_target_roi_correlation(args, roi_name, target_layer, subj)
        sae = load_pretrained_autoencoder(args, layer=target_layer)
        sae_weight = sae.decoder.weight.T.data
        # 下面计算当前层和这个roi最相关的topk个特征
        _, most_correlate_voxel_index = torch.topk(roi_correlation.mean(dim=-1).view(1, -1), k=extract_topk, dim=-1)
        features.append(sae_weight[most_correlate_voxel_index.squeeze(0)])
        visualize_selected_sae_feature(args, subj=subj, roi_name=roi_name, layer=target_layer, feature_index=most_correlate_voxel_index.squeeze(0)[: visualize_topk], roi_level=True, heatmap=heatmap, save_independently=save_independently)
    features = torch.cat(features, dim=0)
    if pca_visualize:
        PCAVisualize(features, n_components=2, save_root="{}_pca.png".format(roi_name))
