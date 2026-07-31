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



def selected_sae_feature_activated_activation_analysis(
        args: EasyDict,
        roi_name: str,
        subj: int,
        all_layers: int = 12,
        topk: int = 100,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    这个函数用于提取出，和指定ROI最相关的特征在ImageNet Test上的激活
    保存与目标脑区最相关的topk特征的激活时激活情况平均值

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
    selected_feature_activation_save_root = args.similarity.roi_selected_feature_activated_activation_save_root.format(subj, roi_name, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate, topk)
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
                    activation = activation[:, :, feature_index]
                    activation = activation.sum(dim=1) / ((activation > 0).sum(dim=1) + 1e-9)
                    selected_feature_activation[i].append(activation.cpu())
                del evaluating_data

        selected_feature_activation = [torch.cat(i, dim=0) for i in selected_feature_activation]
        check_path(selected_feature_activation_save_root)
        torch.save(selected_feature_activation, selected_feature_activation_save_root)
        return selected_feature_activation, all_layer_sae_feature_index