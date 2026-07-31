import nibabel as nib
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from ..dataset.NSD.NSDDataLoader import NSDDataset
from .cls_embedding import save_cls_embedding
from ..models.load_target_model import load_target_model
from ..models.Vision.clip import tokenize
from ..dataset.Coco.CocoNSDAnalysis import AnalysisDataset
from ..util import check_path
from .model.ridge import ridge_prediction

def BrainActivation(args):
    """
    输入args参数集合, 返回对应被试, 对应脑区的独立和共同图像的平均激活值, 以及当前roi中的voxel个数

    Args:
        args (_type_): 参数集合

    Returns:
        (torch.Tensor, torch.Tensor, int): 脑区的独立和共同图像的平均激活值, 以及当前roi中的voxel个数
    """

    subj = args.exp.subj
    roi = args.exp.full_roi
    nsd_dataset = NSDDataset(args)
    voxel_activation = nsd_dataset.load_avg_activation_value(subj=subj, roi_name=roi)
    individual_mask, same_mask = nsd_dataset.load_individual_and_same_image_bool(subj=subj)

    return voxel_activation[individual_mask], voxel_activation[same_mask], voxel_activation.shape[-1]


def ImageEmbedding(args):
    device = args.exp.device
    target_model, image_preprocess = load_target_model(args.exp.model_name)
    target_model = target_model.to(device=device).eval()
    dataset = AnalysisDataset(args=args, image_preprocess=image_preprocess, text_preprocess=tokenize)
    dataset.IndividualCondition()
    individual_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    individual_image_embedding_save_root = args.BrainEncoder.individual_image_embedding_save_root.format(args.exp.model_name, args.exp.subj)
    try:
        individual_image_embedding = torch.load(individual_image_embedding_save_root)
    except:
        check_path(individual_image_embedding_save_root)
        individual_image_embedding = save_cls_embedding(target_model=target_model, dataloader=individual_dataloader, save_path=individual_image_embedding_save_root, device=device)
    dataset = AnalysisDataset(args=args, image_preprocess=image_preprocess, text_preprocess=tokenize)
    dataset.SameCondition()
    same_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    same_image_embedding_save_root = args.BrainEncoder.same_image_embedding_save_root.format(args.exp.model_name, args.exp.subj)
    try:
        same_image_embedding = torch.load(same_image_embedding_save_root)
    except:
        check_path(same_image_embedding_save_root)
        same_image_embedding = save_cls_embedding(target_model=target_model, dataloader=same_dataloader, save_path=same_image_embedding_save_root, device=device)
    
    return individual_image_embedding, same_image_embedding


def RidgeTrainer(args):
    """
    Ridge训练，中间会保存Ridge的权重，同时保存ncsnr和ridge预测出的R2值的图
    最后返回R2值，用于后续的分析
    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """

    ridge_weight_save_root = args.BrainEncoder.ridge_weight_save_root.format(args.exp.model_name, args.exp.subj)

    idv_brain_activation, sm_brain_activation, _ = BrainActivation(args)
    idv_image_embedding, sm_image_embedding = ImageEmbedding(args)

    r2_score, ridge_weight, ridge_bias = ridge_prediction(idv_image_embedding, sm_image_embedding, idv_brain_activation, sm_brain_activation, device=args.exp.device)
    check_path(ridge_weight_save_root)

    torch.save({"weight": ridge_weight, "bias": ridge_bias}, ridge_weight_save_root)

    ncsnr = nib.load(args.NSD.nsd_ncsnr_save_root.format(args.exp.subj)).get_fdata()
    roi_mask = torch.load(args.NSD.roi_mask_save_root.format(args.exp.subj, args.exp.full_roi), weights_only=False)
    ncsnr_value = ncsnr[roi_mask]
    plot_save_root = args.BrainEncoder.rsq_noise_celling_save_root.format(args.exp.model_name, args.exp.subj, args.exp.full_roi)
    check_path(plot_save_root)
    # 绘制散点图
    plt.figure(figsize=(5, 5))
    x = np.linspace(0, 1, 200)
    plt.plot(x, x, color="red", label="noise ceiling", zorder=3)
    plt.plot(x, 0.8 * x, color="orange", linestyle="--", label="80% noise ceiling", zorder=3)
    plt.hexbin(ncsnr_value, r2_score, gridsize=200, cmap="viridis", bins="log")
    plt.colorbar()
    plt.xlabel("noise celling")
    plt.ylabel("r2 score")
    plt.legend(loc="upper left")
    plt.savefig(plot_save_root, dpi=300)
    plt.close()
    return r2_score

def RidgeAnalysis(args):
    """
    TODO: Ridge分析，导入预训练好的ridge模型，分析其R2值，同时保存ncsnr和ridge预测出的R2值的图
    Args:
        args (_type_): _description_
    """
    pass
