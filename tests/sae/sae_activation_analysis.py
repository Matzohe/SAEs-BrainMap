# 这个函数的作用，是对SAEs的所有patch激活平均值，patch激活时的激活平均值进行统计分析

import torch
from easydict import EasyDict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

def density_analysis_sae_activation(
        args: EasyDict, 
        roi_name: str, 
        topk: int = 100, 
        fig_save_root: str = "activation_analysis.png",
    ):
    """
    这个函数是打印所有选择出来的特征在所有ImageNet Test集上的激活平均值。
    激活时激活平均值的密度图，横坐标为激活平均值，纵坐标。
    偏离y=x越多，表明激活时激活强度越大，并且选择性越强
    如果所有patch都激活，那么就在y=x上

    Args:
        args (EasyDict): 算法参数
        roi_name (str): 对应的脑区名称
        topk (int, optional): 最高激活的topk个. Defaults to 100.
        figsave_root (str, optional): _description_. Defaults to "activation_analysis.png".
    """
    subj = args.exp.subj
    selected_feature_activated_activation_save_root = args.similarity.roi_selected_feature_activated_activation_save_root.format(subj, roi_name, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate, topk)
    selected_feature_activation_save_root = args.similarity.roi_selected_feature_activation_save_root.format(subj, roi_name, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate, topk)

    selected_activated_activation = torch.cat(torch.load(selected_feature_activated_activation_save_root), dim=-1).view(-1)
    selected_activation = torch.cat(torch.load(selected_feature_activation_save_root), dim=-1).view(-1)

    # 将activated activation作为y坐标，selected activation作为横坐标，画一个散点图，加上y=x的一个虚线
    plt.figure(figsize=(8, 6))
    plt.hexbin(selected_activation, selected_activated_activation, gridsize=200, cmap="viridis", bins="log")
    plt.colorbar()
    plt.plot(np.linspace(0, max(selected_activation.max(), selected_activated_activation.max()), 200), np.linspace(0, max(selected_activation.max(), selected_activated_activation.max()), 200), color="red", zorder=3)
    plt.xlabel("all mean activation")
    plt.ylabel("activated activation")
    plt.savefig(fig_save_root, dpi=300)
    plt.close()


def activation_distribution_histogram_plot(
        args: EasyDict, 
        roi_name: str, 
        topk: int = 100,  
        mean_activation_fig_save_root: str = "mean)activation_analysis_histogram_plot.png",
        activated_mean_activation_fig_save_root: str = "activated_mean_activation_analysis_histogram_plot.png",
    ):
    subj = args.exp.subj
    selected_feature_activated_activation_save_root = args.similarity.roi_selected_feature_activated_activation_save_root.format(subj, roi_name, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate, topk)
    selected_feature_activation_save_root = args.similarity.roi_selected_feature_activation_save_root.format(subj, roi_name, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate, topk)

    selected_activated_activation = torch.cat(torch.load(selected_feature_activated_activation_save_root), dim=-1).mean(dim=0).view(-1)
    selected_activation = torch.cat(torch.load(selected_feature_activation_save_root), dim=-1).mean(dim=0).view(-1)

    # 打印激活分布柱状图
    plt.figure(figsize=(8, 6))
    plt.hist(selected_activation.numpy(), bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("all mean activation")
    plt.ylabel("Frequency")
    plt.title("Distribution of all mean activation")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(mean_activation_fig_save_root, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(selected_activated_activation.numpy(), bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("activated mean activation")
    plt.ylabel("Frequency")
    plt.title("Distribution of activated mean activation")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(activated_mean_activation_fig_save_root, dpi=300)
    plt.close()
