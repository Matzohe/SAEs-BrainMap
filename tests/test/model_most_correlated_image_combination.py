import torch
import matplotlib.pyplot as plt
import os
from easydict import EasyDict
import numpy as np
from PIL import Image
from src.util import check_path

# 此方法主要实现几个可视化结果
# 第一，将每个模型，每个脑区，每层最相关的前10张图进行SAEs特征可视化

def image_process(image):
    image = image[240 - 112 - 10: 240 + 112 + 10, 320 - 112 - 10: 320 + 112 + 10, :3]
    return image


def visualize_top5_feature(
        args: EasyDict, 
        subj: int, 
        model_name: str, 
        roi_name: str, 
        target_layer: int, 
        select_image_number: int = 10, 
    ):
    """
    提取对应被试，对应模型名称，对应脑区对应层的最相关top5选择性特征。

    Args:
        args (EasyDict): 模型全部参数
        subj (int): 被试
        model_name (str): 目标模型名称
        roi_name (str): 目标脑区
        target_layer (int): 目标层
        select_image_number (int, optional): 每层可视化多少最相关的特征. Defaults to 10.
    """

    target_save_root = args.similarity.roi_selected_feature_heatmap_independent_save_root.format(subj, roi_name, model_name, args.autoencoder.name, args.autoencoder.rate, target_layer, 0, 0, 0)
    target_save_root = "/".join(target_save_root.split("/")[:-2])
    feature_name_list = os.listdir(target_save_root)

    target_heatmap_image_save_root_list = [[] for _ in range(5)]
    target_original_image_save_root_list = [[] for _ in range(5)]

    

    for name in feature_name_list:
        if "top0_" in name:
            for i in range(select_image_number):
                target_heatmap_image_save_root = os.path.join(target_save_root, name, "{}.png".format(i))
                target_original_image_save_root = os.path.join(target_save_root, name, "original_{}.png".format(i))
                target_heatmap_image_save_root_list[0].append(target_heatmap_image_save_root)
                target_original_image_save_root_list[0].append(target_original_image_save_root)

        elif "top1_" in name:
            for i in range(select_image_number):
                target_heatmap_image_save_root = os.path.join(target_save_root, name, "{}.png".format(i))
                target_original_image_save_root = os.path.join(target_save_root, name, "original_{}.png".format(i))
                target_heatmap_image_save_root_list[1].append(target_heatmap_image_save_root)
                target_original_image_save_root_list[1].append(target_original_image_save_root)

        elif "top2_" in name:
            for i in range(select_image_number):
                target_heatmap_image_save_root = os.path.join(target_save_root, name, "{}.png".format(i))
                target_original_image_save_root = os.path.join(target_save_root, name, "original_{}.png".format(i))
                target_heatmap_image_save_root_list[2].append(target_heatmap_image_save_root)
                target_original_image_save_root_list[2].append(target_original_image_save_root)

        elif "top3_" in name:
            for i in range(select_image_number):
                target_heatmap_image_save_root = os.path.join(target_save_root, name, "{}.png".format(i))
                target_original_image_save_root = os.path.join(target_save_root, name, "original_{}.png".format(i))
                target_heatmap_image_save_root_list[3].append(target_heatmap_image_save_root)
                target_original_image_save_root_list[3].append(target_original_image_save_root)

        elif "top4_" in name:
            for i in range(select_image_number):
                target_heatmap_image_save_root = os.path.join(target_save_root, name, "{}.png".format(i))
                target_original_image_save_root = os.path.join(target_save_root, name, "original_{}.png".format(i))
                target_heatmap_image_save_root_list[4].append(target_heatmap_image_save_root)
                target_original_image_save_root_list[4].append(target_original_image_save_root)

    for i in range(5):
        img_list = []
        for image_root in target_heatmap_image_save_root_list[i]:
            img = image_process(plt.imread(image_root))
            img_list.append(img)
        img_list = np.hstack(img_list)
        image_save_root = args.similarity.roi_selected_feature_combined_heatmap_save_root.format(subj, roi_name, model_name, args.autoencoder.name, args.autoencoder.rate, target_layer, i)
        check_path(image_save_root)
        plt.imsave(image_save_root, img_list)

    for i in range(5):
        img_list = []
        for image_root in target_original_image_save_root_list[i]:
            img = image_process(plt.imread(image_root))
            img_list.append(img)
        img_list = np.hstack(img_list)
        image_save_root = args.similarity.roi_selected_feature_combined_original_save_root.format(subj, roi_name, model_name, args.autoencoder.name, args.autoencoder.rate, target_layer, i)
        check_path(image_save_root)
        plt.imsave(image_save_root, img_list)


def inner_cluster_visualize_cluster(
        args: EasyDict, 
        subj: int, 
        model_name: str, 
        roi_name: str, 
        target_layer: int, 
    ):
    inner_cluster_info_save_root = args.similarity.inner_cluster_info_save_root.format(subj, roi_name, model_name, args.autoencoder.name, args.autoencoder.rate, target_layer, 100)
    cluster_info = torch.load(inner_cluster_info_save_root, weights_only=False)
    for cluster_id in cluster_info.keys():
        print("cluster id:", cluster_id)

