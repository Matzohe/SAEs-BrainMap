import torch
from torch.utils.data import DataLoader
from easydict import EasyDict
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from ..util import check_path
from ..models.Vision import clip
from ..models.load_target_model import load_target_model
from ..SAEs.sae_loader import load_pretrained_autoencoder
from ..dataset.Coco.CocoNSDAnalysis import AnalysisDataset
from ..dataset.NSD.NSD_utils import load_target_roi_mask


def plot_rsm_heatmap(corr, save_root, ticks):
    """可视化单个 N x N 的 RSM 矩阵"""
    corr[corr < 0] = 0
    print(max(corr.flatten()))
    plt.imshow(corr, cmap='hot', origin='lower')
    plt.xticks(ticks)
    plt.yticks(ticks)
    # 隐藏 x/y 轴上的数字（保留刻度线）
    plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    plt.savefig(save_root, dpi=600)
    plt.close()

def voxel_dictionary_rsa_selection(
        args: EasyDict, 
        all_layers: int = 12, 
    ):
    brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, 0, args.autoencoder.rate)
    sae_similarity = torch.load(brain_sae_similarity_save_root)

    voxel_dictionary_index = torch.zeros(size=(sae_similarity.shape[-1],))
    voxel_correlation = -torch.ones(size=(sae_similarity.shape[-1],))
    voxel_layers = torch.zeros(size=(sae_similarity.shape[-1],))

    del brain_sae_similarity_save_root
    del sae_similarity

    for layer in range(all_layers):
        brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, layer, args.autoencoder.rate)
        sae_similarity = torch.load(brain_sae_similarity_save_root)
        max_sae_similarity, max_sae_similarity_index = sae_similarity.max(dim=0)
        voxel_correlation_before = voxel_correlation
        voxel_correlation = torch.where(max_sae_similarity > voxel_correlation_before, max_sae_similarity, voxel_correlation)
        voxel_dictionary_index = torch.where(max_sae_similarity > voxel_correlation_before, max_sae_similarity_index, voxel_dictionary_index)
        voxel_layers = torch.where(max_sae_similarity > voxel_correlation_before, layer, voxel_layers)

    # 在这一步完成了每个体素最相关的activation和layer的索引，下面根据选择的内容，获取saes的激活，同时建立voxel dictionary。

    target_model, image_preprocess = load_target_model(args.exp.model_name)
    target_model.eval()
    target_model = target_model.to(device=args.exp.device)

    sae_list = []
    for layer in range(args.exp.layers):
        sae = load_pretrained_autoencoder(args, layer=layer)
        sae_list.append(sae.to(device=args.exp.device))
    activation_info = [[] for _ in range(args.exp.layers)]
    model_neuron_activation_list = [[] for _ in range(args.exp.layers)]
    sae_dtype = eval(args.autoencoder.dtype)
    # 添加一步，计算神经元和大脑激活之间的RSA Score
    with torch.no_grad():
        test_dataset = AnalysisDataset(args=args, image_preprocess=image_preprocess, text_preprocess=clip.tokenize)
        test_dataset.IndividualCondition()
        brain_activation_list = []
        
        for image, _, brain_activation in tqdm(DataLoader(test_dataset, batch_size=512)):
            _, info = target_model.encoder_multilayer_information(image.to(device=args.exp.device), target_layer=[i for i in range(args.exp.layers)])
            for layer, sae in zip(range(args.exp.layers), sae_list):
                sae.eval()
                with torch.no_grad():
                    sae_activation, _ = sae.encode(info[layer][1:, :, :].to(dtype=sae_dtype))
                    # sae_activation = (sae_activation - sae_activation.mean(dim=1, keepdim=True)) / (torch.std(sae_activation, dim=1, keepdim=True) + 1e-8)
                    sae_activation = sae_activation.mean(dim=0)
                    # sae_activation = sae.encode_pre_act(info[layer][0, :, :].to(dtype=sae_dtype))
                model_neuron_activation_list[layer].append(info[layer].mean(dim=0).squeeze(0).cpu())
                activation_info[layer].append(sae_activation.squeeze(0).cpu())
            brain_activation_list.append(brain_activation)
        activation_info = [torch.cat(layer_info, dim=0) for layer_info in activation_info]
        model_neuron_activation_list = [torch.cat(layer_info, dim=0) for layer_info in model_neuron_activation_list]
        brain_activation = torch.cat(brain_activation_list, dim=0)

    voxel_dictionary = torch.zeros(size=(voxel_dictionary_index.shape[0], sae_list[0].encoder.weight.shape[-1]))
    voxel_activation = torch.zeros(size=(brain_activation.shape[0], voxel_dictionary_index.shape[0]))
    neuron_activation = torch.zeros(size=(brain_activation.shape[0], voxel_dictionary_index.shape[0]))
    neuron_brain_correlation = -torch.ones(size=(voxel_dictionary_index.shape[0],))
    neuron_brain_index = torch.zeros(size=(voxel_dictionary_index.shape[0],))
    neuron_brain_layer_index = torch.zeros(size=(voxel_dictionary_index.shape[0],))
    # 下面需要计算神经元和大脑激活之间的相关性，从而确认最终选定的neuron activation list
    for layer in range(all_layers):
        new_brain_activation = brain_activation - brain_activation.mean(dim=0, keepdim=True)
        new_brain_activation = new_brain_activation / (torch.std(new_brain_activation, dim=0, keepdim=True) + 1e-8)
        new_neuron_activation = model_neuron_activation_list[layer] - model_neuron_activation_list[layer].mean(dim=0, keepdim=True)
        new_neuron_activation = new_neuron_activation / (torch.std(new_neuron_activation, dim=0, keepdim=True) + 1e-8)

        brain_neuron_correlation = new_neuron_activation.T @ new_brain_activation

        max_brain_neuron_correlation, max_brain_neuron_correlation_index = brain_neuron_correlation.max(dim=0)
        max_brain_neuron_correlation = max_brain_neuron_correlation.cpu()
        max_brain_neuron_correlation_index = max_brain_neuron_correlation_index.cpu()
        new_neuron_brain_correlation = torch.where(max_brain_neuron_correlation > neuron_brain_correlation, max_brain_neuron_correlation, neuron_brain_correlation)
        new_neuron_brain_index = torch.where(max_brain_neuron_correlation > neuron_brain_correlation, max_brain_neuron_correlation_index, neuron_brain_index)
        new_neuron_brain_layer_index = torch.where(max_brain_neuron_correlation > neuron_brain_correlation, layer, neuron_brain_layer_index)

        neuron_brain_correlation = new_neuron_brain_correlation
        neuron_brain_index = new_neuron_brain_index
        neuron_brain_layer_index = new_neuron_brain_layer_index

    for layer in range(all_layers):
        neuron_activation[:, neuron_brain_layer_index == layer] = model_neuron_activation_list[layer][:, neuron_brain_index[neuron_brain_layer_index == layer].long()].cpu()

    for layer in range(all_layers):
        voxel_dictionary[voxel_layers == layer, :] = sae_list[layer].encoder.weight[voxel_dictionary_index[voxel_layers == layer].long(), :].cpu()
        voxel_activation[:, voxel_layers == layer] = activation_info[layer][:, voxel_dictionary_index[voxel_layers == layer].long()].cpu()
    
    voxel_dictionary_rsm_save_root = args.RSA.voxel_dictionary_rsm_save_root.format(args.exp.subj, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate)
    brain_activation_rsm_save_root = args.RSA.brain_activation_rsm_save_root.format(args.exp.subj)

    if os.path.exists(brain_activation_rsm_save_root):
        brain_activation_rsm = torch.load(brain_activation_rsm_save_root, weights_only=False)
    else:
        brain_activation_rsm = torch.corrcoef(brain_activation.T)
        check_path(brain_activation_rsm_save_root)
        torch.save(brain_activation_rsm, brain_activation_rsm_save_root)
    
    neuron_activation_rsm = torch.corrcoef(neuron_activation.T)


    if os.path.exists(voxel_dictionary_rsm_save_root):
        voxel_dictionary_rsm = torch.load(voxel_dictionary_rsm_save_root, weights_only=False)
    else:
        voxel_dictionary_rsm = torch.corrcoef(voxel_activation.T)
        check_path(voxel_dictionary_rsm_save_root)
        torch.save(voxel_dictionary_rsm, voxel_dictionary_rsm_save_root)
    
    voxel_dictionary_save_root = args.RSA.voxel_dictionary_save_root.format(args.exp.subj, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate)
    check_path(voxel_dictionary_save_root)
    torch.save(voxel_dictionary, voxel_dictionary_save_root)
    new_voxel_dictionary_activation = []
    new_brain_activation = []
    # 可视化brain activation和voxel dictionary的rsm，这里要按照roi进行重新排序
    roi_mask_root = args.NSD.roi_index
    feature_length_list = []
    feature_number = 0
    for roi in ["v1", "v2", "v3", "hv4", "FFA", "EBA", "RSC", "VWFA", "FOOD",]:
        roi_mask = load_target_roi_mask(roi_mask_root.format(args.exp.subj, roi))
        new_voxel_dictionary_activation.append(voxel_activation.T[roi_mask].T)
        new_brain_activation.append(brain_activation.T[roi_mask].T)
        feature_number = feature_number + roi_mask.sum()
        feature_length_list.append(feature_number)

    new_voxel_dictionary_activation = torch.cat(new_voxel_dictionary_activation, dim=1)
    new_brain_activation = torch.cat(new_brain_activation, dim=1)

    # 计算这两个矩阵的RSM

    new_voxel_dictionary_activation = new_voxel_dictionary_activation.T - new_voxel_dictionary_activation.T.mean(dim=1, keepdim=True)
    new_brain_activation = new_brain_activation.T - new_brain_activation.T.mean(dim=1, keepdim=True)
    new_voxel_dictionary_activation = new_voxel_dictionary_activation / (new_voxel_dictionary_activation.std(dim=1, keepdim=True) + 1e-8)
    new_brain_activation = new_brain_activation / (new_brain_activation.std(dim=1, keepdim=True) + 1e-8)

    brain_corr = (new_brain_activation @ new_brain_activation.T) / new_brain_activation.shape[1]
    voxel_dictionary_corr = (new_voxel_dictionary_activation @ new_voxel_dictionary_activation.T) / new_voxel_dictionary_activation.shape[1]
    # 打印两个方法的RSM矩阵
    plot_rsm_heatmap(brain_corr, save_root="brain_rsm_without_info.png", ticks=feature_length_list)
    plot_rsm_heatmap(voxel_dictionary_corr, save_root="voxel_dictionary_rsm_without_info.png", ticks=feature_length_list)


    upper_tri_indices = torch.triu_indices(brain_activation_rsm.shape[0], brain_activation_rsm.shape[0], offset=1)
    brain_activation_rsm = brain_activation_rsm[upper_tri_indices[0], upper_tri_indices[1]]
    voxel_dictionary_rsm = voxel_dictionary_rsm[upper_tri_indices[0], upper_tri_indices[1]]
    neuron_activation_rsm = neuron_activation_rsm[upper_tri_indices[0], upper_tri_indices[1]]
    rsa_score, p_value = spearmanr(brain_activation_rsm.cpu().numpy(), voxel_dictionary_rsm.cpu().numpy())
    neuron_rsa_score, neuron_p_value = spearmanr(brain_activation_rsm.cpu().numpy(), neuron_activation_rsm.cpu().numpy())
    print("SAEs RSA Score Visualization\nSpearman $r$ = {:.3f}, $P$ = {:.3f}".format(rsa_score, p_value))
    print("Neuron RSA Score Visualization\nSpearman $r$ = {:.3f}, $P$ = {:.3f}".format(neuron_rsa_score, neuron_p_value))
    # # 添加RSA的散点可视化图
    # plt.figure(figsize=(7, 7))
    # plt.hexbin(brain_activation_rsm.cpu().numpy(), voxel_dictionary_rsm.cpu().numpy(), bins='log', gridsize=100, mincnt=20)

    # # 使用 np.polyfit 计算线性拟合的系数
    # z = np.polyfit(brain_activation_rsm.cpu().numpy(), voxel_dictionary_rsm.cpu().numpy(), 1)
    # p = np.poly1d(z)
    # print(p)
    # plt.plot(brain_activation_rsm, p(brain_activation_rsm), "r--") # 画出拟合线
    # plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    # # plt.title(f'RSA Score Visualization\nSpearman $r$ = {rsa_score:.3f}, $P$ = {p_value:.3f}')
    # # plt.xlabel('RSM A Similarity Value (Brain Activation)')
    # # plt.ylabel('RSM B Similarity Value (Voxel Dictionary/Model)')
    # # 确保 X 和 Y 轴比例一致，便于观察相关性
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig("rsa_without_label.png", dpi=600)
    # plt.close()
    # brain_voxel_dictionary_rsa_score_save_root = args.RSA.brain_voxel_dictionary_rsa_score_save_root.format(args.exp.subj, args.exp.model_name, args.autoencoder.name, args.autoencoder.rate)
    # check_path(brain_voxel_dictionary_rsa_score_save_root)
    # torch.save({"rsa_score": rsa_score, "p_value": p_value}, brain_voxel_dictionary_rsa_score_save_root)
    
    return {
        "rsa_score": rsa_score,
        "p_value": p_value, 
        "brain_activation_rsm": brain_activation_rsm,
        "voxel_dictionary_rsm": voxel_dictionary_rsm, 
        "voxel_dictionary": voxel_dictionary, 
        "voxel_dictionary_index": voxel_dictionary_index, 
        "voxel_layers": voxel_layers, 
        "voxel_activation": voxel_correlation, 
    }
