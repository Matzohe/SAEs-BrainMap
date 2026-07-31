# 这个函数的目标，是根据特征-脑区相关性，将模型层映射到大脑皮层上
# 为每个体素找到最相关的模型层，这里最相关有mean mapping和max mapping
import torch
from easydict import EasyDict
import numpy as np
from ..visualize.NSDVisualizer.VisualPipeline.visualize import visualize

def mean_layer_mapping(
        args: EasyDict, 
        all_layers: int = 12
    ):
    
    brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, 0, args.autoencoder.rate)
    sae_similarity = torch.load(brain_sae_similarity_save_root)
    max_value = -np.ones(shape=(sae_similarity.shape[-1], ))
    max_layer = np.zeros(shape=(sae_similarity.shape[-1], ))
    del brain_sae_similarity_save_root
    del sae_similarity

    for layer in range(all_layers):
        brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, layer, args.autoencoder.rate)
        sae_similarity = torch.load(brain_sae_similarity_save_root)

        mean_similarity_for_voxels = torch.mean(sae_similarity, dim=0).view(-1).numpy()
        assert max_value is not None and max_layer is not None
        max_value_before = max_value
        max_value = np.where(mean_similarity_for_voxels > max_value_before, mean_similarity_for_voxels, max_value)
        max_layer = np.where(mean_similarity_for_voxels > max_value_before, layer, max_layer)

    max_layer = (max_layer + 1) / all_layers

    mean_layer_mapping_discription = "subj{}/{}_{}/{}_{}/layer_mapping/mean_layer_mapping".format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate) 

    visualize(args, mean_layer_mapping_discription, max_layer)


def max_layer_mapping(
        args: EasyDict, 
        all_layers: int = 12
    ):
    brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, 0, args.autoencoder.rate)
    sae_similarity = torch.load(brain_sae_similarity_save_root)
    max_value = -np.ones(shape=(sae_similarity.shape[-1], ))
    max_layer = np.zeros(shape=(sae_similarity.shape[-1], ))

    for layer in range(all_layers):

        brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, layer, args.autoencoder.rate)
        sae_similarity = torch.load(brain_sae_similarity_save_root)        
        max_similarity_for_voxels, _ = torch.max(sae_similarity, dim=0)
        max_similarity_for_voxels = max_similarity_for_voxels.view(-1).numpy()
        max_value_before = max_value
        max_value = np.where(max_similarity_for_voxels > max_value_before, max_similarity_for_voxels, max_value)
        max_layer = np.where(max_similarity_for_voxels > max_value_before, layer, max_layer)

    max_layer = (max_layer + 1) / all_layers

    mean_layer_mapping_discription = "subj{}/{}_{}/{}_{}/layer_mapping/maxs_layer_mapping".format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, args.autoencoder.rate) 

    visualize(args, mean_layer_mapping_discription, max_layer)
