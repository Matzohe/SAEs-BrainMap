import torch
import torch.nn as nn
import toml
import sys
import os
from easydict import EasyDict

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.dataset.Coco.CocoNSDAnalysis import AnalysisDataset
from src.dataset.Coco.coco_dataset import CoCoCaptionTrainDataset, CoCoCaptionValDataset
from src.util import check_path
from src.models.Vision import clip
from src.models.load_target_model import load_target_model
from src.SAEs.sae_loader import load_pretrained_autoencoder
from src.dataset.NSD.NSD_utils import zscore_by_run


class nsdImageDataset(Dataset):
    def __init__(self, args, image_preprocess):
        nsd_behavious_root = args.NSD.nsd_exp_info_save_root.format(args.exp.subj)
        nsd_behavious = torch.load(nsd_behavious_root, weights_only=False)
        self.cocoTrainDataset = CoCoCaptionTrainDataset(args.dataset.coco, preprocess=image_preprocess)
        self.cocoValDataset = CoCoCaptionValDataset(args.dataset.coco, preprocess=image_preprocess)
        self.all_image_root_list = []
        self.independent_image_root = []
        self.same_image_root = []
        for each in nsd_behavious:
            if each['independent']:
                self.independent_image_root.append(each['iamge_root'])
            else:
                self.same_image_root.append(each['image_root'])
            self.all_image_root_list.append(each['image_root'])
        del nsd_behavious, nsd_behavious_root
        self.length = len(self.all_image_root_list)
        self.image_root_list = self.all_image_root_list

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_root = self.image_root_list[index]
        name = image_root.split("/")[-1].split(".")[0]
        if "val" in image_root:
            img, _ = self.cocoValDataset.getFromName(name)
        else:
            img, _ = self.cocoTrainDataset.getFromName(name)
        
        return img
    
    def independentCondition(self):
        self.length = len(self.independent_image_root)
        self.image_root_list = self.independent_image_root

    def sameCondition(self):
        self.length = len(self.same_image_root)
        self.image_root_list = self.same_image_root

def mean_similarity_analysis(args, target_model, image_preprocess):
    sae_list = []
    for layer in range(args.exp.layers):
        sae = load_pretrained_autoencoder(args, layer=layer)
        sae_list.append(sae.to(device=args.exp.device))
    activation_info = [[] for _ in range(args.exp.layers)]
    inference_dtype = eval(args.exp.inference_dtype)
    sae_dtype = eval(args.autoencoder.dtype)
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
                activation_info[layer].append(sae_activation.squeeze(0).cpu())
            brain_activation_list.append(brain_activation)
        activation_info = [torch.cat(layer_info, dim=0) for layer_info in activation_info]
        brain_activation = torch.cat(brain_activation_list, dim=0)
        if torch.isnan(brain_activation).any():
            nan_mask = torch.isnan(brain_activation).sum(dim=-1) == 0
            brain_activation = brain_activation[nan_mask]
            activation_info = [layer_info[nan_mask] for layer_info in activation_info]
        brain_activation = (brain_activation - brain_activation.mean(dim=0))
        brain_activation = brain_activation / (brain_activation.norm(dim=0) + 1e-8)
        for layer in range(args.exp.layers):
            activation_compute = (activation_info[layer] - activation_info[layer].mean(dim=0))
            activation_compute = activation_compute / (activation_compute.norm(dim=0) + 1e-8)
            similarity = activation_compute.T @ brain_activation
            brain_sae_similarity_save_root = args.similarity.brain_sae_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, args.autoencoder.name, layer, args.autoencoder.rate)
            check_path(brain_sae_similarity_save_root)
            torch.save(similarity, brain_sae_similarity_save_root)


def neuron_mean_similarity_analysis(args, target_model, image_preprocess):
    activation_info = [[] for _ in range(args.exp.layers)]
    with torch.no_grad():
        test_dataset = AnalysisDataset(args=args, image_preprocess=image_preprocess, text_preprocess=clip.tokenize)
        test_dataset.IndividualCondition()
        brain_activation_list = []
        for image, _, brain_activation in tqdm(DataLoader(test_dataset, batch_size=512)):
            _, info = target_model.encoder_multilayer_information(image.to(device=args.exp.device), target_layer=[i for i in range(args.exp.layers)])
            for layer in range(args.exp.layers):
                activation_info[layer].append(info[layer][1:, :, :].mean(dim=0).squeeze(0).cpu())
            brain_activation_list.append(brain_activation)
        activation_info = [torch.cat(layer_info, dim=0) for layer_info in activation_info]
        brain_activation = torch.cat(brain_activation_list, dim=0)
        if torch.isnan(brain_activation).any():
            nan_mask = torch.isnan(brain_activation).sum(dim=-1) == 0
            brain_activation = brain_activation[nan_mask]
            activation_info = [layer_info[nan_mask] for layer_info in activation_info]
        brain_activation = (brain_activation - brain_activation.mean(dim=0))
        brain_activation = brain_activation / (brain_activation.norm(dim=0) + 1e-8)
        for layer in range(args.exp.layers):
            activation_compute = (activation_info[layer] - activation_info[layer].mean(dim=0))
            activation_compute = activation_compute / (activation_compute.norm(dim=0) + 1e-8)
            similarity = activation_compute.T @ brain_activation
            brain_sae_similarity_save_root = args.similarity.brain_neuron_similarity_save_root.format(args.exp.subj, args.exp.model_name, args.exp.full_roi, layer)
            check_path(brain_sae_similarity_save_root)
            torch.save(similarity, brain_sae_similarity_save_root)
