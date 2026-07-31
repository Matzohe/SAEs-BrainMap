import torch
import toml
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import math
from PIL import Image
from easydict import EasyDict
from tqdm import tqdm
from torchvision import transforms
from src.SAEs.sae_loader import load_pretrained_autoencoder
from src.models.load_target_model import load_target_model
from src.util import check_path
from src.dataset.ImageNet.ImageNet import ImageNetTestDataset


def sae_activation_visualize(args, layers=[11], topk_feature=1000, topk_image=100, visualize=True, NACLIP=False, visualize_feature_num=100, visualize_image_num=10):
    device = args.exp.device
    model_name = args.exp.model_name
    target_model, image_preprocess = load_target_model(args.exp.model_name)
    target_model = target_model.to(device=device).eval()
    if NACLIP:
        naclip_model, _ = load_target_model("naclip_vit-b_16")
        naclip_model.visual.set_params(arch = "reduced", attn_strategy = "naclip", gaussian_std = 5.)
        naclip_model = naclip_model.to(device=device).eval()
    dataset  = ImageNetTestDataset(args.dataset.imagenet_test_root, image_preprocess=image_preprocess)
    image_root_list = dataset.root_list
    ImageNetTestTokenSavePath = args.SAEsEvaluation.imagenet_test_token_save_root
    if not os.path.exists(ImageNetTestTokenSavePath.format(model_name, 0)):
        pass
        # extract_test_image_token()
    # 应对多层模型同时提取的情况
    layer_list = []
    topk_index_list = []
    topk_activation_image_index_list = []
    target_layer = []
    target_layer_topk_index_list = []
    target_layer_topk_activation_image_index_list = []
    for layer in layers:
        topk_index_save_root = args.SAEsEvaluation.sae_topk_feature_id_save_root.format(model_name, args.autoencoder.name, args.autoencoder.rate, layer, topk_feature)
        topk_feature_selective_image_index_save_root = args.SAEsEvaluation.sae_topk_feature_selective_image_index_save_root.format(model_name, args.autoencoder.name, args.autoencoder.rate, layer, topk_feature, topk_image)
        try:
            topk_index = torch.load(topk_index_save_root)
            topk_activation_image_index = torch.load(topk_feature_selective_image_index_save_root)
            topk_index_list.append(topk_index)
            topk_activation_image_index_list.append(topk_activation_image_index)
            layer_list.append(layer)
        except:
            target_layer.append(layer)
    if len(target_layer) != 0:
        saes = []
        for layer in target_layer:
            sae = load_pretrained_autoencoder(args, layer=layer)
            sae = sae.to(device=device).eval()
            saes.append(sae)

        all_activation = [[] for _ in range(len(target_layer))]
        with torch.no_grad():
            for batch in tqdm(range(98), desc="selectivity extraction", total=98):
                with h5py.File(ImageNetTestTokenSavePath.format(model_name, batch), "r") as f:
                    evaluating_data = torch.from_numpy(f['token embedding'][target_layer, :, 1:, :]).to(device=device)  # (len(target_layer),1024, 196, 768)
                    f.close()
                for i, layer in enumerate(target_layer):
                    sae = saes[i]
                    activation, _ = sae.encode(evaluating_data[i].squeeze(0))
                    activation = activation.sum(dim=1)
                    all_activation[i].append(activation.cpu())
                del evaluating_data

        for i, layer in enumerate(target_layer):
            target_all_activation = torch.cat(all_activation[i], dim=0)
            mean_activation = target_all_activation.mean(dim=0).view(-1)
            _, topk_index = torch.topk(mean_activation, k=topk_feature, dim=-1)
            target_layer_topk_index_list.append(topk_index)
            topk_index_save_root = args.SAEsEvaluation.sae_topk_feature_id_save_root.format(model_name, args.autoencoder.name, args.autoencoder.rate, layer, topk_feature)
            check_path(topk_index_save_root)
            torch.save(topk_index, topk_index_save_root)

        topk_activation = [[] for _ in range(len(target_layer))]
        with torch.no_grad():
            for batch in tqdm(range(98), desc="Top Activation Extraction", total=98):
                with h5py.File(ImageNetTestTokenSavePath.format(model_name, batch), "r") as f:
                    evaluating_data = torch.from_numpy(f['token embedding'][target_layer, :, 1:, :]).to(device=device)  # (1024, 196, 768)
                    f.close()
                for i, layer in enumerate(target_layer):
                    sae = saes[i]
                    topk_index = target_layer_topk_index_list[i]
                    activation, _ = sae.encode(evaluating_data[i].squeeze(0))
                    activation = activation.mean(dim=1)
                    activation = activation[:, topk_index]
                    topk_activation[i].append(activation.cpu())
                del evaluating_data

        for i, layer in enumerate(target_layer):
            target_topk_activation = torch.cat(topk_activation[i], dim=0)
            image_info = torch.topk(target_topk_activation, k=topk_image, dim=0).indices
            topk_feature_selective_image_index_save_root = args.SAEsEvaluation.sae_topk_feature_selective_image_index_save_root.format(model_name, args.autoencoder.name, args.autoencoder.rate, layer, topk_feature, topk_image)
            check_path(topk_feature_selective_image_index_save_root)
            torch.save(image_info, topk_feature_selective_image_index_save_root)
            target_layer_topk_activation_image_index_list.append(image_info)
    
        del saes

    if topk_activation_image_index_list and target_layer_topk_activation_image_index_list:
        all_target_layer_topk_activation_image_index_list = target_layer_topk_activation_image_index_list.extend(topk_activation_image_index_list)
        all_target_layer_topk_feature_index_list = target_layer_topk_index_list.extend(topk_index_list)
        all_layer = target_layer.extend(layer_list)
    elif not target_layer_topk_activation_image_index_list:
        all_target_layer_topk_activation_image_index_list = topk_activation_image_index_list
        all_target_layer_topk_feature_index_list = topk_index_list
        all_layer = layer_list
    elif not topk_activation_image_index_list:
        all_target_layer_topk_activation_image_index_list = target_layer_topk_activation_image_index_list
        all_target_layer_topk_feature_index_list = target_layer_topk_index_list
        all_layer = target_layer
    else:
        raise RuntimeError("two list could not be empty at the same time")
    
    if visualize:
        for layer_id, image_info in tqdm(enumerate(all_target_layer_topk_activation_image_index_list)):
            layer = all_layer[layer_id]
            topk_index = all_target_layer_topk_feature_index_list[layer_id]
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
            sae = load_pretrained_autoencoder(args, layer=layer)
            sae = sae.to(device=device).eval()
            with torch.no_grad():
                for topk_feature_id in range(visualize_feature_num):
                    if NACLIP:
                        fig, axes = plt.subplots(5, 6, figsize=(12, 15))
                    else:
                        fig, axes = plt.subplots(5, 4, figsize=(6, 15))
                    axes = axes.flatten()
                    for i, id in enumerate(image_info[:visualize_image_num, topk_feature_id]):
                        image_root = image_root_list[id]
                        old_image = Image.open(image_root).convert("RGB")
                        image = image_preprocess(old_image)
                        old_image = transform(old_image)
                        _, image_embedding = target_model.encoder_multilayer_information(image.unsqueeze(0).to(device), target_layer=[i for i in range(args.exp.layers)])
                        image_embedding = image_embedding[layer].squeeze(1)[1:, :]
                        sae_activation, _ = sae.encode(image_embedding)
                        sae_activation = sae_activation[:, topk_index[topk_feature_id]]
                        heatmap_shape = int(math.sqrt(sae_activation.shape[0]))
                        if NACLIP:
                            image_embedding_naclip = naclip_model.encode_image(image.unsqueeze(0).to(device), return_all=True)
                            naclip_sae_activation, _ = sae.encode(image_embedding_naclip.squeeze(0))
                            naclip_sae_activation = naclip_sae_activation[:, topk_index[topk_feature_id]]
                            naclip_heatmap = naclip_sae_activation.view(heatmap_shape, heatmap_shape).detach().cpu().unsqueeze(0).unsqueeze(0)
                            naclip_heatmap_resized = F.interpolate(naclip_heatmap, size=(224, 224), mode='bilinear')
                            naclip_heatmap_resized = (naclip_heatmap_resized - naclip_heatmap_resized.min()) / (naclip_heatmap_resized.max() - naclip_heatmap_resized.min() + 1e-8)
                            naclip_heatmap_resized = (naclip_heatmap_resized * 255).squeeze(0).squeeze(0).numpy().astype(np.uint8)
                        heatmap = sae_activation.view(heatmap_shape, heatmap_shape).detach().cpu().unsqueeze(0).unsqueeze(0)
                        heatmap_resized = F.interpolate(heatmap, size=(224, 224), mode='bilinear')
                        heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
                        heatmap_resized = (heatmap_resized * 255).squeeze(0).squeeze(0).numpy().astype(np.uint8)
                        
                        if NACLIP:
                            ax = axes[i * 3]
                            ax.imshow(old_image)
                            ax.axis("off")
                            ax = axes[i * 3 + 1]
                            ax.imshow(old_image)
                            ax.imshow(heatmap_resized, cmap='jet', alpha=0.3)
                            ax.axis("off")
                            ax = axes[i * 3 + 2]
                            ax.imshow(old_image)
                            ax.imshow(naclip_heatmap_resized, cmap='jet', alpha=0.3)
                            ax.axis("off")
                        else:
                            ax = axes[i * 2]
                            ax.imshow(old_image)
                            ax.axis("off")
                            ax = axes[i * 2 + 1]
                            ax.imshow(old_image)
                            ax.imshow(heatmap_resized, cmap='jet', alpha=0.3)
                            ax.axis("off")
                    image_save_root = args.SAEsEvaluation.sae_topk_feature_selectivity_heatmap_save_root.format(model_name, args.autoencoder.name, args.autoencoder.rate, layer, topk_feature_id, topk_index[topk_feature_id])
                    check_path(image_save_root)
                    plt.savefig(image_save_root, dpi=300)
                    plt.close()
