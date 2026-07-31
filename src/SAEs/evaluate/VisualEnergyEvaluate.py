# extract the most likely category use the energy method in arXiv:2502.03714

import torch
import torch.nn as nn
from tqdm import tqdm

from ..sae_loader import load_pretrained_autoencoder
from ...models.load_target_model import load_target_model
from ...dataset.ImageNet.ImageNet import ImageNetTestDataset
from ...util import check_path

def visual_energy_evaluate(args, layers: int):
    # 在Imagenet测试集上，收集每个特征的平均激活强度，看平均激活强度最高的那几个特征，然后进行选择
    # 在这里同时统计每个特征的激活次数，以及总的平均激活强度和激活时的激活强度，同时记录每个特征在哪些图片中存在激活。
    # 输入为一个args，然后记录每个特征的平均激活值 [l, n,], 记录激活时激活强度平均值 [l, n,], 记录激活图片id [l, n, n_image], 激活时的最大energy,每个特征的激活次数[l, n,]
    device = args.exp.device
    
    visual_model, image_preprocess = load_target_model(args.exp.model_name)
    visual_model = visual_model.eval().to(device)

    SAE_dtype = eval(args.autoencoder.dtype)
    inference_dtype = next(visual_model.parameters()).dtype

    test_dataset = ImageNetTestDataset(args.dataset.imagenet_test_root, image_preprocess=image_preprocess)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    model_ckpt = args.SAEsckpt.ckpt
    sae_model_list = []
    for layer in range(layers):
        model = load_pretrained_autoencoder(args, layer=layer)
        model = model.eval().to(device)
        sae_model_list.append(model)
    sae_model_list = nn.ModuleList(sae_model_list)

    # save info
    average_activation = torch.zeros(size=(layers, int(visual_model.getVisualDim() * args.autoencoder.rate))).cpu()
    activated_average_activation = torch.zeros(size=(layers, int(visual_model.getVisualDim() * args.autoencoder.rate))).cpu()
    activated_average_activation_id = torch.zeros(size=(layers, int(visual_model.getVisualDim() * args.autoencoder.rate), len(test_dataset))).cpu()
    activated_average_activation_count = torch.zeros(size=(layers, int(visual_model.getVisualDim() * args.autoencoder.rate))).cpu()
    with torch.no_grad():

        for i, images in tqdm(enumerate(test_dataloader), desc="SAE evaluation", total=len(test_dataloader)):
            _, image_info = visual_model.encoder_multilayer_information(images.to(device=device, dtype=inference_dtype), target_layer=[i for i in range(layers)])

            for layer in range(layers):
                sae = sae_model_list[layer]
                target_layer_info = image_info[layer].to(dtype=SAE_dtype)
                middle_activation, _ = sae.encode(target_layer_info)
                middle_activation = middle_activation.detach().permute(1, 0, 2)
                middle_activation = middle_activation.cpu()

                mask = (middle_activation.permute(0, 2, 1) > 0).any(dim=-1).int()
                activated_average_activation_id[layer, :, int(i * test_dataloader.batch_size):  int(i * test_dataloader.batch_size) + images.shape[0]] = mask.permute(1, 0)
                
                count = mask.sum(dim=0).view(-1)
                activated_average_activation_count[layer] += count

                average_activation[layer] += middle_activation.mean(dim=(0, 1)) / len(test_dataloader)
                activated_average_activation[layer] += middle_activation.sum(dim=(0, 1))
            break
        activated_average_activation = activated_average_activation / (activated_average_activation_count + 1e-9)
        average_activation = average_activation.to(device=device)
        activated_average_activation = activated_average_activation.to(device=device)

        for layer in range(layers):
            sae = sae_model_list[layer]
            decoder_weight = sae.decoder.weight()
            activation_norm = torch.norm(average_activation[layer].unsqueeze(-1) * decoder_weight, dim=-1).view(-1).cpu()
            activated_average_activation_norm = torch.norm(activated_average_activation[layer].unsqueeze(-1) * decoder_weight, dim=-1).view(-1).cpu()
            average_energy_activation_save_root = args.SAEsEvaluation.average_energy_activation_save_root.format(args.exp.model_name, args.autoencoder.name, layer, args.autoencoder.rate)
            activated_average_energy_activation_save_root = args.SAEsEvaluation.activated_average_energy_activation_save_root.format(args.exp.model_name, args.autoencoder.name, layer, args.autoencoder.rate)
            activated_average_energy_activation_id_save_root = args.SAEsEvaluation.activated_average_energy_activation_id_save_root.format(args.exp.model_name, args.autoencoder.name, layer, args.autoencoder.rate)
            activated_average_energy_activation_count_save_root = args.SAEsEvaluation.activated_average_energy_activation_count_save_root.format(args.exp.model_name, args.autoencoder.name, layer, args.autoencoder.rate)
            check_path(average_energy_activation_save_root)
            check_path(activated_average_energy_activation_save_root)
            check_path(activated_average_energy_activation_id_save_root)
            check_path(activated_average_energy_activation_count_save_root)
            torch.save(activation_norm, average_energy_activation_save_root)
            torch.save(activated_average_activation_norm, activated_average_energy_activation_save_root)
            torch.save(activated_average_activation_id[layer], activated_average_energy_activation_id_save_root)
            torch.save(activated_average_activation_count[layer], activated_average_energy_activation_count_save_root)
