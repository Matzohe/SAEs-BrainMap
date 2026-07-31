import torch
import torch.nn as nn
from tqdm import tqdm
from easydict import EasyDict
from ...SAEs.sae_loader import load_pretrained_autoencoder
from ...dataset.ImageNet.ImageNet import ImageNetTestDataset
from ...models.load_target_model import load_target_model
from ...util import check_path
from ...visualize.umap import UMAPVisualize
import h5py
from torch.utils.data import DataLoader


def ImageNetTestTokenExtraction(
        args: EasyDict,
        batch_size: int = 1024,
    ):
    """
    提取模型每层在imagenet test上的token embedding
    Args:
        args (EasyDict): 参数控制
        batch_size (int, optional): 批次大小. Defaults to 1024.

    Middle save:
        将输出保存到args.SAEsEvaluation.imagenet_test_token_save_root
    
    """

    model_layer_list = [i for i in range(args.exp.layers)]
    device = args.exp.device
    model, image_preprocess = load_target_model(args.exp.model_name)
    model = model.to(device)
    model.eval()
    dataset = ImageNetTestDataset(args.dataset.imagenet_test_root, image_preprocess=image_preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    save_path = args.SAEsEvaluation.imagenet_test_token_save_root
    check_path(save_path.format(args.exp.model_name, 0))

    for i, images in tqdm(enumerate(dataloader), desc="test token extraction", total=len(dataloader)):
        all_information = []
        images = images.to(device=device)
        with torch.no_grad():
            _, info = model.encoder_multilayer_information(images, target_layer=model_layer_list)
            for layer in model_layer_list:
                all_information.append(info[layer].detach().permute(1, 0, 2).unsqueeze(0).cpu())
                info[layer] = 0
            all_information = torch.cat(all_information, dim=0)
            h5_file = h5py.File(save_path.format(args.exp.model_name, i), "w")
            h5_file.create_dataset(f"token embedding",
                                data=all_information,
                                compression="gzip")
            h5_file.close()
            del all_information
