import torch
import h5py
import cv2
import toml
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.load_target_model import load_target_model
from src.dataset.ImageNet.ImageNet import ImageNetTrainDataset, ImageNetValDataset, ImageNetTestDataset
from src.util import check_path

from overcomplete.visualization import overlay_top_heatmaps

config_dict = toml.load('config.toml')
args = EasyDict(config_dict)
args.exp.device = "cuda:1"
device = args.exp.device

train_token_save_root = "/home/brainai1/VDisk2/BrainAi1/Tokens/ImageNetTrain/{}_activations/batch_{}.h5"
val_token_save_root = "/home/brainai1/VDisk2/BrainAi1/Tokens/ImageNetVal/{}_activations/batch_{}.h5"

clip_model, image_preprocess = load_target_model(args.exp.model_name)
clip_model = clip_model.to(device)
train_dataset = ImageNetTrainDataset(args.dataset.imagenet_train_root, args.dataset.imagenet_caffe, image_preprocess)
val_dataset = ImageNetValDataset(args.dataset.imagenet_val_root, args.dataset.imagenet_caffe, image_preprocess=image_preprocess)

train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
check_path(train_token_save_root.format(args.exp.model_name, 0))
check_path(val_token_save_root.format(args.exp.model_name, 0))

for j, (images, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    images = images.to(device)
    save_activation = []
    with torch.no_grad():
        _, middle_activation = clip_model.encoder_multilayer_information(images, target_layer=[i for i in range(12)])
        for i in range(12):
            save_activation.append(middle_activation[i].detach().permute(1, 0, 2).cpu().unsqueeze(0))
            middle_activation[i] = 0
        save_activation = torch.cat(save_activation, dim=0)
        h5_file = h5py.File(train_token_save_root.format(args.exp.model_name, j), "w")
        h5_file.create_dataset(f"token embedding",
                                data=save_activation,
                                compression="gzip")
        h5_file.close()

for j, (images, _) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    images = images.to(device)
    save_activation = []
    with torch.no_grad():
        _, middle_activation = clip_model.encoder_multilayer_information(images, target_layer=[i for i in range(12)])
        for i in range(12):
            save_activation.append(middle_activation[i].detach().permute(1, 0, 2).cpu().unsqueeze(0))
            middle_activation[i] = 0
        
        save_activation = torch.cat(save_activation, dim=0)
        h5_file = h5py.File(val_token_save_root.format(args.exp.model_name, j), "w")
        h5_file.create_dataset(f"token embedding",
                                data=save_activation,
                                compression="gzip")
        h5_file.close()