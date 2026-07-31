import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import socket
import os
from datetime import datetime
from tqdm import tqdm

from ...models.load_target_model import load_target_model
from ...DataLoader.ImageNet.ImageNet import ImageNetTrainDataset, ImageNetValDataset
from ...util import check_path
from ..model.OriginalSAEs import OriginalSAE
from ..model.RotarySAEs import RotarySAE
from ..loss import autoencoder_loss
from ..Activation import TopK

def set_seed(seed: int):
    
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_autoencoder(args, feature_dim):
    saes_name = args.autoencoder.name
    if saes_name == 'original':
        return OriginalSAE(args, feature_dim=feature_dim)
    elif saes_name == "topk":
        return OriginalSAE(args, feature_dim=feature_dim, activation=TopK(k=int(args.autoencoder.topk)))
    elif saes_name == "rotary":
        return RotarySAE(args, feature_dim=feature_dim)
    elif saes_name == "topk_rotary":
        return RotarySAE(args, feature_dim=feature_dim, activation=TopK(k=int(args.autoencoder.topk)))  
    else:
        raise NotImplementedError

def load_model(args):
    model_name = args.exp.model_name
    target_model, image_preprocess = load_target_model(model_name)
    return target_model, image_preprocess

# for visual models, the training dataset is performed on Imagenet.
def load_dataloader(args, image_preprocess):
    imagenet_train_root = args.dataset.imagenet_train_root
    imagenet_val_root = args.dataset.imagenet_val_root
    imagenet_caffe_root = args.dataset.imagenet_caffe
    training_dataset = ImageNetTrainDataset(imagenet_train_root, imagenet_caffe_root, image_preprocess)
    valid_dataset = ImageNetValDataset(imagenet_val_root, imagenet_caffe_root, image_preprocess)

    batch_size = int(args.autoencoder.batch_size)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return training_dataloader, valid_dataloader

def create_optimizer(args, saes):
    optimizer = torch.optim.AdamW(saes.parameters(), lr=args.autoencoder.lr)
    return optimizer

def saveCheckPoint(args, saes, layer_index):
    SAEsckpt = args.SAEsckpt.ckpt.format(args.exp.model_name, args.autoencoder.name, layer_index, args.autoencoder.rate)
    check_path(SAEsckpt)
    torch.save(saes.state_dict(), SAEsckpt)

def unit_norm_decoder_(sae) -> None:
    """
    Unit normalize the decoder weights of an sae.
    """
    sae.decoder.weight.data /= sae.decoder.weight.data.norm(dim=0)


def unit_norm_decoder_grad_adjustment_(sae) -> None:
    """project out gradient information parallel to the dictionary vectors - assumes that the decoder is already unit normed"""
    
    cross_info = torch.einsum("bn,bn->n", sae.decoder.weight.data, sae.decoder.weight.grad)
    sae.decoder.weight.grad.addcmul_(sae.decoder.weight.data, cross_info, value=-1)

def VisualTrainer(args):
    set_seed(int(args.exp.seed))
    device = args.exp.device
    target_layer_index = args.autoencoder.target_layer
    target_model, image_preprocess = load_model(args)
    target_model = target_model.to(device=device).eval()
    target_model.visual.set_target_layer_index(target_layer_index)
    SAE_dtype = eval(args.autoencoder.dtype)
    inference_dtype = next(target_model.parameters()).dtype

    saes = load_autoencoder(args, feature_dim=target_model.getVisualDim())
    saes = saes.to(device=device, dtype=SAE_dtype)

    training_dataloader, valid_dataloader = load_dataloader(args, image_preprocess)
    optimizer = create_optimizer(args, saes)

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    summaryWriter = SummaryWriter(log_dir="runs/VisualSAEs/{}_layer{}_{}".format(args.exp.model_name, target_layer_index, current_time + "_" + socket.gethostname()))

    # 判断是否需要进行模长方向上的梯度裁剪
    clip = args.autoencoder.name == "topk"

    for epoch in range(int(args.autoencoder.epochs)):
        for i, (image, _) in tqdm(enumerate(training_dataloader), total=len(training_dataloader), desc="Epoch {}/{}".format(epoch, int(args.autoencoder.epochs))):
            image = image.to(device=device, dtype=inference_dtype)
            with torch.no_grad():
                _, activation = target_model.encode_information(image)
            activation = activation.to(dtype=SAE_dtype)
            _, middle_activation, reconstruct_activation = saes(activation)
            loss = autoencoder_loss(activation, reconstruct_activation, middle_activation, l1_weight=args.autoencoder.l1_weight)
            loss.backward()
            summaryWriter.add_scalar("Loss/train", loss.detach().cpu(), epoch * len(training_dataloader) + i)
            if clip:
                with torch.no_grad():
                    # normalize the decoder volumn after each mini batch, to restrict the feature's length to 1 and will make a clip to adamgrad
                    # 需要剪除掉列向量相关的梯度
                    unit_norm_decoder_(saes)
                    unit_norm_decoder_grad_adjustment_(saes)
            optimizer.step()
            optimizer.zero_grad()

        VisualEvaluate(saes, valid_dataloader, device, target_model, epoch, SAE_dtype, inference_dtype, summaryWriter)
    saveCheckPoint(args, saes, target_layer_index)

def VisualEvaluate(saes, valid_dataloader, device, target_model, epoch, SAE_dtype, inference_dtype, summaryWriter):
    saes.eval()
    losses = []
    for image, _ in tqdm(valid_dataloader, total=len(valid_dataloader), desc="Epoch {} Valid".format(epoch)):
        image = image.to(device=device, dtype=inference_dtype)
        with torch.no_grad():
            _, activation = target_model.encode_information(image)
        activation = activation.to(dtype=SAE_dtype)
        _, middle_activation, reconstruct_activation = saes(activation)
        loss = autoencoder_loss(activation, reconstruct_activation, middle_activation, l1_weight=0.0)
        losses.append(float(loss.detach().cpu().item()))
    val_loss = float(np.mean(losses))
    summaryWriter.add_scalar("Loss/val", val_loss, epoch)
