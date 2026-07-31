import torch

from .model.OriginalSAEs import OriginalSAE
from .model.RotarySAEs import RotarySAE
from .Activation import TopK

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
    

def load_pretrained_autoencoder(args, layer):
    saes_name = args.autoencoder.name
    ckpt = torch.load(args.SAEsckpt.ckpt.format(args.exp.model_name, saes_name, layer, args.autoencoder.rate), map_location=torch.device('cpu'))
    if saes_name == 'original' or saes_name == "topk":
        return OriginalSAE.from_state_dict(args, ckpt)
    elif saes_name == "rotary" or saes_name == "topk_rotary":
        return RotarySAE.from_state_dict(args, ckpt)
    else:
        raise NotImplementedError