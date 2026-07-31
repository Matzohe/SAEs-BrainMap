import toml
from easydict import EasyDict

from src.SAEs.trainer.MultiLayerVisualTrainer import MultiLayerVisualTrainer


config_dict = toml.load('config.toml')
args = EasyDict(config_dict)
args.autoencoder.name = "original"
args.exp.model_name = "clip_vit-b_16"
args.exp.device = 'cuda:0'
args.autoencoder.tied = "True"
args.autoencoder.rate = 16
args.autoencoder.k = 512
args.autoencoder.batch_size = 512
args.autoencoder.epochs = 5
for target_layer_list in [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]:
    MultiLayerVisualTrainer(args, target_layer_list)
