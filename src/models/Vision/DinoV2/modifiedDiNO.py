import torch
from torch import nn
import numpy as np
import dinov2
from dinov2.dinov2.models.vision_transformer import DinoVisionTransformer


"""
get_tokens() need to be implemented by user.
"""
class ModifiedDiNOv2(nn.Module):
    def __init__(self,  model_path="dinov2_vitb14.pth", ver="dinov2_vitb14", **kwargs) -> None:
        super().__init__()
        vision_model = torch.hub.load("/home/brainai1/anaconda3/envs/brainnet/lib/python3.9/site-packages/dinov2", ver, source="local")
        vision_model.load_state_dict(torch.load(model_path))
        self.vision_model: DinoVisionTransformer = vision_model
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()

    def encode_information(self, image):
        x = self.vision_model.prepare_tokens_with_masks(image)
        global_tokens = {}
        global_tokens["original"] = x.clone()[:, 0, :]
        for i, blk in enumerate(self.vision_model.blocks):
            x = blk(x)
            saved_x = x.clone()
            global_tokens[i] = saved_x[:, 0, :]  # [B, C]
        
        return x, global_tokens

    def encoder_multilayer_information(self, image: torch.Tensor, target_layer):
        info_dict = {}
        x = self.vision_model.prepare_tokens_with_masks(image)
        for i, blk in enumerate(self.vision_model.blocks):
            x = blk(x)
            if i in target_layer:
                info_dict[i] = x.detach().permute(1, 0, 2)
        return x, info_dict
    
    def encoder_undetached_multilayer_information(self, image: torch.Tensor, target_layer):
        info_dict = {}
        x = self.vision_model.prepare_tokens_with_masks(image)
        for i, blk in enumerate(self.vision_model.blocks):
            x = blk(x)
            if i in target_layer:
                info_dict[i] = x.permute(1, 0, 2)
        return x, info_dict
    
    def getVisualDim(self, target_layer=None):
        return 768
    
    def forward(self, image):
        x = self.vision_model.prepare_tokens_with_masks(image)
        for i, blk in enumerate(self.vision_model.blocks):
            x = blk(x)
        return x
