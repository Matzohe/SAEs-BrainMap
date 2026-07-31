import open_clip
from open_clip.transformer import VisionTransformer

import torch
from torch import nn

import numpy as np

from einops import rearrange

from typing import List, Optional
import timm

class ModifiedMAE(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(ModifiedMAE, self).__init__(**kwargs)

        sd = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
        )

        checkpoint_model = sd["model"]
        state_dict = self.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        self.requires_grad_(False)
        self.eval()

    def encode_information(
        self,
        x,
    ):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        global_tokens = {}
        global_tokens['original'] = x.clone()[:, 0, :]
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            saved_x = x.clone()
            global_tokens[i] = saved_x[:, 0, :]  # [B, C]
        return x, global_tokens
    
    def getVisualDim(self, target_layer=None):
        return 768
    
    def encoder_multilayer_information(self, image: torch.Tensor, target_layer):
        info_dict = {}
        B = image.shape[0]
        x = self.patch_embed(image)
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in target_layer:
                info_dict[i] = x.detach().permute(1, 0, 2)
        return x, info_dict
    
    def encoder_undetached_multilayer_information(self, image: torch.Tensor, target_layer):
        info_dict = {}
        B = image.shape[0]
        x = self.patch_embed(image)
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in target_layer:
                info_dict[i] = x.permute(1, 0, 2)
        return x, info_dict

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        return x
