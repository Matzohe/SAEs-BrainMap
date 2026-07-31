import open_clip
from open_clip.transformer import VisionTransformer

import torch
from torch import nn

import numpy as np

from einops import rearrange

from typing import List, Optional

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling.sam import Sam

class ModifiedSAM(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sam: Sam = sam_model_registry["vit_b"](checkpoint=None)
        sd = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )
        sam.load_state_dict(sd)

        def new_forward(self, x: torch.Tensor):
            with torch.no_grad():
                x = self.patch_embed(x)
                if self.pos_embed is not None:
                    x = x + self.pos_embed
                global_tokens = {}
                global_tokens["original"] = x.clone().permute(0, 3, 1, 2).mean(dim=(2, 3))
                for i, blk in enumerate(self.blocks):
                    x = blk(x)
                    x_save = x.clone()
                    x_save = x_save.permute(0, 3, 1, 2)
                    global_tokens[i] = x_save.mean(dim=(2, 3))

                return x, global_tokens

        setattr(sam.image_encoder.__class__, "forward", new_forward)

        self.image_encoder = sam.image_encoder
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

    def encode_information(
        self,
        image,
    ):
        with torch.no_grad():
            x = torch.nn.functional.interpolate(image, size=(1024, 1024), mode="bilinear")
        x, global_tokens = self.image_encoder(x)
        return x, global_tokens

    def getVisualDim(self, target_layer=None):
        return 768

    def forward(
        self,
        image,
    ):
        with torch.no_grad():
            x = torch.nn.functional.interpolate(image, size=(1024, 1024), mode="bilinear")
        x, _ = self.image_encoder(x)
        return x