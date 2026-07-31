import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights, ViT_H_14_Weights
from torchvision.models import vit_b_16, vit_l_16, vit_h_14
from torchvision.models import list_models, get_model
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)


class ModifiedImgNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        model = get_model("vit_b_16", weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.requires_grad_(False)
        model.eval()
        self.original_model = model
        layers = ["conv_proj"]
        layers.extend([f"encoder.layers.encoder_layer_{i}.add_1" for i in range(12)])
        new_model = create_feature_extractor(model, layers)
        self.model = new_model

    def encode_information(
        self,
        x,
    ):
        em = self.model(x)
        imeb = self.original_model(x)
        out_list = list(em.values())

        local_tokens = {}
        global_tokens = {}
        for i, out in enumerate(out_list):
            if i == 0:
                global_tokens['original'] = out.mean(dim=(2, 3))  # [B, C]
                continue
            global_tokens[i - 1] = out[:, 0, :]  # [B, C]
        return imeb, global_tokens
    
    def encoder_multilayer_information(self, image: torch.Tensor, target_layer):
        info_dict = {}

        em = self.model(image)
        imeb = self.original_model(image)
        out_list = list(em.values())

        for i, out in enumerate(out_list):
            if i == 0:
                continue
            if (i - 1) in target_layer:
                info_dict[i - 1] = out.detach().permute(1, 0, 2)
        return imeb, info_dict
    
    def encoder_undetached_multilayer_information(self, image: torch.Tensor, target_layer):
        info_dict = {}

        em = self.model(image)
        imeb = self.original_model(image)
        out_list = list(em.values())

        for i, out in enumerate(out_list):
            if i == 0:
                continue
            if (i - 1) in target_layer:
                info_dict[i - 1] = out.permute(1, 0, 2)
        return imeb, info_dict

    def getVisualDim(self, target_layer=None):
        return 768
    
    def forward(self, x):
        return self.original_model(x)