import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from  .utils import ResidualAttentionBlock
from .utils import LayerNorm


# Make a Transformer Encoder Block with Residual Attention Block.
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, target_layer_index: int = 11):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        assert target_layer_index <= layers - 1
        self.target_layer_index = target_layer_index

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

    def new_forward(self, x: torch.Tensor):
        for i in range(0, self.target_layer_index + 1):
            x = self.resblocks[i](x)
        return_info = x.detach()
        for i in range(self.target_layer_index + 1, self.layers):
            x = self.resblocks[i](x)
        return x, return_info
    
    def multi_layer_forward(self, x: torch.Tensor, layer_list):
        info_dict = {}
        for i, block in enumerate(self.resblocks):
            x = block(x)
            if i in layer_list:
                info_dict[i] = x.detach()
        return x, info_dict

    def multi_layer_undetached_forward(self, x: torch.Tensor, layer_list):
        info_dict = {}
        for i, block in enumerate(self.resblocks):
            x = block(x)
            if i in layer_list:
                info_dict[i] = x
        return x, info_dict

# Make a Vision Transformer with Residual Attention Block.
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, target_layer_index: int = 11):
        """_summary_

        Args:
            input_resolution (int): Input Image resolution
            patch_size (int): Each piece size
            width (int): How many pieces to split the image
            layers (int): Attention Block number
            heads (int): Multi-head number
            output_dim (int): Output dimension
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5  # scale is used to initialize the model parameters
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, target_layer_index=target_layer_index)
        self.width = width
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def set_target_layer_index(self, target_layer_index):
        assert target_layer_index <= self.transformer.layers - 1
        self.transformer.target_layer_index = target_layer_index

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        original_x = x.detach().cpu()
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x, original_x
    
    def get_info_forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x, info = self.transformer.new_forward(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x, info
    
    def get_multi_layer_info_forward(self, x: torch.Tensor, layer_list):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x, info = self.transformer.multi_layer_forward(x, layer_list=layer_list)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x, info
    
    def get_undetached_multi_layer_info_forward(self, x: torch.Tensor, layer_list):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x, info = self.transformer.multi_layer_undetached_forward(x, layer_list=layer_list)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x, info
    
