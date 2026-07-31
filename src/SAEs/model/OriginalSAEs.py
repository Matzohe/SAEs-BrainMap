from typing import Callable, Any
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from easydict import EasyDict
from typing import Any, Optional

from ..Activation import ACTIVATIONS_CLASSES

def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class OriginalSAE(nn.Module):
    def __init__(self, args, feature_dim: int, activation: nn.Module = nn.ReLU()):
        """
        Args:
            args (): The args, comming from the config.toml
            feature_dim (int): the model middle activation dimension.
        """
        super().__init__()
        self.tied = eval(args.autoencoder.tied)
        self.normalize = eval(args.autoencoder.normalize)

        self.dtype = args.autoencoder.dtype
        self.device = args.exp.device
        self.rate = int(args.autoencoder.rate)

        self.pre_bias = nn.Parameter(torch.zeros(feature_dim))
        self.latent_bias = nn.Parameter(torch.zeros(self.rate * feature_dim))
        self.encoder = nn.Linear(feature_dim, self.rate * feature_dim, bias=False)

        if self.tied:
            self.decoder = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(self.rate * feature_dim, feature_dim, bias=False)

        self.activation = activation

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: Optional[dict[str, Any]] = None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x):
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        return latents_pre_act, latents, recons

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        return sd

    @classmethod
    def from_state_dict(
        cls, args, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "OriginalSAE":
        """_summary_

        Args:
            args (EasyDict): comming from the config.toml, controling the model hyperparameters
            state_dict (dict[str, torch.Tensor]): loading from the save path.
            strict (bool, optional), Defaults to True.

        Returns:
            OriginalSAE: the SAEs model
        """
        _, feature_dim = state_dict["encoder.weight"].shape

        # Retrieve activation
        activation_class_name = state_dict.pop("activation", "ReLU")
        activation_class = ACTIVATIONS_CLASSES.get(activation_class_name, nn.ReLU)
        activation_state_dict = state_dict.pop("activation_state_dict", {})
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(args, feature_dim, activation=activation)
        autoencoder.load_state_dict(state_dict, strict=False)
        return autoencoder
