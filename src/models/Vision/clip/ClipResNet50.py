# Clip ResNet50 model
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AttentionPool2d, Bottleneck


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, int(layers[0]))
        self.layer2 = self._make_layer(width * 2, int(layers[1]), stride=2)
        self.layer3 = self._make_layer(width * 4, int(layers[2]), stride=2)
        self.layer4 = self._make_layer(width * 8, int(layers[3]), stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        similarity_matrix = {}
        layer = 0
        def stem(x, similarity_matrix, layer):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
                similarity_matrix[layer] = x.detach().cpu()
                layer += 1
            x = self.avgpool(x)
            return x, layer

        x = x.type(self.conv1.weight.dtype)
        original_x = x.detach().cpu()
        x, layer = stem(x, similarity_matrix, layer)
        x = self.layer1(x)
        similarity_matrix[layer] = x.detach().cpu()
        layer += 1
        x = self.layer2(x)
        similarity_matrix[layer] = x.detach().cpu()
        layer += 1
        x = self.layer3(x)
        similarity_matrix[layer] = x.detach().cpu()
        layer += 1
        x = self.layer4(x)
        similarity_matrix[layer] = x.detach().cpu()
        layer += 1
        x = self.attnpool(x)
        similarity_matrix['original'] = original_x
        return x, similarity_matrix
