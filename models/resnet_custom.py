from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch import Tensor
from dropblock import DropBlock2D, LinearScheduler


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152"
]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int,out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
    

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        input_channels: int  =3,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropblock: bool = False,
        dropblock_prob: float = 0.0,
        dropout: bool = False,
        dropout_prob: float = 0.0
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.input_channels = input_channels
        self.dropblock = dropblock
        self.dropblock_prob = dropblock_prob
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        # network layers:
        self.conv1 = nn.Conv2d(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        if self.dropblock:
            self.dropblock2d = LinearScheduler(
                    DropBlock2D(drop_prob=self.dropblock_prob, block_size=3),
                    start_value=0.0,
                    stop_value=self.dropblock_prob,
                    nr_steps=int(25e3)
                )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        if self.dropblock:
            x4 = self.dropblock2d(x4)

        x_avgpool = self.avgpool(x4)
        x_flat = torch.flatten(x_avgpool, 1)

        if self.dropout:
            x_flat = self.dropout_layer(x_flat)

        x_out = self.fc(x_flat)

        return x_out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(arch_name: str,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            input_channels: int = 3,
            num_classes: int = 1000,  # ImageNet-1000
            dropblock: bool = False,
            dropblock_prob: float = 0.0,
            dropout: bool = False,
            dropout_prob: float = 0.0,
            pretrained: bool = False,
            progress: bool = True,
            **kwargs):
    
    model = ResNet(block,
                   layers,
                   input_channels=input_channels,
                   num_classes=num_classes,
                   dropblock=dropblock,
                   dropblock_prob=dropblock_prob,
                   dropout=dropout,
                   dropout_prob=dropout_prob,
                   **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch_name], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(input_channels=3,
             num_classes=1000,
             dropblock=False,
             dropblock_prob=0.0,
             dropout=False,
             dropout_prob=0.0,
             pretrained=False,
             progress=True,
             **kwargs):

    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18',
                   BasicBlock,
                   [2, 2, 2, 2],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


def resnet34(input_channels=3,
             num_classes=1000,
             dropblock=False,
             dropblock_prob=0.0,
             dropout=False,
             dropout_prob=0.0,
             pretrained=False,
             progress=True,
             **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34',
                   BasicBlock,
                   [3, 4, 6, 3],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


def resnet50(input_channels=3,
             num_classes=1000,
             dropblock=False,
             dropblock_prob=0.0,
             dropout=False,
             dropout_prob=0.0,
             pretrained=False,
             progress=True,
             **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50',
                   Bottleneck,
                   [3, 4, 6, 3],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


def resnet101(input_channels=3,
              num_classes=1000,
              dropblock=False,
              dropblock_prob=0.0,
              dropout=False,
              dropout_prob=0.0,
              pretrained=False,
              progress=True,
              **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101',
                   Bottleneck,
                   [3, 4, 23, 3],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


def resnet152(input_channels=3,
              num_classes=1000,
              dropblock=False,
              dropblock_prob=0.0,
              dropout=False,
              dropout_prob=0.0,
              pretrained=False,
              progress=True,
              **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152',
                   Bottleneck,
                   [3, 8, 36, 3],
                   input_channels,
                   num_classes,
                   dropblock,
                   dropblock_prob,
                   dropout,
                   dropout_prob,
                   pretrained,
                   progress,
                   **kwargs)


if __name__ == "__main__":
    # resnet18_model = resnet18()
    # resnet18_model = resnet18(dropblock=True)
    resnet18_model = resnet18(num_classes=10,
                              dropblock=True,
                              dropblock_prob=0.5,
                              dropout=True,
                              dropout_prob=0.3)
    
    print(resnet18_model)