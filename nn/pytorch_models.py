from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import numpy as np


class LinearThenRelu(torch.nn.Sequential):
    def __init__(self,
                 in_features: int, out_features: int, bias: bool=True):
        super(LinearThenRelu, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        return x


def _np_to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).type(torch.FloatTensor)


def _build_avgpool_weights(kernel_size: Tuple[int, int],
                           num_channels: int) -> np.ndarray:
    kernel_height = kernel_size[0]
    kernel_width = kernel_size[1]
    one_mat = np.ones((kernel_height, kernel_width))
    eye_in_channels = np.eye(num_channels)

    to_split = np.kron(eye_in_channels, one_mat)
    split_once = np.split(to_split, num_channels, axis=1)
    split_twice = [np.split(_, num_channels, axis=0) for _ in split_once]
    w_new_np = np.array(split_twice) / (kernel_width * kernel_height)
    return w_new_np


class FlatAvgPool(torch.nn.Sequential):
    def __init__(self,
                 in_shape: Tuple[int, int, int],
                 kernel_size: Tuple[int, int]):
        super(FlatAvgPool, self).__init__()
        num_in_channels = in_shape[0]
        assert (2, 2) == kernel_size, "Only tested for 2 by 2 avgpooling"
        stride = (2, 2)

        num_out_channels = num_in_channels
        self.in_shape = in_shape
        self.num_out_channels = num_out_channels
        self.conv = torch.nn.Conv2d(num_in_channels,
                                    num_out_channels,
                                    kernel_size,
                                    stride=stride)

        w_new_np = _build_avgpool_weights(kernel_size, num_in_channels)
        w = _np_to_torch(w_new_np)

        b = torch.zeros([num_in_channels])
        self.conv.weight.data = w
        self.conv.bias.data = b

        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

        # for idx, p in enumerate(generator.layers.parameters()):
        #     p.requires_grad = False
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, *self.in_shape)
        x = self.conv(x)
        x = torch.nn.Flatten()(x)
        return x


class FlatConv2d(torch.nn.Sequential):
    def __init__(self,
                 in_shape: Tuple[int, int, int],
                 num_out_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int]):
        super(FlatConv2d, self).__init__()
        num_in_channels = in_shape[0]
        self.conv = torch.nn.Conv2d(num_in_channels,
                                    num_out_channels,
                                    kernel_size,
                                    stride=stride)
        self.in_shape = in_shape
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, *self.in_shape)
        x = self.conv(x)
        x = torch.nn.Flatten()(x)
        return x


class FlatIdent(torch.nn.Sequential):
    def __init__(self,
                 out_features: int):
        super(FlatIdent, self).__init__()
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.Flatten()(x)
        return x


class FlatRelu(torch.nn.Sequential):
    def __init__(self,
                 in_features: int):
        super(FlatRelu, self).__init__()
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.ReLU()(x)
        return x


class Net(torch.nn.Module):
    def __init__(self,
                 layer_list: List[torch.nn.Module]):
        super(Net, self).__init__()
        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def build_relu_layers(input_dim: int,
                      hidden_layer_widths: List[int],
                      output_dim: int,
                      include_bias: bool) -> List[torch.nn.Module]:
    num_layers = len(hidden_layer_widths)
    all_layer_widths = [input_dim] + hidden_layer_widths + [output_dim]
    final_linear_layer = torch.nn.Linear(hidden_layer_widths[-1], output_dim, bias=include_bias)
    layer_list = []
    for i in range(num_layers):
        w0 = all_layer_widths[i]
        w1 = all_layer_widths[i + 1]
        layer_list += [torch.nn.Linear(w0, w1, bias=include_bias), FlatRelu(w1)]
    layer_list += [final_linear_layer]
    return layer_list
