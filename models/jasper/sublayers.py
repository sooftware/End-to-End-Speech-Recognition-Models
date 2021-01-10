# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from typing import Tuple, Optional
from torch import Tensor


class MaskConv1d(nn.Conv1d):
    """1D convolution with sequence masking """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
    ) -> None:
        super(MaskConv1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias)

    def get_sequence_lengths(self, seq_lengths):
        return (
            (seq_lengths + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        """
        inputs: BxDxT
        input_lengths: B
        """
        max_length = inputs.size(2)

        indices = torch.arange(max_length).to(input_lengths.dtype).to(input_lengths.device)
        indices = indices.expand(len(input_lengths), max_length)

        mask = indices >= input_lengths.unsqueeze(1)
        inputs = inputs.masked_fill(mask.unsqueeze(1).to(device=inputs.device), 0)

        output_lengths = self.get_sequence_lengths(input_lengths)
        output = super(MaskConv1d, self).forward(inputs)

        del mask, indices

        return output, output_lengths


class JasperBlock(nn.Module):
    """
    Jasper Block: The Jasper Block consists of R Jasper sub-block.

    Args:
        num_sub_blocks (int): number of sub block
        in_channels (int): number of channels in the input feature
        out_channels (int): number of channels produced by the convolution
        kernel_size (int): size of the convolving kernel
        stride (int): stride of the convolution. (default: 1)
        dilation (int): spacing between kernel elements. (default: 1)
        bias (bool): if True, adds a learnable bias to the output. (default: True)
        dropout_p (float): probability of dropout
        activation (str): activation function

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
        - **output**: tensor contains output sequence vector
        - **output**: tensor contains output sequence lengths
    """
    def __init__(
            self,
            num_sub_blocks: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            bias: bool = True,
            dropout_p: float = 0.2,
            activation: str = 'relu',
    ) -> None:
        super(JasperBlock, self).__init__()
        padding = self.get_same_padding(kernel_size, stride, dilation)
        self.layers = nn.ModuleList([
            JasperSubBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                dropout_p=dropout_p,
                activation=activation,
            ) for i in range(num_sub_blocks)
        ])

    def get_same_padding(self, kernel_size: int, stride: int, dilation: int):
        if stride > 1 and dilation > 1:
            raise ValueError("Only stride OR dilation may be greater than 1")
        return (kernel_size // 2) * dilation

    def forward(self, inputs: Tensor, input_lengths: Tensor, residual: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.layers[:-1]:
            inputs, input_lengths = layer(inputs, input_lengths)

        output, output_lengths = self.layers[-1](inputs, input_lengths, residual)

        return output, output_lengths


class JasperSubBlock(nn.Module):
    """
    Jasper sub-block applies the following operations: a 1D-convolution, batch norm, ReLU, and dropout.

    Args:
        in_channels (int): number of channels in the input feature
        out_channels (int): number of channels produced by the convolution
        kernel_size (int): size of the convolving kernel
        stride (int): stride of the convolution. (default: 1)
        dilation (int): spacing between kernel elements. (default: 1)
        padding (int): zero-padding added to both sides of the input. (default: 0)
        bias (bool): if True, adds a learnable bias to the output. (default: False)
        dropout_p (float): probability of dropout
        activation (str): activation function

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
        - **output**: tensor contains output sequence vector
        - **output**: tensor contains output sequence lengths
    """
    supported_activations = {
        'hardtanh': nn.Hardtanh(0, 20, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'gelu': nn.GELU()
    }

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            padding: int = 0,
            bias: bool = False,
            dropout_p: float = 0.2,
            activation: str = 'relu',
    ) -> None:
        super(JasperSubBlock, self).__init__()

        self.conv = MaskConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation
        )
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        self.activation = self.supported_activations[activation]
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor, residual: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        output, output_lengths = self.conv(inputs, input_lengths)
        output = self.batch_norm(output)

        if residual is not None:
            output += residual

        output = self.dropout(self.activation(output))
        del inputs, input_lengths, residual

        return output, output_lengths
