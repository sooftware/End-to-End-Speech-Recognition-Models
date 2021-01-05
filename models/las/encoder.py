# -*- coding: utf-8 -*-
# Soohwan Kim @ https://github.com/sooftware/
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
from models.modules import (
    BaseRNN,
    Linear,
    Transpose
)
from models.extractor import (
    VGGExtractor,
    DeepSpeech2Extractor
)


class Listener(BaseRNN):
    """
    Converts low level speech signals into higher level features

    Args:
        input_size (int): size of input
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability (default: 0.3)
        extractor (str): type of CNN extractor (default: vgg)
        device (torch.device): device - 'cuda' or 'cpu'
        activation (str): type of activation function (default: hardtanh)
        mask_conv (bool): flag indication whether apply mask convolution or not

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: encoder_outputs, hidden
        - **encoder_outputs**: tensor containing the encoded features of the input sequence
        - **encoder_log__probs**: tensor containing log probability for ctc loss
    """
    def __init__(
            self,
            input_size: int,                         # size of input
            num_classes: int,                        # number of class
            hidden_dim: int = 512,                   # dimension of RNN`s hidden state
            device: str = 'cuda',                    # device - 'cuda' or 'cpu'
            dropout_p: float = 0.3,                  # dropout probability
            num_layers: int = 3,                     # number of RNN layers
            bidirectional: bool = True,              # if True, becomes a bidirectional encoder
            rnn_type: str = 'lstm',                  # type of RNN cell
            extractor: str = 'vgg',                  # type of CNN extractor
            activation: str = 'hardtanh',            # type of activation function
            mask_conv: bool = False,                 # flag indication whether apply mask convolution or not
            joint_ctc_attention: bool = False        # Use CTC Loss & Cross Entropy Joint Learning
    ) -> None:
        self.mask_conv = mask_conv
        self.extractor = extractor.lower()
        self.joint_ctc_attention = joint_ctc_attention

        if self.extractor == 'vgg':
            input_size = (input_size - 1) << 5 if input_size % 2 else input_size << 5
            super(Listener, self).__init__(
                input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device
            )
            self.conv = VGGExtractor(activation, mask_conv)

        elif self.extractor == 'ds2':
            input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
            input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
            input_size <<= 6
            super(Listener, self).__init__(
                input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device
            )
            self.conv = DeepSpeech2Extractor(activation, mask_conv)

        else:
            raise ValueError("Unsupported Extractor : {0}".format(extractor))

        if self.joint_ctc_attention:
            assert self.mask_conv, "if joint_ctc_attention training, mask_conv should be True"
            self.fc = nn.Sequential(
                nn.BatchNorm1d(self.hidden_dim << 1),
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_p),
                Linear(self.hidden_dim << 1, num_classes, bias=False)
            )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        encoder_log_probs = None
        encoder_output_lengths = None

        if self.mask_conv:
            inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
            conv_feat, encoder_output_lengths = self.conv(inputs, input_lengths)

            batch_size, num_channels, hidden_dim, seq_length = conv_feat.size()
            conv_feat = conv_feat.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

            conv_feat = nn.utils.rnn.pack_padded_sequence(conv_feat, encoder_output_lengths.cpu())
            encoder_outputs, hidden = self.rnn(conv_feat)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs)
            encoder_outputs = encoder_outputs.transpose(0, 1)

        else:
            conv_feat = self.conv(inputs.unsqueeze(1), input_lengths).to(self.device)
            conv_feat = conv_feat.transpose(1, 2)

            batch_size, seq_length, num_channels, hidden_dim = conv_feat.size()
            conv_feat = conv_feat.contiguous().view(batch_size, seq_length, num_channels * hidden_dim)

            if self.training:
                self.rnn.flatten_parameters()

            encoder_outputs, hidden = self.rnn(conv_feat)

        if self.joint_ctc_attention:
            encoder_log_probs = self.fc(encoder_outputs.transpose(1, 2)).log_softmax(dim=2)

        return encoder_outputs, encoder_log_probs, encoder_output_lengths

