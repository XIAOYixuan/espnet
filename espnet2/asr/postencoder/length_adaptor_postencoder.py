#!/usr/bin/env python3
#  2022, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length Adaptor PostEncoder."""

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from typeguard import check_argument_types
from typing import Tuple

import torch


class LengthAdaptorPostEncoder(AbsPostEncoder):
    """Length Adaptor PostEncoder."""

    def __init__(
        self,
        input_size: int,
        n_layers: int = 0,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        self.output_dim = input_size

        # Length Adaptor as in https://aclanthology.org/2021.acl-long.68.pdf

        layers = []
        for _ in range(n_layers):
            layers.append(torch.nn.Conv1d(input_size, input_size, 2, 2))
            layers.append(torch.nn.ReLU())

        self.length_adaptor = torch.nn.Sequential(*layers)
        self.length_adaptor_ratio = 2**n_layers

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        input = input.permute(0, 2, 1)
        output = self.length_adaptor(input)
        output = output.permute(0, 2, 1)

        output_lengths = input_lengths.div(
            self.length_adaptor_ratio, rounding_mode="floor"
        )

        return output, output_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
