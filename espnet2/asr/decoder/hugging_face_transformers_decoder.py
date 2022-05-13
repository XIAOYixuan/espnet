#!/usr/bin/env python3
#  2022, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Decoder."""

from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet2.asr.decoder.abs_decoder import AbsDecoder

import copy
import logging
import torch

try:
    from transformers import AutoModelForSeq2SeqLM

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTransformersDecoder(AbsDecoder, BatchScorerInterface):
    """Hugging Face Transformers Decoder.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of encoder attention
        model_name_or_path: Hugging Face Transformers model name
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        model_name_or_path: str,
        lang_token_id: int = -1,
    ):
        assert check_argument_types()
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.model_name_or_path = model_name_or_path

        self.decoder_pretrained_params = copy.deepcopy(self.decoder.state_dict())
        self.lm_head_pretrained_params = copy.deepcopy(self.lm_head.state_dict())

        if encoder_output_size != self.decoder.config.hidden_size:
            self.linear_in = torch.nn.Linear(
                encoder_output_size, self.decoder.config.hidden_size
            )
        else:
            self.linear_in = torch.nn.Identity()

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad: input tensor (batch, maxlen_out, #mels)
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        args = {"return_dict": True}

        if self.decoder.__class__.__name__ == "MBartDecoder":
            ys_in_pad[:, 0] = 2

        args["input_ids"] = ys_in_pad
        mask = (~make_pad_mask(ys_in_lens)).to(ys_in_pad.device).float()
        args["attention_mask"] = mask

        args["encoder_hidden_states"] = self.linear_in(hs_pad)
        hs_mask = (~make_pad_mask(hlens)).to(hs_pad.device).float()
        args["encoder_attention_mask"] = hs_mask

        x = self.decoder(**args).last_hidden_state
        x = self.lm_head(x)

        olens = mask.sum(1).to(torch.int)
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
        Returns:
            y NN output value.
            y.shape is (batch, token)
        """
        decoder_output = self.decoder(
            input_ids=tgt,
            encoder_hidden_states=self.linear_in(memory),
            return_dict=True,
        )
        y = decoder_output.last_hidden_state[:, -1]
        logp = torch.log_softmax(self.lm_head(y), dim=-1)
        return logp

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        logp = self.forward_one_step(ys, xs)

        return logp, states
