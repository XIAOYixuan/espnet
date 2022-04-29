#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Teacher."""

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.teacher.abs_teacher import AbsTeacher
from espnet2.asr.teacher.loss.square_alignment import (
    SquareAlignmentL1Loss,
    SquareAlignmentL2Loss,
    NormalizedSquareAlignmentL1Loss,
    NormalizedSquareAlignmentL2Loss,
)
from espnet2.asr.teacher.loss.rectangular_alignment import (
    RectangularAlignmentLoss,
)
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.train.class_choices import ClassChoices
from espnet2.utils.sized_dict import SizedDict
from copy import deepcopy
from typeguard import check_argument_types
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import logging
import torch

try:
    from transformers import AutoTokenizer, AutoModel

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        l1=torch.nn.L1Loss,
        square_alignment_l1=SquareAlignmentL1Loss,
        square_alignment_l2=SquareAlignmentL2Loss,
        normalized_square_alignment_l1=NormalizedSquareAlignmentL1Loss,
        normalized_square_alignment_l2=NormalizedSquareAlignmentL2Loss,
        rectangular_alignment=RectangularAlignmentLoss,
    ),
    type_check=torch.nn.Module,
    default="l1",
    optional=False,
)


class HuggingFaceTransformersTeacher(AbsTeacher):
    """Hugging Face Transformers Teacher."""

    def __init__(
        self,
        model_name_or_path: str,
        loss: str,
        token_type: str,
        token_list: Union[Path, str, Iterable[str]],
        bpemodel: Union[Path, str, Iterable[str]],
        average_output: bool = False,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        model = AutoModel.from_pretrained(model_name_or_path)

        if model.config.is_encoder_decoder:
            self.teacher_model = model.encoder
        else:
            self.teacher_model = model

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.teacher_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if token_type == "bpe" or token_type == "hugging_face":
            self.tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        else:
            self.tokenizer = build_tokenizer(token_type=token_type)

        self.converter = TokenIDConverter(token_list=token_list)
        self.average_output = average_output

        loss_class = loss_choices.get_class(loss)
        self.loss = loss_class()

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoder_out: Optional[torch.Tensor],
        speech: Optional[torch.Tensor],
        speech_lengths: Optional[torch.Tensor],
        text: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
        lids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Calculate the Teacher output and Student output's distance from it."""

        if self.tokenizer.model == self.teacher_tokenizer.name_or_path:
            sentences = self.teacher_tokenizer.batch_decode(
                text, skip_special_tokens=True
            )
        else:
            sentences = []
            for i in range(text.shape[0]):
                token_int = text[i][: text_lengths[i]].tolist()
                token = self.converter.ids2tokens(token_int)
                sentences.append(self.tokenizer.tokens2text(token))

        # Tokenize sentences
        encoded_input = self.teacher_tokenizer(
            sentences, padding=True, return_tensors="pt"
        )

        if lids is not None:
            encoded_input["input_ids"][:, 0:1] = lids

        encoded_input["attention_mask"] = encoded_input["attention_mask"].to(
            self.teacher_model.device
        )
        encoded_input["input_ids"] = encoded_input["input_ids"].to(
            self.teacher_model.device
        )

        encoded_input["return_dict"] = True

        # Compute token embeddings
        with torch.no_grad():
            teacher_model_output = self.teacher_model(**encoded_input).last_hidden_state

        if self.average_output:
            # Perform pooling
            teacher_model_output = mean_pooling(
                teacher_model_output, encoded_input["attention_mask"]
            )
            # Normalize embeddings
            teacher_model_output = torch.nn.functional.normalize(
                teacher_model_output, p=2, dim=1
            )
            teacher_model_output = teacher_model_output.unsqueeze(1)

        loss = self.loss(encoder_out, teacher_model_output)

        return loss


# From https://huggingface.co/sentence-transformers/all-mpnet-base-v2
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    )
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
