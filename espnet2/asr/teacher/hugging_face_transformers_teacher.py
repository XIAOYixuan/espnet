#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Teacher."""

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.teacher.abs_teacher import AbsTeacher
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.train.class_choices import ClassChoices
from typeguard import check_argument_types
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import copy
import logging
import torch

try:
    from transformers import AutoTokenizer, AutoModel

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

loss_choices = ClassChoices(
    name="loss",
    classes=dict(l1=torch.nn.L1Loss),
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
        average_output: bool = True,
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

        self.teacher_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.teacher_model = AutoModel.from_pretrained(model_name_or_path)

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        if token_type == "bpe":
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
    ) -> torch.Tensor:
        """Calculate the Teacher output and its distance from the Student output."""

        sentences = []

        for i in range(text.shape[0]):
            token_int = text[i][: text_lengths[i]].tolist()
            token = self.converter.ids2tokens(token_int)
            sentences.append(self.tokenizer.tokens2text(token).capitalize())

        # Tokenize sentences
        encoded_input = self.teacher_tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        encoded_input["attention_mask"] = encoded_input["attention_mask"].to(
            self.teacher_model.device
        )
        encoded_input["input_ids"] = encoded_input["input_ids"].to(
            self.teacher_model.device
        )

        # Compute token embeddings
        with torch.no_grad():
            teacher_model_output = self.teacher_model(**encoded_input)

        if self.average_output:
            # Perform pooling
            teacher_model_output = mean_pooling(
                teacher_model_output, encoded_input["attention_mask"]
            )
            # Normalize embeddings
            teacher_model_output = torch.nn.functional.normalize(
                teacher_model_output, p=2, dim=1
            )

        loss = self.loss(encoder_out, teacher_model_output.unsqueeze(1))

        return loss


# Copy from https://huggingface.co/sentence-transformers/all-mpnet-base-v2
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
