# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from scipy.spatial import transform

from esm.data import Alphabet

from .gvp_encoder import GVPEncoder
from .gvp_decoder import GVPDecoder
from .gvp_utils import unflatten_graph
from .util import rotate, sample, CoordBatchConverter


class GVPGNNModel(nn.Module):
    """
    GVP-GNN inverse folding model.

    Architecture: Geometric GVP-GNN as both encoder and decoder layers.
    """

    def __init__(self, args, alphabet):
        super().__init__()
        encoder = self.build_encoder(args, alphabet)
        decoder = self.build_decoder(args, alphabet)
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, args, src_dict):
        return GVPEncoder(args, dictionary=src_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict):
        return GVPDecoder(args, tgt_dict)

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        prev_output_tokens,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence)
        return self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
        )

    def sample(self, coords, partial_seq=None, temperature=1.0, confidence=None):
        """
        Samples sequences based on greedy sampling (no beam search).
    
        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        return sample(self, coords, partial_seq=partial_seq,
                temperature=temperature, confidence=confidence,
                incremental=False)
