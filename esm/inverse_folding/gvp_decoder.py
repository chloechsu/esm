# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import GVPGraphEmbedding
from .gvp_modules import GVPConvLayer, LayerNorm, GVP
from .gvp_utils import unflatten_graph
from .util import rbf


class GVPDecoder(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.args = args

        self.dictionary = dictionary
        vocab_size = len(dictionary)
        self.embed_token = torch.nn.Embedding(vocab_size, vocab_size)

        node_hidden_dim = (args.node_hidden_dim_scalar,
                args.node_hidden_dim_vector)
        edge_hidden_dim = (args.edge_hidden_dim_scalar + vocab_size,
                args.edge_hidden_dim_vector)

        self.embed_confidence = nn.Linear(16, node_hidden_dim[0])
        
        conv_activations = (F.relu, torch.sigmoid)
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(
                    node_hidden_dim,
                    edge_hidden_dim,
                    autoregressive=True, # Different from GVPEncoder
                    drop_rate=args.dropout,
                    vector_gate=True,
                    attention_heads=0,
                    n_message=3,
                    conv_activations=conv_activations,
                    n_edge_gvps=0,
                    eps=1e-4,
                    layernorm=True,
                ) 
            for i in range(args.num_decoder_layers)
        )

        self.logit_projection = GVP(
            node_hidden_dim, (vocab_size, 0), 
            activations=(None, None), 
            tuple_io=False
        )

    def forward(self, prev_output_tokens, encoder_out):
        node_embeddings = encoder_out['node_embeddings']
        edge_embeddings = encoder_out['edge_embeddings']
        edge_index = encoder_out['edge_index']
        confidence = encoder_out['confidence']
        batch_size = encoder_out['batch_size']

        # Embed confidence scores into node embeddings
        confidence = torch.flatten(confidence, 0, 1)
        rbf_rep = rbf(confidence, 0., 1.)
        encoder_embeddings = node_embeddings = (
            node_embeddings[0] + self.embed_confidence(rbf_rep),
            node_embeddings[1]
        )

        # Embed prev_output_tokens into edge embeddings
        sequence_embeddings = self.embed_token(prev_output_tokens.flatten(0, 1))
        sequence_embeddings = sequence_embeddings[edge_index[0]]
        # Causal masking
        sequence_embeddings[edge_index[0] >= edge_index[1]] = 0
        edge_embeddings = (
            torch.cat([edge_embeddings[0], sequence_embeddings], dim=-1),
            edge_embeddings[1]
        )
        
        for i, layer in enumerate(self.decoder_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings,
                    edge_index, edge_embeddings, encoder_embeddings)

        node_embeddings = unflatten_graph(node_embeddings, batch_size)
        logits = self.logit_projection(node_embeddings)
        logits = logits.permute(0, 2, 1)
        return logits, None
