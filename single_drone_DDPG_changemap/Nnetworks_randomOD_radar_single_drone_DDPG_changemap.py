# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2023 10:34 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import math
import torch
from torch import Tensor
from typing import Optional, Tuple
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.distributions import Normal
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.

     Args:
         hidden_dim (int): dimesion of hidden state vector

     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.

     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn


class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        hidden_dim (int): dimesion of hidden state vector
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn, smoothing
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    def __init__(self, hidden_dim: int, smoothing: bool = True) -> None:
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + conv_attn
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)  # Bx1xT X BxTxD => Bx1xD => BxD

        return context, attn


class MultiHeadLocationAwareAttention(nn.Module):
    """
    Applies a multi-headed location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    In the above paper applied a signle head, but we applied multi head concept.

    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The number of out channel in convolution

    Inputs: query, value, prev_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **prev_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, conv_out_channel: int = 10) -> None:
        super(MultiHeadLocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.conv1d = nn.Conv1d(num_heads, conv_out_channel, kernel_size=3, padding=1)
        self.loc_proj = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.value_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.score_proj = nn.Linear(self.dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(self.dim).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len = value.size(0), value.size(1)

        if last_attn is None:
            last_attn = value.new_zeros(batch_size, self.num_heads, seq_len)

        loc_energy = torch.tanh(self.loc_proj(self.conv1d(last_attn).transpose(1, 2)))
        loc_energy = loc_energy.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, seq_len, self.dim)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.dim).permute(0, 2, 1, 3)
        query = query.contiguous().view(-1, 1, self.dim)
        value = value.contiguous().view(-1, seq_len, self.dim)

        score = self.score_proj(torch.tanh(value + query + loc_energy + self.bias)).squeeze(2)
        attn = F.softmax(score, dim=1)

        value = value.view(batch_size, seq_len, self.num_heads, self.dim).permute(0, 2, 1, 3)
        value = value.contiguous().view(-1, seq_len, self.dim)

        context = torch.bmm(attn.unsqueeze(1), value).view(batch_size, -1, self.num_heads * self.dim)
        attn = attn.view(batch_size, self.num_heads, -1)

        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)

        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _compute_relative_positional_encoding(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class CustomizingAttention(nn.Module):
    r"""
    Customizing Attention

    Applies a multi-head + location-aware attention mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    I combined these two attention mechanisms as custom.

    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )
        conv_out_channel (int): The dimension of convolution

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s alignment

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch * num_heads, v_len): tensor containing the alignment from the encoder outputs.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, conv_out_channel: int = 10) -> None:
        super(CustomizingAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.dim)
        self.conv1d = nn.Conv1d(1, conv_out_channel, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=True)
        self.value_proj = nn.Linear(hidden_dim, self.dim * num_heads, bias=False)
        self.loc_proj = nn.Linear(conv_out_channel, self.dim, bias=False)
        self.bias = nn.Parameter(torch.rand(self.dim * num_heads).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, value: Tensor, last_attn: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, q_len, v_len = value.size(0), query.size(1), value.size(1)

        if last_attn is None:
            last_attn = value.new_zeros(batch_size * self.num_heads, v_len)

        loc_energy = self.get_loc_energy(last_attn, batch_size, v_len)  # get location energy

        query = self.query_proj(query).view(batch_size, q_len, self.num_heads * self.dim)
        value = self.value_proj(value).view(batch_size, v_len, self.num_heads * self.dim) + loc_energy + self.bias

        query = query.view(batch_size, q_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        value = value.view(batch_size, v_len, self.num_heads, self.dim).permute(2, 0, 1, 3)
        query = query.contiguous().view(-1, q_len, self.dim)
        value = value.contiguous().view(-1, v_len, self.dim)

        context, attn = self.scaled_dot_attn(query, value)
        attn = attn.squeeze()

        context = context.view(self.num_heads, batch_size, q_len, self.dim).permute(1, 2, 0, 3)
        context = context.contiguous().view(batch_size, q_len, -1)

        return context, attn

    def get_loc_energy(self, last_attn: Tensor, batch_size: int, v_len: int) -> Tensor:
        conv_feat = self.conv1d(last_attn.unsqueeze(1))
        conv_feat = conv_feat.view(batch_size, self.num_heads, -1, v_len).permute(0, 1, 3, 2)

        loc_energy = self.loc_proj(conv_feat).view(batch_size, self.num_heads, v_len, self.dim)
        loc_energy = loc_energy.permute(0, 2, 1, 3).reshape(batch_size, v_len, self.num_heads * self.dim)

        return loc_energy

class ActorNetwork(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork, self).__init__()

        # self.n_heads = 3
        # self.single_head_dim = int((64+64+64) / self.n_heads)

        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 128), nn.ReLU())
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        # self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())

        # gru layer
        # self.gru = nn.GRU(256, 256, 1, batch_first=True)

        self.own_fc_lay2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                         nn.Linear(64, 64), nn.ReLU())
        # self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.own_fc_outlay = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())
        # self.intrude_fc = nn.Sequential(nn.Linear(6, 256), nn.ReLU())
        # self.own_grid_fc = nn.Sequential(nn.Linear(actor_dim[1], 128), nn.ReLU())

        # perform a self-attention for own obs_grids, actor_obs[1], assume actor_obs = [6, 6, 6]

        # self.grid_K = nn.Linear(actor_obs[1], 64)
        # self.grid_Q = nn.Linear(actor_obs[1], 64)
        # self.grid_V = nn.Linear(actor_obs[1], 64)

        # self.own_grid = nn.Sequential(nn.Linear(actor_obs[1], 64), nn.ReLU())

        # self.surr_drone = nn.Sequential(nn.Linear(max_nei_num * actor_obs[2], 64), nn.ReLU(),
        #                                 nn.Linear(64, 64), nn.ReLU())
        # self-attention for 2D grids that has arbitrary length after flattened, with 2 head.

        # # use attention here
        # self.combine_att_xe = nn.Sequential(nn.Linear(64+64+64, 128), nn.ReLU())
        # # NOTE: For the com_k,q,v here, they are used for "single head" attention calculation, so we only use
        # # dimension of the single_head_dim, in the linear transformation.
        #
        # self.com_k = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_v = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        #
        # self.multi_att_out = nn.Linear(self.n_heads * self.single_head_dim, 128)
        #
        # self.action_out = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        #
        # self.action_out_V3 = nn.Sequential(nn.Linear(64+64+64, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        # self.action_out_V4 = nn.Sequential(nn.Linear(64+64, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        # self.action_out_V5 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, n_actions), nn.Tanh())
        # self.action_out_V5_1 = nn.Sequential(nn.Linear(128+128+128, 128), nn.ReLU(), nn.Linear(128, n_actions), nn.Tanh())
        # self.action_out_V5_1 = nn.Sequential(nn.Linear(256+256, 256), nn.ReLU(), nn.Linear(256, n_actions), nn.Tanh())
        # self.action_out_V5_2 = nn.Sequential(nn.Linear(128+128+128, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
        #                                      nn.Linear(512, n_actions), nn.Tanh())

        # attention for NN
        # self.k = nn.Linear(256, 256, bias=False)
        # self.q = nn.Linear(256, 256, bias=False)
        # self.v = nn.Linear(256, 256, bias=False)

        # self.name = name

        # self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #
        # self.to(self.device)

    def forward(self, state):
        own_e = self.own_fc(state[0])

        # gru_output, hn = self.gru(own_e)  # h0 is not necessary, as it auto set to 0.

        own_hidden = self.own_fc_lay2(own_e)
        # own_hidden = self.own_fc_lay2(hn)

        action_out = self.own_fc_outlay(own_hidden)
        # x_e = self.intrude_fc(state[1])
        # #
        # # # mask attention embedding
        # q = self.q(own_e)
        # k = self.k(x_e)
        # v = self.v(x_e)
        # mask = state[1].mean(axis=2, keepdim=True).bool()
        # score = torch.bmm(k, q.unsqueeze(axis=2))
        # score_mask = score.clone()  # clone操作很必要
        # score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度
        # #
        # alpha = F.softmax(score_mask / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        # alpha_mask = alpha.clone()
        # alpha_mask[~mask] = 0
        # v_att = torch.sum(v * alpha_mask, axis=1)
        # #
        # concat = torch.cat((own_hidden, v_att), dim=1)
        # action_out = self.action_out_V5_1(concat)
        return action_out


class ActorNetwork_TwoPortion(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork_TwoPortion, self).__init__()

        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(128, n_actions), nn.Tanh())

    def forward(self, cur_state):
        own_obs = self.own_fc(cur_state[0])
        own_grid = self.own_grid(cur_state[1])
        merge_obs_grid = torch.cat((own_obs, own_grid), dim=1)
        merge_feature = self.merge_feature(merge_obs_grid)
        out_action = self.act_out(merge_feature)
        return out_action


class ActorNetwork_OnePortion(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork_OnePortion, self).__init__()

        self.own_fcWgrid = nn.Sequential(nn.Linear(actor_dim[0]+actor_dim[1], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())

    def forward(self, cur_state):
        own_obsWgrid = torch.cat((cur_state[0], cur_state[1]), dim=1)
        obsWgrid_feat = self.own_fcWgrid(own_obsWgrid)
        merge_feature = self.merge_feature(obsWgrid_feat)
        out_action = self.act_out(merge_feature)
        return out_action


class GRU_actor(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRU_actor, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # gru layer
        self.gru = nn.GRU(actor_dim[0], actor_hidden_state_size, 1, batch_first=True)

        self.own_fc_outlay = nn.Sequential(nn.Linear(64+64, n_actions), nn.Tanh())

    def forward(self, cur_state, history_info):
        own_e = self.own_fc(cur_state)
        gru_out, hn = self.gru(history_info)  # hn = last column (or the most recent one) of the output hidden state
        hn_owne = torch.cat((own_e, hn.squeeze(0)),dim=1)
        action_out = self.own_fc_outlay(hn_owne)
        return action_out, hn


class GRUCELL_actor(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRUCELL_actor, self).__init__()
        self.rnn_hidden_dim = actor_hidden_state_size
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.gru_cell = nn.GRUCell(64, actor_hidden_state_size)
        self.own_fc_outlay = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())

    def forward(self, cur_state, history_hidden_state):
        own_e = self.own_fc(cur_state)
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.gru_cell(own_e, h_in)
        action_out = self.own_fc_outlay(h)
        return action_out, h


class GRUCELL_actor_TwoPortion(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRUCELL_actor_TwoPortion, self).__init__()
        # v1 original
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # self.rnn_hidden_dim = actor_hidden_state_size
        # self.gru_cell = nn.GRUCell(64+64, actor_hidden_state_size)
        # self.outlay = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())

        # V1.1
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.BatchNorm1d(64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.BatchNorm1d(64), nn.ReLU())
        # self.rnn_hidden_dim = actor_hidden_state_size
        # self.gru_cell = nn.GRUCell(64, actor_hidden_state_size)
        # # self.outlay = nn.Sequential(nn.Linear(64+64, n_actions), nn.Tanh())
        # # self.outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
        # #                             nn.Linear(64+64, n_actions), nn.Tanh())
        # self.outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
        #                             nn.Linear(128, 128), nn.ReLU(),
        #                             nn.Linear(64+64, n_actions), nn.Tanh())

        # V1.2
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # self.rnn_hidden_dim = actor_hidden_state_size
        # self.gru_cell = nn.GRUCell(64+64, actor_hidden_state_size)
        # self.outlay = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
        #                             nn.Linear(64, n_actions), nn.Tanh())

        # v2
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.gru_cell = nn.GRUCell(actor_dim[1], actor_hidden_state_size)
        self.rnn_hidden_dim = actor_hidden_state_size
        self.own_grid = nn.Sequential(nn.Linear(actor_hidden_state_size, 64), nn.ReLU())
        self.outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(64+64, n_actions), nn.Tanh())


    def forward(self, cur_state, history_hidden_state):
        # V1
        # own_obs = self.own_fc(cur_state[0])
        # own_grid = self.own_grid(cur_state[1])
        # merge_obs_grid = torch.cat((own_obs, own_grid), dim=1)
        # h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        # h_out = self.gru_cell(merge_obs_grid, h_in)
        # action_out = self.outlay(h_out)
        # V1.1
        # own_obs = self.own_fc(cur_state[0])
        # own_grid = self.own_grid(cur_state[1])
        # h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        # h_out = self.gru_cell(own_grid, h_in)
        # merge_obs_H_grid = torch.cat((own_obs, h_out), dim=1)
        # action_out = self.outlay(merge_obs_H_grid)
        # V1.2
        # own_obs = self.own_fc(cur_state[0])
        # own_grid = self.own_grid(cur_state[1])
        # merge_obs_H_grid = torch.cat((own_obs, own_grid), dim=1)
        # h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        # h_out = self.gru_cell(merge_obs_H_grid, h_in)
        # action_out = self.outlay(h_out)

        # V2
        own_obs = self.own_fc(cur_state[0])
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h_out_grid = self.gru_cell(cur_state[1], h_in)
        merge_obs_H_grid = torch.cat((own_obs, h_out_grid), dim=1)
        action_out = self.outlay(merge_obs_H_grid)
        return action_out, h_out_grid


class GRU_batch_actor_TwoPortion(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRU_batch_actor_TwoPortion, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.gru = nn.GRU(actor_dim[1], actor_hidden_state_size, batch_first=True)
        self.rnn_hidden_dim = actor_hidden_state_size
        self.own_grid = nn.Sequential(nn.Linear(actor_hidden_state_size, 64), nn.ReLU())
        self.outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(64+64, n_actions), nn.Tanh())

    def forward(self, cur_state, history_hidden_state):
        own_obs = self.own_fc(cur_state[0])
        h_out_grid, hx = self.gru(cur_state[1])
        merge_obs_H_grid = torch.cat((own_obs, h_out_grid), dim=1)
        action_out = self.outlay(merge_obs_H_grid)
        return action_out, hx


class LSTM_batch_actor_TwoPortion(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(LSTM_batch_actor_TwoPortion, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.lstm = nn.LSTM(actor_dim[1], actor_hidden_state_size, batch_first=True)
        self.rnn_hidden_dim = actor_hidden_state_size
        self.own_grid = nn.Sequential(nn.Linear(actor_hidden_state_size, 64), nn.ReLU())
        self.outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(64+64, n_actions), nn.Tanh())

    def forward(self, cur_state, history_hidden_state):
        own_obs = self.own_fc(cur_state[0])
        h_out_grid, (hn, cn) = self.lstm(cur_state[1], history_hidden_state)
        merge_obs_H_grid = torch.cat((own_obs, h_out_grid), dim=2)
        action_out = self.outlay(merge_obs_H_grid)
        return action_out, (hn, cn)


class GRUCELL_actor_TwoPortion_wATT(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRUCELL_actor_TwoPortion_wATT, self).__init__()

        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.BatchNorm1d(64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.BatchNorm1d(64), nn.ReLU())
        self.rnn_hidden_dim = actor_hidden_state_size
        self.gru_cell = nn.GRUCell(64, actor_hidden_state_size)
        self.outlay = nn.Sequential(nn.Linear(64+128, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, n_actions), nn.Tanh())
        # attention
        self.k = nn.Linear(128, 128, bias=False)
        self.q = nn.Linear(64, 128, bias=False)
        self.v = nn.Linear(128, 128, bias=False)

    def forward(self, cur_state, history_hidden_state):
        own_obs = self.own_fc(cur_state[0])
        own_grid = self.own_grid(cur_state[1])
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h_out = self.gru_cell(own_grid, h_in)
        merge_obs_H_grid = torch.cat((own_obs, h_out), dim=1)
        # mask attention embedding
        q = self.q(own_obs)
        k = self.k(merge_obs_H_grid)
        v = self.v(merge_obs_H_grid)
        score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        v_att = torch.sum(v * alpha, axis=1)
        final_merge = torch.cat((own_obs, v_att), dim=1)
        action_out = self.outlay(final_merge)
        return action_out, h_out


class actor_TwoPortion_wATT(nn.Module):
    def __init__(self, actor_dim, n_actions):
        super(actor_TwoPortion_wATT, self).__init__()
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # self.outlay = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
        #                             nn.Linear(128, n_actions), nn.Tanh())
        # # attention
        # self.k = nn.Linear(128, 128, bias=False)
        # self.q = nn.Linear(128, 128, bias=False)
        # self.v = nn.Linear(128, 128, bias=False)

        # attentionV2
        # self.k = nn.Linear(actor_dim[0]+actor_dim[1], 64, bias=False)
        # self.q = nn.Linear(actor_dim[0]+actor_dim[1], 64, bias=False)
        # self.v = nn.Linear(actor_dim[0]+actor_dim[1], 64, bias=False)
        # self.outlay = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
        #                             nn.Linear(64, n_actions), nn.Tanh())

        # attentionV3
        # self.q = nn.Linear(2, 32, bias=False)
        # self.k = nn.Linear(actor_dim[1], 32, bias=False)
        # self.v = nn.Linear(actor_dim[1], 32, bias=False)
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # self.combine_feature = nn.Sequential(nn.Linear(64+64+32, 256), nn.ReLU())
        # self.outlay = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
        #                             nn.Linear(256, n_actions), nn.Tanh())
        # V4
        # self.q = nn.Linear(actor_dim[0], 64, bias=False)
        # self.k = nn.Linear(actor_dim[1], 64, bias=False)
        # self.v = nn.Linear(actor_dim[1], 64, bias=False)
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
        #                             nn.Linear(128, n_actions), nn.Tanh())
        #V4.1
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
                                    nn.Linear(128, n_actions), nn.Tanh())
        #4.2


    def forward(self, cur_state):
        # own_obs = self.own_fc(cur_state[0])
        # own_grid = self.own_grid(cur_state[1])
        # obs_grid_concate = torch.cat((own_obs, own_grid), dim=1)
        # #attention
        # q = self.q(obs_grid_concate)
        # k = self.k(obs_grid_concate)
        # v = self.v(obs_grid_concate)
        # score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        # alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        # v_att = torch.sum(v * alpha, axis=1)
        # action_out = self.outlay(v_att)

        # obs_grid_concate = torch.cat((cur_state[0], cur_state[1]), dim=1)
        # q = self.q(obs_grid_concate)
        # k = self.k(obs_grid_concate)
        # v = self.v(obs_grid_concate)
        # score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        # alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        # v_att = torch.sum(v * alpha, axis=1)
        # action_out = self.outlay(v_att)
        #V3
        # own_obs = self.own_fc(cur_state[0])
        # own_grid = self.own_grid(cur_state[1])
        # norm_pos = cur_state[0][:, 0:2]
        # q = self.q(norm_pos)
        # k = self.k(cur_state[1])
        # v = self.v(cur_state[1])
        # score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        # alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        # v_att = torch.sum(v * alpha, axis=1)
        # feature_comb = torch.cat((v_att, own_obs, own_grid), dim=1)
        # combine_fea = self.combine_feature(feature_comb)
        # action_out = self.outlay(combine_fea)
        #V4
        # q = self.q(cur_state[0])
        # k = self.k(cur_state[1])
        # v = self.v(cur_state[1])
        # score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        # alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        # v_att = torch.sum(v * alpha, axis=1)
        # own_obs = self.own_fc(cur_state[0])
        # feature_comb = torch.cat((own_obs, v_att), dim=1)
        # action_out = self.outlay(feature_comb)
        #V4.1
        own_obs = self.own_fc(cur_state[0])
        own_grid = self.own_grid(cur_state[1])
        feature_comb = torch.cat((own_obs, own_grid), dim=1)
        action_out = self.outlay(feature_comb)
        return action_out


class GRUCELL_actor_TwoPortion_wATT_v2(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRUCELL_actor_TwoPortion_wATT_v2, self).__init__()
        #v1
        # # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.BatchNorm1d(64), nn.ReLU())
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.BatchNorm1d(64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # self.rnn_hidden_dim = actor_hidden_state_size
        # self.gru_cell = nn.GRUCell(64, actor_hidden_state_size)
        # # self.outlay = nn.Sequential(nn.Linear(64+64, 64+64), nn.ReLU(),
        # #                             nn.Linear(64+64, n_actions), nn.Tanh())
        # self.outlay = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
        #                             nn.Linear(64, n_actions), nn.Tanh())
        # # attention
        # self.k = nn.Linear(64, 64, bias=False)
        # self.q = nn.Linear(64, 64, bias=False)
        # self.v = nn.Linear(64, 64, bias=False)
        #V2
        self.k = nn.Linear(actor_dim[0]+actor_dim[1], 64, bias=False)
        self.q = nn.Linear(actor_dim[0]+actor_dim[1], 64, bias=False)
        self.v = nn.Linear(actor_dim[0]+actor_dim[1], 64, bias=False)
        self.outlay = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                                    nn.Linear(128, n_actions), nn.Tanh())
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.BatchNorm1d(64), nn.ReLU())
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.BatchNorm1d(64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.rnn_hidden_dim = actor_hidden_state_size
        self.gru_cell = nn.GRUCell(64, actor_hidden_state_size)
        # self.outlay = nn.Sequential(nn.Linear(64+64, 64+64), nn.ReLU(),
        #                             nn.Linear(64+64, n_actions), nn.Tanh())
        self.outlay = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                    nn.Linear(64, n_actions), nn.Tanh())
        # attention
        self.k = nn.Linear(64, 64, bias=False)
        self.q = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)


    def forward(self, cur_state, history_hidden_state):
        own_obs = self.own_fc(cur_state[0])
        own_grid = self.own_grid(cur_state[1])
        #attention embedding
        q = self.q(own_obs)
        k = self.k(own_grid)
        v = self.v(own_grid)
        score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        v_att = torch.sum(v * alpha, axis=1)
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        # h_out = self.gru_cell(own_grid, h_in)
        # attWGru = torch.cat((v_att, h_out), dim=1)
        # action_out = self.outlay(attWGru)
        h_out = self.gru_cell(v_att, h_in)
        action_out = self.outlay(h_out)
        return action_out, h_out


class Stocha_actor(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(Stocha_actor, self).__init__()
        init_w = 3e-3
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 256), nn.ReLU())
        self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                         nn.Linear(256, 256), nn.ReLU(),
                                         nn.Linear(256, 256), nn.ReLU())
        self.log_std_min = -20
        self.log_std_max = 2
        self.mean_linear = nn.Linear(256, n_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(256, n_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        own_e = self.own_fc(state[0])
        own_hidden = self.own_fc_lay2(own_e)
        mean = self.mean_linear(own_hidden)
        log_std = self.log_std_linear(own_hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        normal = Normal(0, 1)  # this is the re-parameterization trick
        z = normal.sample(mean.shape)
        action = torch.tanh(mean + std * z.to(device))
        return action


class CriticNetwork(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions):
        super(CriticNetwork, self).__init__()

        # in critic network we should use multi-head attention mechanism to help to capture more complex relationship
        # between different inputs, in the context of this paper, the input consists of many drone's states as well as
        # their actions. This is two group of inputs, therefore my hypothesis is that using multi-head attention is
        # better here.

        # critic_obs[0] is sum of all agent's own observed states
        # critic_obs[1] is sum of all agent's observed grid maps
        # critic_obs[3] is sum of all agent's action taken

        # self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs*n_agents, 1024), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs[0]*n_agents, 256), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.sum_grid_fc = nn.Sequential(nn.Linear(critic_obs[1]*n_agents, 128), nn.ReLU())
        # self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0]+critic_obs[1])*n_agents, 128), nn.ReLU())
        self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0])*n_agents, 256), nn.ReLU())
        # self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0])*n_agents, 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 1024), nn.ReLU())
        # self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0])*n_agents, 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 2048), nn.ReLU())
        # self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0]+critic_obs[1])*n_agents, 256), nn.ReLU())
        # self.sum_combine_actionFC = nn.Sequential(nn.Linear(2*n_agents, 64), nn.ReLU())
        #
        # self.single_own_fc = nn.Sequential(nn.Linear(critic_obs[0], 128), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.single_grid_fc = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.ReLU())
        # self.single_surr = nn.Sequential(nn.Linear(critic_obs[2], 128), nn.ReLU())

        # for surrounding agents' encoding, for each agent, we there are n-neighbours, each neighbour is represented by
        # a vector of length = 6. Before we put into an experience replay, we pad it up to max_num_neigh * 6 array.
        # so, one agent will have an array of max_num_neigh * 6, after flatten, then for one batch, there are a total of
        # n_agents exist in the airspace, therefore, the final dimension will be max_num_neigh * 6 * max_num_neigh.
        # self.sum_sur_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents*n_agents, 256), nn.ReLU())

        # critic attention for overall sur_neighbours with overall own_state
        # self.single_k = nn.Linear(128, 128, bias=False)
        # self.single_q = nn.Linear(128, 128, bias=False)
        # self.single_v = nn.Linear(128, 128, bias=False)
        #
        # self.n_heads = 3
        # self.single_head_dim = int((256+256+256) / self.n_heads)
        # self.com_k = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_v = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.multi_att_out = nn.Sequential(nn.Linear(self.n_heads * self.single_head_dim + n_agents * n_actions, 128),
        #                                    nn.ReLU())
        #
        # self.combine_env_fc = nn.Sequential(nn.Linear(256+256+256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
        #                                     nn.Linear(128, 64), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(256+128, 128), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(128+(2*n_agents), 256), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(1024+(2*n_agents), 512), nn.ReLU())
        self.combine_env_fc = nn.Sequential(nn.Linear(256+(2*n_agents), 256), nn.ReLU(),
                                            nn.Linear(256, 256), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(1024+(2*n_agents), 512), nn.ReLU(),
        #                                     nn.Linear(512, 512), nn.ReLU(),
        #                                     nn.Linear(512, 512), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(2048+(2*n_agents), 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 512), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(128+(2*n_agents), 512), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(256+(2*n_agents), 512), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(128+64, 256), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(30+4, 256), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear((n_agents*128)+(n_agents*128), 64), nn.ReLU())

        # self.combine_all = nn.Sequential(nn.Linear(128+n_agents * n_actions, 64), nn.ReLU(), nn.Linear(64, 1))
        # self.combine_all = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.combine_all = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        # self.combine_all = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        # self.combine_all = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))

        # self.sum_agents_action_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents, 256), nn.ReLU())
        #
        # self.multi_attention = nn.MultiheadAttention(embed_dim=256+256, num_heads=2)  # 1st input is the sum of neurons from actions and combined states encoding
        #
        # # the input of this judgement layer is 256+256 because this is right after the multi-head attention layer
        # # the output dimension of the multi-head attention is default to be the dimension of the "embed_dim"
        # self.judgement_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        # self.name = name

        # self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #
        # self.to(self.device)

    def forward(self, state, actor_actions):  # state[0] is sum of all agent's own observed states, state[1] is is sum of all agent's observed grid maps
        # pre-process, compute attention for every agent based on their surrounding agents
        # attention_all_agent = []
        # grid_all_agent = []
        # own_all_agent = []
        # for one_agent_batch_own, one_agent_batch_grid, one_agent_batch_surr in zip(*state):  # automatically loop over 5 times
        #     single_grid_out = self.single_grid_fc(one_agent_batch_grid)
        #     single_own_out = self.single_own_fc(one_agent_batch_own)
        #
        #     # single_surr_out = self.single_surr(one_agent_batch_surr)
        #     # single_q = self.single_q(single_own_out)
        #     # single_k = self.single_k(single_surr_out)
        #     # single_v = self.single_v(single_surr_out)
        #     # mask = one_agent_batch_surr.mean(axis=2, keepdim=True).bool()
        #     # score = torch.bmm(single_k, single_q.unsqueeze(axis=2))
        #     # score_mask = score.clone()  # clone操作很必要
        #     # score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度
        #     # alpha = F.softmax(score_mask / np.sqrt(single_k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        #     # alpha_mask = alpha.clone()
        #     # alpha_mask[~mask] = 0
        #     # v_att = torch.sum(single_v * alpha_mask, axis=1)
        #     # attention_all_agent.append(v_att)
        #
        #     grid_all_agent.append(single_grid_out)
        #     own_all_agent.append(single_own_out)


        # sum_att = torch.stack(attention_all_agent).transpose(0, 1)
        # sum_att = sum_att.reshape(sum_att.shape[0], -1)

        # sum_grid = torch.stack(grid_all_agent).transpose(0, 1)
        # sum_grid = sum_grid.reshape(sum_grid.shape[0], -1)
        #
        # sum_own = torch.stack(grid_all_agent).transpose(0, 1)
        # sum_own = sum_own.reshape(sum_own.shape[0], -1)

        # sum_own = self.sum_own_fc(state[0])
        # sum_grid = self.sum_grid_fc(state[1])
        #
        # # env_concat = torch.cat((sum_att, sum_grid), dim=1)
        # env_concat = torch.cat((sum_own, sum_grid), dim=1)
        combine_state = self.sum_combine_fc(state)
        # combine_action = self.sum_combine_actionFC(actor_actions)

        # combine_raw_SA = torch.cat((state, actor_actions), dim=1)
        combine_SA = torch.cat((combine_state, actor_actions), dim=1)
        # combine_SA = torch.cat((combine_state, combine_action), dim=1)

        # env_encode = self.combine_env_fc(env_concat)
        env_encode = self.combine_env_fc(combine_SA)
        # entire_comb = torch.cat((env_encode, actor_actions), dim=1)
        # entire_comb = torch.cat((sum_own_e, actor_actions), dim=1)
        q = self.combine_all(env_encode)
        return q


class CriticNetwork_woGru(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, history_horizon_step):
        super(CriticNetwork_woGru, self).__init__()
        self.combine_state_fc = nn.Sequential(nn.Linear((critic_obs[0]) * n_agents, 256), nn.ReLU()) # extract combine state information
        self.combine_hn_fc = nn.Sequential(nn.Linear(history_horizon_step * (critic_obs[0]) * n_agents, 512), nn.ReLU()) # extract history state information
        self.sum_inputs = nn.Sequential(nn.Linear(256+512+(n_actions * n_agents), 512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))  # obtain Q value

    def forward(self, state, actor_actions, history_info):
        combine_state = self.combine_state_fc(state)
        combine_hn = self.combine_hn_fc(history_info)
        combine_S_A_hn = torch.cat((combine_state, combine_hn, actor_actions), dim=1)
        q = self.sum_inputs(combine_S_A_hn)
        return q


class CriticNetwork_wGru(nn.Module):  #
    def __init__(self, critic_obs, n_agents, n_actions, combine_history):
        super(CriticNetwork_wGru, self).__init__()
        self.combine_state_fc = nn.Sequential(nn.Linear((critic_obs[0]) * n_agents, 256), nn.ReLU()) # extract combine state information
        # gru layer
        # self.gru = nn.GRU((critic_obs[0]) * n_agents, 256, 1, batch_first=True)
        self.gru = nn.GRU(critic_obs[0], 256, 1, batch_first=True)

        self.sum_inputs = nn.Sequential(nn.Linear(256+256+(n_actions * n_agents), 512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))  # obtain Q value

    def forward(self, state, actor_actions, history_info):
        combine_state = self.combine_state_fc(state)
        combine_out, combine_hn = self.gru(history_info)
        combine_S_A_hn = torch.cat((combine_state, combine_hn.squeeze(0), actor_actions), dim=1)
        q = self.sum_inputs(combine_S_A_hn)
        return q


class critic_single_obs_wGRU(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_obs_wGRU, self).__init__()
        self.rnn_hidden_dim = hidden_state_size
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.gru_cell = nn.GRUCell(64, hidden_state_size)
        self.own_fc_outlay = nn.Linear(64, 1)

    def forward(self, single_state, single_action, history_hidden_state):
        SA_combine = torch.cat((single_state, single_action), dim=1)
        SA_feature = self.SA_fc(SA_combine)
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.gru_cell(SA_feature, h_in)
        q = self.own_fc_outlay(h)
        return q, h


class critic_single_obs_wGRU_TwoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_obs_wGRU_TwoPortion, self).__init__()
        # V1 original
        # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        # self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        # self.rnn_hidden_dim = hidden_state_size
        # self.gru_cell = nn.GRUCell(64+64, hidden_state_size)
        # # self.own_fc_outlay = nn.Linear(64, 1)
        # self.own_fc_outlay = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
        #                                    nn.Linear(64, 1))
        # V1.1 (GRU to grid)
        # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        # self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        # self.rnn_hidden_dim = hidden_state_size
        # self.gru_cell = nn.GRUCell(64, hidden_state_size)
        # self.own_fc_outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
        #                                    nn.Linear(128, 1))
        # V2 (GRU to grid feature extraction)
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.rnn_hidden_dim = hidden_state_size
        self.gru_cell = nn.GRUCell(critic_obs[1], hidden_state_size)
        self.SA_grid = nn.Sequential(nn.Linear(hidden_state_size, 64), nn.ReLU())
        self.own_fc_outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
                                           nn.Linear(128, 1))

    def forward(self, single_state, single_action, history_hidden_state):
        # obsWaction = torch.cat((single_state[0], single_action), dim=1)
        # own_obsWaction = self.SA_fc(obsWaction)
        # own_grid = self.SA_grid(single_state[1])
        # merge_obs_grid = torch.cat((own_obsWaction, own_grid), dim=1)
        # h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        # h_out = self.gru_cell(merge_obs_grid, h_in)
        # q = self.own_fc_outlay(h_out)

        # V1.1
        # obsWaction = torch.cat((single_state[0], single_action), dim=1)
        # own_obsWaction = self.SA_fc(obsWaction)
        # own_grid = self.SA_grid(single_state[1])
        # h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        # h_out = self.gru_cell(own_grid, h_in)
        # merge_obs_grid = torch.cat((own_obsWaction, h_out), dim=1)
        # q = self.own_fc_outlay(merge_obs_grid)

        # V2
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction = self.SA_fc(obsWaction)
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h_out = self.gru_cell(single_state[1], h_in)
        merge_obs_grid = torch.cat((own_obsWaction, h_out), dim=1)
        q = self.own_fc_outlay(merge_obs_grid)
        return q, h_out


class critic_single_obs_GRU_batch_twoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_obs_GRU_batch_twoPortion, self).__init__()
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.rnn_hidden_dim = hidden_state_size
        self.gru = nn.GRU(critic_obs[1], hidden_state_size)
        self.own_fc_outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
                                           nn.Linear(128, 1))

    def forward(self, single_state, single_action, history_hidden_state):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction = self.SA_fc(obsWaction)
        h_out, hx = self.gru(single_state[1])
        merge_obs_grid = torch.cat((own_obsWaction, h_out), dim=1)
        q = self.own_fc_outlay(merge_obs_grid)
        return q, hx

class critic_single_obs_LSTM_batch_twoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_obs_LSTM_batch_twoPortion, self).__init__()
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.rnn_hidden_dim = hidden_state_size
        self.lstm = nn.LSTM(critic_obs[1], hidden_state_size, batch_first=True)
        self.own_fc_outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
                                           nn.Linear(128, 1))

    def forward(self, single_state, single_action, history_hidden_state):
        obsWaction = torch.cat((single_state[0], single_action), dim=2)
        own_obsWaction = self.SA_fc(obsWaction)
        h_out, (hn, cn) = self.lstm(single_state[1], history_hidden_state)
        merge_obs_grid = torch.cat((own_obsWaction, h_out), dim=2)
        q = self.own_fc_outlay(merge_obs_grid)
        return q, (hn, cn)


class critic_single_obs_TwoPortion_wATT(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions):
        super(critic_single_obs_TwoPortion_wATT, self).__init__()
        # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        # self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        # self.own_fc_outlay = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
        #                                    nn.Linear(128, 1))

        # self.k = nn.Linear(128, 128, bias=False)
        # self.q = nn.Linear(128, 128, bias=False)
        # self.v = nn.Linear(128, 128, bias=False)
        
        # self.k = nn.Linear(critic_obs[0]+critic_obs[1]+n_actions, 128, bias=False)
        # self.q = nn.Linear(critic_obs[0]+critic_obs[1]+n_actions, 128, bias=False)
        # self.v = nn.Linear(critic_obs[0]+critic_obs[1]+n_actions, 128, bias=False)
        # self.own_fc_outlay = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
        #                                    nn.Linear(128, 1))
        # V3
        # self.q = nn.Linear(2, 32, bias=False)
        # self.k = nn.Linear(critic_obs[1], 32, bias=False)
        # self.v = nn.Linear(critic_obs[1], 32, bias=False)
        # self.own_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        # self.combine_feature = nn.Sequential(nn.Linear(64+64+32, 256), nn.ReLU())
        # self.outlay = nn.Sequential(nn.Linear(256, 1))
        #V4
        # self.q = nn.Linear(critic_obs[0]+n_actions, 64, bias=False)
        # self.k = nn.Linear(critic_obs[1], 64, bias=False)
        # self.v = nn.Linear(critic_obs[1], 64, bias=False)
        # self.own_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        # self.combine_feature = nn.Sequential(nn.Linear(64+64, 256), nn.ReLU())
        # self.outlay = nn.Sequential(nn.Linear(256, 1))
        #V4.1
        self.own_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        self.outlay = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, single_state, single_action):
        # obsWaction = torch.cat((single_state[0], single_action), dim=1)
        # own_obsWaction = self.SA_fc(obsWaction)
        # own_grid = self.SA_grid(single_state[1])
        # obsWaction_grid = torch.cat((own_obsWaction, own_grid), dim=1)
        # # attention embedding
        # q = self.q(obsWaction_grid)
        # k = self.k(obsWaction_grid)
        # v = self.v(obsWaction_grid)
        # score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        # alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)
        # v_att = torch.sum(v*alpha, axis=1)
        # q = self.own_fc_outlay(v_att)

        # obsWaction_grid = torch.cat((single_state[0], single_state[1], single_action), dim=1)
        # # attention embedding
        # q = self.q(obsWaction_grid)
        # k = self.k(obsWaction_grid)
        # v = self.v(obsWaction_grid)
        # score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        # alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)
        # v_att = torch.sum(v*alpha, axis=1)
        # q = self.own_fc_outlay(v_att)
        #V3
        # obsWaction = torch.cat((single_state[0], single_action), dim=1)
        # own_obs = self.own_fc(obsWaction)
        # own_grid = self.own_grid(single_state[1])
        # norm_pos = single_state[0][:, 0:2]
        # q = self.q(norm_pos)
        # k = self.k(single_state[1])
        # v = self.v(single_state[1])
        # score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        # alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        # v_att = torch.sum(v * alpha, axis=1)
        # feature_comb = torch.cat((v_att, own_obs, own_grid), dim=1)
        # comb_fea = self.combine_feature(feature_comb)
        # q = self.outlay(comb_fea)
        #V4
        # obsWaction = torch.cat((single_state[0], single_action), dim=1)
        # own_obs = self.own_fc(obsWaction)
        # q = self.q(obsWaction)
        # k = self.k(single_state[1])
        # v = self.v(single_state[1])
        # score = torch.bmm(k.unsqueeze(axis=1), q.unsqueeze(axis=2))
        # alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        # v_att = torch.sum(v * alpha, axis=1)
        # feature_comb = torch.cat((v_att, own_obs), dim=1)
        # comb_fea = self.combine_feature(feature_comb)
        # q = self.outlay(comb_fea)
        #V4.1
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obs = self.own_fc(obsWaction)
        own_grid = self.own_grid(single_state[1])
        feature_comb = torch.cat((own_obs, own_grid), dim=1)
        q = self.outlay(feature_comb)
        return q



class critic_single_obs_wGRU_TwoPortion_TD3(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_obs_wGRU_TwoPortion_TD3, self).__init__()
        # Q1
        self.SA_fc_q1 = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.SA_grid_q1 = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        self.rnn_hidden_dim_q1 = hidden_state_size
        self.gru_cell_q1 = nn.GRUCell(64+64, hidden_state_size)
        self.own_fc_outlay_q1 = nn.Linear(64, 1)

        # Q2
        self.SA_fc_q2 = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.SA_grid_q2 = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        self.rnn_hidden_dim_q2 = hidden_state_size
        self.gru_cell_q2 = nn.GRUCell(64+64, hidden_state_size)
        self.own_fc_outlay_q2 = nn.Linear(64, 1)

    def forward(self, single_state, single_action, history_hidden_state):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)

        own_obsWaction_q1 = self.SA_fc_q1(obsWaction)
        own_grid_q1 = self.SA_grid_q1(single_state[1])
        merge_obs_grid_q1 = torch.cat((own_obsWaction_q1, own_grid_q1), dim=1)
        h_in_q1 = history_hidden_state.reshape(-1, self.rnn_hidden_dim_q1)
        h_q1 = self.gru_cell_q1(merge_obs_grid_q1, h_in_q1)
        q1 = self.own_fc_outlay_q1(h_q1)

        own_obsWaction_q2 = self.SA_fc_q2(obsWaction)
        own_grid_q2 = self.SA_grid_q2(single_state[1])
        merge_obs_grid_q2 = torch.cat((own_obsWaction_q2, own_grid_q2), dim=1)
        h_in_q2 = history_hidden_state.reshape(-1, self.rnn_hidden_dim_q2)
        h_q2 = self.gru_cell_q2(merge_obs_grid_q2, h_in_q2)
        q2 = self.own_fc_outlay_q2(h_q2)
        return q1, h_q1, q2, h_q2

    def q1(self, single_state, single_action, history_hidden_state):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction_q1 = self.SA_fc_q1(obsWaction)
        own_grid_q1 = self.SA_grid_q1(single_state[1])
        merge_obs_grid_q1 = torch.cat((own_obsWaction_q1, own_grid_q1), dim=1)
        h_in_q1 = history_hidden_state.reshape(-1, self.rnn_hidden_dim_q1)
        h_q1 = self.gru_cell_q1(merge_obs_grid_q1, h_in_q1)
        q1 = self.own_fc_outlay_q1(h_q1)
        return q1, h_q1


class critic_single_TwoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_TwoPortion, self).__init__()
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        self.merge_fc_grid = nn.Sequential(nn.Linear(64+64, 256), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(256, 1))

    def forward(self, single_state, single_action):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction = self.SA_fc(obsWaction)
        own_grid = self.SA_grid(single_state[1])
        merge_obs_grid = torch.cat((own_obsWaction, own_grid), dim=1)
        merge_feature = self.merge_fc_grid(merge_obs_grid)
        q = self.out_feature_q(merge_feature)
        return q


class critic_single_TwoPortion_TD3(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_TwoPortion_TD3, self).__init__()
        # Q1 no GRU
        self.SA_fc_q1 = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.SA_grid_q1 = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        self.merge_fc_grid_q1 = nn.Sequential(nn.Linear(64+64, 256), nn.ReLU())
        self.out_feature_q_q1 = nn.Sequential(nn.Linear(256, 1))
        # Q2 no GRU
        self.SA_fc_q2 = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.SA_grid_q2 = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        self.merge_fc_grid_q2 = nn.Sequential(nn.Linear(64+64, 256), nn.ReLU())
        self.out_feature_q_q2 = nn.Sequential(nn.Linear(256, 1))

    def forward(self, single_state, single_action):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)

        own_obsWaction_q1 = self.SA_fc_q1(obsWaction)
        own_grid_q1 = self.SA_grid_q1(single_state[1])
        merge_obs_grid_q1 = torch.cat((own_obsWaction_q1, own_grid_q1), dim=1)
        merge_feature_q1 = self.merge_fc_grid_q1(merge_obs_grid_q1)
        q1 = self.out_feature_q_q1(merge_feature_q1)

        own_obsWaction_q2 = self.SA_fc_q2(obsWaction)
        own_grid_q2 = self.SA_grid_q2(single_state[1])
        merge_obs_grid_q2 = torch.cat((own_obsWaction_q2, own_grid_q2), dim=1)
        merge_feature_q2 = self.merge_fc_grid_q2(merge_obs_grid_q2)
        q2 = self.out_feature_q_q2(merge_feature_q2)
        return q1, q2

    def q1(self, single_state, single_action):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction_q1 = self.SA_fc_q1(obsWaction)
        own_grid_q1 = self.SA_grid_q1(single_state[1])
        merge_obs_grid_q1 = torch.cat((own_obsWaction_q1, own_grid_q1), dim=1)
        merge_feature_q1 = self.merge_fc_grid_q1(merge_obs_grid_q1)
        q1 = self.out_feature_q_q1(merge_feature_q1)
        return q1


class critic_single_OnePortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_OnePortion, self).__init__()
        self.SA_fcWgrid = nn.Sequential(nn.Linear(critic_obs[0]+n_actions+critic_obs[1], 64), nn.ReLU())
        self.merge_fc_grid = nn.Sequential(nn.Linear(64, 256), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(256, 1))

    def forward(self, single_state, single_action):
        obsWactionWgrid = torch.cat((single_state[0], single_action, single_state[1]), dim=1)
        own_obsWactionWgrid = self.SA_fcWgrid(obsWactionWgrid)
        merge_feature = self.merge_fc_grid(own_obsWactionWgrid)
        q = self.out_feature_q(merge_feature)
        return q


class critic_combine_TwoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_combine_TwoPortion, self).__init__()
        # v1 #
        # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+(n_actions*n_agents), 128), nn.ReLU())
        # self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.ReLU())
        # self.merge_fc_grid = nn.Sequential(nn.Linear(128+128, 256), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # end of v1 #

        # v2 #
        self.S_fc = nn.Sequential(nn.Linear(critic_obs[0], 128), nn.ReLU())
        self.grid_fc = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.ReLU())
        self.combine_inputWact = nn.Sequential(nn.Linear(128+128+(n_actions*n_agents), 256), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(256, 1))

    def forward(self, combine_state, combine_action):
        # ---- v1 -----
        # obsWaction = torch.cat((combine_state[0], combine_action), dim=1)  # obs + action
        # own_obsWaction = self.SA_fc(obsWaction)
        # own_grid = self.SA_grid(combine_state[1])  # grid
        # merge_obs_grid = torch.cat((own_obsWaction, own_grid), dim=1)
        # merge_feature = self.merge_fc_grid(merge_obs_grid)
        # q = self.out_feature_q(merge_feature)
        # --- end of v1 ---

        # --- v2 ---
        own_obs = self.S_fc(combine_state[0])
        own_grid = self.grid_fc(combine_state[1])
        combine_obs = torch.cat((own_obs, own_grid, combine_action), dim=1)
        combine_obs_feature = self.combine_inputWact(combine_obs)
        q = self.out_feature_q(combine_obs_feature)
        # --- end of v2 ---
        return q

