import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from typing import Optional

from .utils import causal_mask, upper_lower_diagonal_mask, modulate
from .modules import Transpose, Linear, ResidualConnectionModule, Mlp, PositionalEncoding
from .cnn_block import ConformerConvModule


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask=None):
        b, n, _, h = *queries.shape, self.heads
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        queries = queries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b, n, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return out


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
            mask_selection: str = 'causalMask',
            use_DropKey: bool = False,
            mask_ratio: float = 0.3,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)
        self.mask_selection = mask_selection
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio

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
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if self.use_DropKey:
            m_r = torch.ones_like(score) * self.mask_ratio
            score = score + torch.bernoulli(m_r) * -1e12

        if mask is not None:
            mask = mask.unsqueeze(1)
            # to device
            mask = mask.to(score.device)
            score.masked_fill_(mask, -1e9)
            # score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        # padded_pos_score: [batch_size, num_heads, seq_length1, seq_length2+1]
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout_p: float = 0.1,
                 mask_selection: str = 'causalMask',
                 window_size_max_len: int = 300,
                 causal_mask_diagonal: int = 0,
                 upper_offset: int = 20,
                 lower_offset: int = -20,
                 use_DropKey=False,
                 mask_ratio=0.3):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.pos_embedding = None
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_size_max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.num_heads = num_heads
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p,
                                                    mask_selection=mask_selection,
                                                    use_DropKey=use_DropKey,
                                                    mask_ratio=mask_ratio,
                                                    )
        self.dropout = nn.Dropout(p=dropout_p)
        self.mask_selection = mask_selection
        self.mask = None
        self.causal_mask_diagonal = causal_mask_diagonal
        self.upper_offset = upper_offset
        self.lower_offset = lower_offset

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()

        self.pos_embedding = self.positional_encoding(seq_length)
        self.pos_embedding = self.pos_embedding.repeat(batch_size, 1, 1)

        if self.mask is None:
            if self.mask_selection == 'causalMask':
                self.mask = causal_mask(batch_size, seq_length, self.causal_mask_diagonal)
            elif self.mask_selection == 'diagonal_matrix':
                print(f'upper_offset: {self.upper_offset}, lower_offset: {self.lower_offset}......')
                self.mask = upper_lower_diagonal_mask(batch_size, seq_length, self.upper_offset,
                                                                    self.lower_offset)
            elif self.mask_selection == 'no_mask':
                self.mask = None

        if self.mask is not None:
            if self.mask.shape[0] != batch_size:
                self.mask = self.mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=self.pos_embedding, mask=self.mask)

        return self.dropout(outputs)


class MultiHeadedCrossAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout_p: float = 0.1,
                 mask_selection: str = 'no_mask',
                 window_size_max_len: int = 300,
                 causal_mask_diagonal: int = 0,
                 upper_offset: int = 20,
                 lower_offset: int = -20,
                 use_DropKey=False,
                 mask_ratio=0.3):
        super(MultiHeadedCrossAttentionModule, self).__init__()
        self.pos_embedding = None
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_size_max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.num_heads = num_heads
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p,
                                                    mask_selection=mask_selection,
                                                    use_DropKey=use_DropKey,
                                                    mask_ratio=mask_ratio,
                                                    )
        self.dropout = nn.Dropout(p=dropout_p)
        self.mask_selection = mask_selection
        self.mask = None
        self.causal_mask_diagonal = causal_mask_diagonal
        self.upper_offset = upper_offset
        self.lower_offset = lower_offset

    def forward(self, inputs: Tensor, conds: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()
        assert inputs.size() == conds.size()

        self.pos_embedding = self.positional_encoding(seq_length)
        self.pos_embedding = self.pos_embedding.repeat(batch_size, 1, 1)

        if self.mask is None:
            if self.mask_selection == 'causalMask':
                self.mask = causal_mask(batch_size, seq_length, self.causal_mask_diagonal)
            elif self.mask_selection == 'diagonal_matrix':
                print(f'upper_offset: {self.upper_offset}, lower_offset: {self.lower_offset}......')
                self.mask = upper_lower_diagonal_mask(batch_size, seq_length, self.upper_offset,
                                                                    self.lower_offset)
            elif self.mask_selection == 'no_mask':
                self.mask = None

        if self.mask is not None:
            if self.mask.shape[0] != batch_size:
                self.mask = self.mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        conds = self.layer_norm(conds)
        outputs = self.attention(inputs, conds, conds, pos_embedding=self.pos_embedding, mask=self.mask)

        return self.dropout(outputs)


class MultiHeadedSelfAttentionModule_with_object_condition(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout_p: float = 0.1,
                 mask_selection: str = 'causalMask',
                 window_size_max_len: int = 300,
                 causal_mask_diagonal: int = 1,
                 upper_offset: int = 20,
                 lower_offset: int = -20,
                 use_DropKey=False,
                 mask_ratio=0.3):
        super(MultiHeadedSelfAttentionModule_with_object_condition, self).__init__()
        self.pos_embedding = None
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_size_max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.num_heads = num_heads
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p,
                                                                        mask_selection=mask_selection,
                                                                        use_DropKey=use_DropKey,
                                                                        mask_ratio=mask_ratio,
                                                                        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.mask_selection = mask_selection
        self.mask = None
        self.causal_mask_diagonal = causal_mask_diagonal
        self.upper_offset = upper_offset
        self.lower_offset = lower_offset

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        # input: [B, T_m*2, D_m] object & subject
        batch_size, seq_length, _ = inputs.size()

        self.pos_embedding = self.positional_encoding(seq_length)

        self.pos_embedding = self.pos_embedding.repeat(batch_size, 2, 1)

        if self.mask is None:
            if self.mask_selection == 'causalMask':
                self.mask = causal_mask(batch_size, seq_length // 2, self.causal_mask_diagonal)
                self.mask = self.mask.repeat(1, 2, 2)
            # elif self.mask_selection == 'diagonal_matrix':
            #     print(f'upper_offset: {self.upper_offset}, lower_offset: {self.lower_offset}......')
            #     self.mask = upper_lower_diagonal_mask(batch_size, seq_length, self.upper_offset,
            #                                                         self.lower_offset)
            # elif self.mask_selection == 'no_mask':
            #     self.mask = None

        if self.mask is not None:
            if self.mask.shape[0] != batch_size:
                self.mask = self.mask[0].unsqueeze(0).repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=self.pos_embedding, mask=self.mask)

        return self.dropout(outputs)
