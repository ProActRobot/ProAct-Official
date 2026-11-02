import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import copy

from .module.utils import modulate, prob_mask_like
from .module.cnn_block import PoseEncoderConv, PoseDecoderConv, ConformerConvModule
from .module.attention_block import MultiHeadedSelfAttentionModule, CrossAttention
from .module.modules import Transpose, Linear, ResidualConnectionModule, Mlp, PositionalEncoding


def _build_embedding(dim, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
    table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=5000):  # 5000
        super().__init__()
        self.register_buffer(
            "embedding", _build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


class Audio_Encoder(nn.Module):
    def __init__(self, mfcc_dim, latent_dim, cond_dropout_rate=0.1):
        super().__init__()
        self.mfcc_dim = mfcc_dim
        self.latent_dim = latent_dim

        self.audio_encoder = nn.Linear(self.mfcc_dim, self.latent_dim)
        self.dropout = nn.Dropout(p=cond_dropout_rate)

    def forward(self, mfcc_cond):
        aduio_encoded = self.audio_encoder(mfcc_cond)
        aduio_encoded = self.dropout(aduio_encoded)
        return aduio_encoded

class Resnet1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()

        padding = ((kernel_size - 1) * dilation) // 2

        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: [B, C, T]
        return x + self.block(x)

class Audio_Encoder_V2(nn.Module):
    def __init__(self, mfcc_dim, latent_dim, cond_dropout_rate=0.1, num_layers=3, kernel_size=3):
        super().__init__()
        self.mfcc_dim = mfcc_dim
        self.latent_dim = latent_dim



        self.input_proj = nn.Sequential(
            nn.Conv1d(mfcc_dim, latent_dim, kernel_size=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU()
        )


        self.resnet_blocks = nn.ModuleList([
            Resnet1DBlock(latent_dim, kernel_size=kernel_size, dilation=1)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(p=cond_dropout_rate)

    def forward(self, mfcc_cond):

        x = mfcc_cond.transpose(1, 2) # -> [B, mfcc_dim, T]
        x = self.input_proj(x)
        for block in self.resnet_blocks:
            x = block(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)
        return x


class FinalLayer(nn.Module):
    def __init__(self, latent_dim, motion_decoder=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.output = motion_decoder
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.output(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 mlp_ratio: float = 4.0,
                 num_attention_heads: int = 8,
                 attention_dropout_p: float = 0.1,
                 conv_kernel_size: int = 3,
                 conv_dropout_p: float = 0.1,
                 mask_selection: str = 'causalMask',
                 window_size_max_len: int = 150,
                 condition_strategy: str = 'adaptive_layer_norm',
                 causal_mask_diagonal: int = 0,
                 upper_offset: int = 20,
                 lower_offset: int = -20,
                 use_DropKey: bool = False,
                 mask_ratio: float = 0.3,
                 **block_kwargs):
        super().__init__()

        self.condition_strategy = condition_strategy

        self.attn = ResidualConnectionModule(
            module=MultiHeadedSelfAttentionModule(
                d_model=latent_dim,
                num_heads=num_attention_heads,
                dropout_p=attention_dropout_p,
                mask_selection=mask_selection,
                window_size_max_len=window_size_max_len,
                causal_mask_diagonal=causal_mask_diagonal,
                upper_offset=upper_offset,
                lower_offset=lower_offset,
                use_DropKey=use_DropKey,
                mask_ratio=mask_ratio,
            )
        )
        self.big_conv = ResidualConnectionModule(
            module=ConformerConvModule(
                in_channels=latent_dim,
                kernel_size=conv_kernel_size,
                dropout_p=conv_dropout_p,
            )
        )

        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=latent_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.norm1 = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.norm1_object = nn.LayerNorm(latent_dim)

        if self.condition_strategy == 'adaptive_layer_norm':
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(2 * latent_dim),
                nn.Linear(2 * latent_dim, 6 * latent_dim, bias=True),
            )
        elif self.condition_strategy == 'cross_attention':
            self.norm3 = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
            self.condition_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
            self.cross_atten = CrossAttention(dim=latent_dim, heads=num_attention_heads)
        elif self.condition_strategy == 'in_context_conditioning':
            pass
        # elif self.condition_strategy == 'concat_feature':
        #     pass


    def forward(self, x, c_subject, c_object):
        if self.condition_strategy == 'adaptive_layer_norm':

            concat_c = torch.cat([c_subject, c_object], dim=-1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(concat_c).chunk(6, dim=2)

            adaln_1_x = modulate(self.norm1(x), shift_msa, scale_msa)
            x_attn = self.attn(adaln_1_x)

            x_conv = self.big_conv(x_attn)
            x = x + gate_msa * x_conv

            adaln_2_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            x_mlp = self.mlp(adaln_2_x)

            x = x + gate_mlp * x_mlp
        else:
            raise NotImplementedError
        return x


class Gesture_DiT_interaction_7_cfg_mfcc(nn.Module):
    def __init__(self,
                 latent_dim: int = 1280,
                 motion_dim: int = 166,
                 time_emb_dim: int = 1280,
                 window_size_max_len: int = 150,
                 frame_rate: int = 30,
                 block_depth: int = 12,
                 # audio encoder
                 style_encode: bool = True,
                 mfcc_dim: int = 437,
                 cond_dropout_rate: float = 0.1,
                 # motion encoder
                 encoder_kernel_size: int = 3,
                 # Conformer parameters
                 mask_sel: str = 'causalMask',  # 'tridiagonal_matrix', 'no_mask'
                 condition_strategy: str = 'adaptive_layer_norm',
                 num_att_heads: int = 16,
                 attention_dropout_p: float = 0.1,
                 conv_dropout_p: float = 0.1,
                 conv_kernel_size: int = 3,
                 mlp_ratio: float = 4.0,
                 causal_mask_diagonal: int = 0,
                 upper_offset: int = 20,
                 lower_offset: int = -20,
                 use_DropKey: bool = False,
                 mask_ratio: float = 0.3,
                 ):
        super().__init__()

        self.latent_dim = latent_dim
        self.style_encode = style_encode
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio

        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=latent_dim
        )
        # self.audio_encoder = Audio_Encoder(mfcc_dim=mfcc_dim, latent_dim=latent_dim, cond_dropout_rate=cond_dropout_rate)
        self.audio_encoder = Audio_Encoder_V2(mfcc_dim=mfcc_dim, latent_dim=latent_dim, cond_dropout_rate=cond_dropout_rate)
        self.object_start_frame_cond_a = nn.Parameter(torch.zeros(1, 1, latent_dim))


        self.subject_wav_feature_null = nn.Parameter(torch.randn(1, window_size_max_len, latent_dim))
        self.object_wav_feature_null = nn.Parameter(torch.randn(1, window_size_max_len, latent_dim))


        self.state_feature_embeddings = nn.Parameter(torch.randn(3, latent_dim))

        self.motion_dim = motion_dim
        self.motion_encoder = PoseEncoderConv(motion_dim=self.motion_dim, latent_dim=latent_dim, encoder_kernel_size=encoder_kernel_size)
        self.motion_decoder = PoseDecoderConv(motion_dim=self.motion_dim, latent_dim=latent_dim)

        self.blocks = nn.ModuleList([])
        if use_DropKey:
            total_mask_ratio = self.mask_ratio
            decrease_step = total_mask_ratio / block_depth
        for i in range(block_depth):
            self.blocks.append(ConformerBlock(latent_dim=latent_dim,
                                              mlp_ratio=mlp_ratio,
                                              num_attention_heads=num_att_heads,
                                              attention_dropout_p=attention_dropout_p,
                                              conv_dropout_p=conv_dropout_p,
                                              conv_kernel_size=conv_kernel_size,
                                              mask_selection=mask_sel,
                                              window_size_max_len=window_size_max_len,
                                              condition_strategy=condition_strategy,
                                              causal_mask_diagonal=causal_mask_diagonal,
                                              upper_offset=upper_offset,
                                              lower_offset=lower_offset,
                                              use_DropKey=self.use_DropKey,
                                              mask_ratio=self.mask_ratio,
                                              ))
            if use_DropKey:
                self.mask_ratio -= decrease_step
        self.final_layer = FinalLayer(latent_dim, self.motion_decoder)

    def forward(self, inputs, cond_a, cond_m, state, time, processing_state='training', last_time_stamp=999, cfg_cond_drop_prob: tuple[float, float, float] = (0.1, 0.1, 0.1)):
        """
        inputs: [B, T_m, D_m] tensor of subject's gestures seq
        cond_m: [B, T_m, D_m] tensor of object's gestures seq
        condition: [B, T_a, D_a*2] tensor of control seq, such as audio features of subject and object
        time: [B,] tensor of diffusion timesteps
        """
        t = self.diffusion_embedding(time.type(torch.long)).unsqueeze(1)

        x = self.motion_encoder(inputs)

        assert state is not None
        state = state.to(torch.int64)

        bs, seq_len, double_cond_dim = cond_a.shape
        cond_subject, cond_object = cond_a.chunk(2, dim=-1)
        wav_encode_subject, wav_encode_object = self.audio_encoder(mfcc_cond=cond_subject), self.audio_encoder(mfcc_cond=cond_object)
        wav_encode_object = torch.cat([self.object_start_frame_cond_a.repeat(bs, 1, 1), wav_encode_object[:, 0:-1, :]], dim=1)

        state_embeddings = self.state_feature_embeddings[state]
        t = t + state_embeddings

        # create null conditional embedding with conditional dropout
        assert len(cfg_cond_drop_prob) == 3
        cfg_cond_s_a_drop_prob, cfg_cond_o_a_drop_prob, _ = cfg_cond_drop_prob
        keep_mask_subject_a = prob_mask_like((bs,), 1 - cfg_cond_s_a_drop_prob, device=cond_subject.device)
        keep_mask_subject_a_embed = rearrange(keep_mask_subject_a, "b -> b 1 1")
        keep_mask_object_a = prob_mask_like((bs,), 1 - cfg_cond_o_a_drop_prob, device=cond_object.device)
        keep_mask_object_a_embed = rearrange(keep_mask_object_a, "b -> b 1 1")

        # print(keep_mask_subject_a)
        # print(keep_mask_object_a)
        # print(keep_mask_object_m)


        cfg_wav_encode_subject_a = torch.where(keep_mask_subject_a_embed, wav_encode_subject, self.subject_wav_feature_null.to(wav_encode_subject.dtype))
        cfg_wav_encode_object_a = torch.where(keep_mask_object_a_embed, wav_encode_object, self.object_wav_feature_null.to(wav_encode_object.dtype))

        c_subject = t + cfg_wav_encode_subject_a
        c_object = t + cfg_wav_encode_object_a

        for block in self.blocks:
            x = block(x, c_subject, c_object)
        x = self.final_layer(x, c_subject)

        return x


class TextTemporalEncoder(nn.Module):
    def __init__(
        self,
        clip_dim: int = 512,
        latent_dim: int = 1280,
        num_layers: int = 2,
        num_heads: int = 16,
        ff_mult: float = 2.0,
        dropout: float = 0.1,
        max_text_len: int = 77,
        max_time_len: int = 150,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_text_len = max_text_len
        self.max_time_len = max_time_len


        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.token_pos = nn.Embedding(max_text_len, latent_dim)


        self.time_queries = nn.Embedding(max_time_len, latent_dim)

        # 3) TransformerDecoder:tgt=time_queries, memory=text_tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=int(latent_dim * ff_mult),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


        self.text_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, text_emb: torch.Tensor, T: int, text_pad_mask: torch.Tensor = None):
        B, T_text, C = text_emb.shape
        assert T <= self.max_time_len, f"T={T}  max_time_len={self.max_time_len}"
        assert T_text <= self.max_text_len, f"T_text={T_text}  max_text_len={self.max_text_len}"


        x_text = self.text_proj(text_emb)                         # [B, T_text, D]
        pos_idx = torch.arange(T_text, device=x_text.device)
        x_text = x_text + self.token_pos(pos_idx)[None, :, :]


        if text_pad_mask is None:

            text_pad_mask = (text_emb.abs().sum(dim=-1) < 1e-12)  # [B, T_text]


        t_idx = torch.arange(T, device=x_text.device)
        q_time = self.time_queries(t_idx)[None, :, :].expand(B, T, -1)  # [B, T, D]

        # (3) TransformerDecoder:tgt=q_time, memory=x_text
        out = self.decoder(
            tgt=q_time,
            memory=x_text,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=text_pad_mask,
        )
        return self.text_gain * out  # [B, T, D]

class ControlNet_DiT(nn.Module):
    def __init__(self,
                 base_dit: Gesture_DiT_interaction_7_cfg_mfcc,
                 base_dit_checkpoint: str = None,
                 clip_dim: int = 512,
                 text_latent_dim: int = 1280,
                 text_encoder_layers: int = 2,
                 text_length: int = 77,
                 max_time_len: int = 150
                 ):
        super().__init__()

        self.base_dit = base_dit
        self.block_depth = len(base_dit.blocks)
        self.latent_dim = base_dit.latent_dim
        self.text_encoder = TextTemporalEncoder(
            clip_dim=clip_dim,
            latent_dim=text_latent_dim,
            num_layers=text_encoder_layers,
            num_heads=16,
            ff_mult=2.0,
            dropout=0.1,
            max_text_len=text_length,
            max_time_len=max_time_len
        )

        self.res_scales = nn.Parameter(torch.full((self.block_depth,), -3.0))  # sigmoid(-3) ~ 0.047
        # Global modality gates (text strong, audio weak by default)
        self.w_text  = nn.Parameter(torch.tensor(1.0))
        self.w_audio = nn.Parameter(torch.tensor(0.1))

        if base_dit_checkpoint is not None:
            self.load_base_dit(base_dit_checkpoint, strict=True)
        else:
            raise ValueError("Base DiT checkpoint path must be provided.")

        # deep copy the loaded weights for the control branch.
        self.control_net = copy.deepcopy(self.base_dit)
        # Zero-init only the control branch residuals
        self._zero_initialize_control_net()

        self.freeze_base_dit()

    def _zero_initialize_control_net(self):
        for m in self.control_net.modules():
            if hasattr(m, 'adaLN_modulation'):
                for sub in m.adaLN_modulation.modules():
                    if isinstance(sub, nn.Linear):
                        nn.init.zeros_(sub.weight)
                        nn.init.zeros_(sub.bias)

    def freeze_base_dit(self):
        for p in self.base_dit.parameters():
            p.requires_grad = False

    def load_base_dit(self, checkpoint_path: str, strict: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

            self.keys_to_remove = []
            for name, param in checkpoint['model_state_dict'].items():
                if 'model' not in name:
                    self.keys_to_remove.append(name)
            for key in self.keys_to_remove:
                if key in checkpoint['model_state_dict']:
                    del checkpoint['model_state_dict'][key]

            state_dict = checkpoint["model_state_dict"]

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                k = k.replace("model.", "")
                new_state_dict[k] = v
        else:
            raise ValueError("Checkpoint format not recognized.")

        missing, unexpected = self.base_dit.load_state_dict(new_state_dict, strict=strict)
        print(f"[ControlNet] Loaded base_dit from: {checkpoint_path}")
        if missing:
            print(f"[ControlNet] Missing keys: {missing}")
            raise ValueError("Missing keys when loading base_dit checkpoint.")
        if unexpected:
            print(f"[ControlNet] Unexpected keys: {unexpected}")

    def forward(self, inputs, cond_a, cond_t, state, time, processing_state='training', last_time_stamp=999, cfg_cond_drop_prob: tuple[float, float, float] = (0.1, 0.1, 0.1)):
        """
        inputs: [B, T_m, D_m] tensor of subject's gestures seq
        cond_a: [B, T_a, D_a*2] tensor of control seq, such as audio features of subject and object
        cond_t: [B, T_t, D_t] tensor of text features
        time: [B,] tensor of diffusion timesteps
        """
        B, T_m, D_m = inputs.shape
        t = self.base_dit.diffusion_embedding(time.type(torch.long)).unsqueeze(1)

        x = self.base_dit.motion_encoder(inputs)

        assert state is not None
        state = state.to(torch.int64)

        bs, seq_len, double_cond_dim = cond_a.shape
        cond_subject, cond_object = cond_a.chunk(2, dim=-1)
        wav_encode_subject, wav_encode_object = self.base_dit.audio_encoder(mfcc_cond=cond_subject), self.base_dit.audio_encoder(mfcc_cond=cond_object)
        wav_encode_object = torch.cat([self.base_dit.object_start_frame_cond_a.repeat(bs, 1, 1), wav_encode_object[:, 0:-1, :]], dim=1)

        state_embeddings = self.base_dit.state_feature_embeddings[state]
        t = t + state_embeddings

        # create null conditional embedding with conditional dropout
        assert len(cfg_cond_drop_prob) == 3
        cfg_cond_s_a_drop_prob, cfg_cond_o_a_drop_prob, _ = cfg_cond_drop_prob
        keep_mask_subject_a = prob_mask_like((bs,), 1 - cfg_cond_s_a_drop_prob, device=cond_subject.device)
        keep_mask_subject_a_embed = rearrange(keep_mask_subject_a, "b -> b 1 1")
        keep_mask_object_a = prob_mask_like((bs,), 1 - cfg_cond_o_a_drop_prob, device=cond_object.device)
        keep_mask_object_a_embed = rearrange(keep_mask_object_a, "b -> b 1 1")


        cfg_wav_encode_subject_a = torch.where(keep_mask_subject_a_embed, wav_encode_subject, self.base_dit.subject_wav_feature_null.to(wav_encode_subject.dtype))
        cfg_wav_encode_object_a = torch.where(keep_mask_object_a_embed, wav_encode_object, self.base_dit.object_wav_feature_null.to(wav_encode_object.dtype))

        c_subject = t + cfg_wav_encode_subject_a
        c_object = t + cfg_wav_encode_object_a

        text_cond = self.text_encoder(cond_t, T_m)  # [B, T_m, latent_dim]
        c_ctrl_subject = t + text_cond

        c_subject_all = self.w_audio * c_subject + self.w_text * c_ctrl_subject
        c_object_all = c_object

        for i, (base_block, control_block) in enumerate(zip(self.base_dit.blocks, self.control_net.blocks)):
            x_main    = base_block(x, c_subject, c_object) # frozen speech backbone
            # print(f"c_subject_all shape: {c_subject_all.shape}, c_object_all shape: {c_object_all.shape}")
            x_control = control_block(x_main, c_subject_all, c_object_all)  # control residual on current features
            scale     = torch.sigmoid(self.res_scales[i])     # ∈ (0,1)
            x         = x_main + scale * x_control

        x = self.control_net.final_layer(x, c_subject_all)

        return x

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_dit.eval()
        return self

    def eval(self):
        super().eval()
        self.base_dit.eval()
        return self
