import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from tqdm import tqdm
import os
import time as time_test
from typing import Any, Optional, Union, List
from functools import partial
import math
import contextlib
from scipy.ndimage import gaussian_filter1d
from diffusers import DDPMScheduler

from .utils.utils import extract, noise_like

# -----------------------------------------------------------------------------
# Utils (from flow_matching_new.py)
# -----------------------------------------------------------------------------

def make_beta_schedule(schedule_type, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule_type == "linear":
        betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2)
    elif schedule_type == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_type == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule_type == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule_type '{schedule_type}' unknown.")
    return betas.numpy()

torch.compiler.cudagraph_mark_step_begin()

class Gesture_Diffusion_double(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 schedule_type: str = "linear",
                 predict_type: str = "eps", # flow matching ignores this but kept for init compatibility
                 n_timestep: int = 1000,
                 linear_start: float = 1e-4,
                 linear_end: float = 8e-2,
                 cosine_s: float = 8e-3,
                 cfg_cond_drop_prob: tuple[float, float, float] = (0.1, 0.1, 0.1),
                 # for training
                 loss_type: str = "l2",
                 l_simple_weight: float = 1.0,
                 original_elbo_weight: float = 0.0,
                 # for inference
                 clip_denoised: bool = False,
                 save_intermediates_every_t: int = 10
                 ):
        super(Gesture_Diffusion_double, self).__init__()
        print("New Flow-Matching based Gesture Triplet Diffusion initialized.")
        self.model = model
        self.schedule_type = schedule_type
        self.num_timesteps = n_timestep
        self.predict_type = predict_type
        self.processing_state = "training"
        self.loss_type = loss_type
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight
        self.clip_denoised = clip_denoised
        self.save_intermediates_every_t = save_intermediates_every_t
        self.cfg_cond_drop_prob = tuple(cfg_cond_drop_prob)

        # Register schedule mainly for time_embedding mapping (0-999) compatibility
        self.register_schedule(schedule_type, n_timestep, linear_start, linear_end, cosine_s)

        self.use_amp = False
        self.model = self.model.half() if self.use_amp else self.model

        # Performance optimization
        self.use_compile = False
        if hasattr(torch, 'compile') and self.use_compile:
            print("Compiling the model for performance optimization...")
            self.model = torch.compile(self.model, backend="eager")

    def _gaussian_kernel1d(self, kernel_size: int, sigma: float, device, dtype):
        x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / (k.sum() + 1e-12)
        return k.view(1, 1, kernel_size)

    def _temporal_gaussian_smooth(self, x: torch.Tensor, kernel_size: int = 9, sigma: float = 1.0):
        # x: [B, T, D]
        B, T, D = x.shape
        k = self._gaussian_kernel1d(kernel_size, sigma, x.device, x.dtype)  # [1,1,K]
        x_ = x.transpose(1, 2).reshape(B * D, 1, T)
        pad = kernel_size // 2
        x_ = F.pad(x_, (pad, pad), mode="replicate")
        y_ = F.conv1d(x_, k)
        y = y_.reshape(B, D, T).transpose(1, 2)
        return y

    def register_schedule(self, schedule_type, n_timestep, linear_start, linear_end, cosine_s):
        betas = make_beta_schedule(schedule_type, n_timestep, linear_start, linear_end, cosine_s).astype(np.float32)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # Kept for compatibility if needed, though FM doesn't strictly need them
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean: loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean: loss = torch.nn.functional.mse_loss(target, pred)
            else: loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == "huber":
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")
        return loss

    # ------------------------------------------ Flow Matching Training (Triplet) ------------------------------------------#

    def p_losses_mix(self, x_start, cond_audio, cond_text, mask, t_float=None, noise=None):
        """
        Flow Matching Training Logic for Triplet Model.
        x_start: [B, T, D] (Motion)
        cond_audio: [B, T, D_a]
        cond_text: [B, 77, D_t]
        mask: [B, 3]
        t_float: Optional [B,], if None, sampled uniformly [0, 1]
        """
        B = x_start.shape[0]
        device = x_start.device

        if noise is None:
            noise = torch.randn_like(x_start)

        # Sample continuous time [0, 1] if not provided
        if t_float is None:
            t_float = torch.rand(B, device=device)

        # Broadcast time for interpolation [B, 1, 1]
        t_cont = t_float.view(B, 1, 1)

        # 1. Conditional Flow Matching (Independent Coupling)
        # Path: x_t = (1 - t) * noise + t * data
        # Target Velocity: v = data - noise
        x_t = (1.0 - t_cont) * noise + t_cont * x_start

        # Add small perturbation during training to make Vector Field robust
        if self.training:
            perturb_noise = torch.randn_like(x_t) * (0.005 * (1.0 - t_cont))
            x_t = x_t + perturb_noise

        target_v = x_start - noise

        # 2. Time Mapping
        # Map continuous t [0, 1] to discrete index [0, 999] for embedding layers
        time_index = (t_float * (self.num_timesteps - 1)).round().long()
        time_index = time_index.to(device)

        # 3. Model Forward
        # Use first element of tuple for training drop prob (standard convention)
        train_drop_prob = self.cfg_cond_drop_prob[0]

        model_out = self.model(
            inputs=x_t,
            cond_audio=cond_audio,
            cond_text=cond_text,
            mask=mask,
            time=time_index,
            cfg_drop_prob=train_drop_prob
        )

        # 4. Loss Calculation
        loss_dict = {}

        # Simple velocity matching loss
        loss_simple = self.get_loss(model_out, target_v, mean=False).mean(dim=[1, 2]) # [B]

        # Auxiliary consistency/smoothness losses (from flow_matching_new.py)
        # Reconstruct x0 to calculate physical smoothness
        x0_hat = x_t + (1.0 - t_cont) * model_out  # [B, T, D]
        x_vel = x0_hat[:, 1:] - x0_hat[:, :-1]
        x_acc = x_vel[:, 1:] - x_vel[:, :-1]
        loss_x_smooth = 0.05 * (x_vel ** 2).mean(dim=[1, 2]) + 0.02 * (x_acc ** 2).mean(dim=[1, 2])

        # Consistency loss (small time perturbation)
        delta = torch.rand_like(t_float) * (2/50)
        t2 = torch.clamp(t_float + delta, max=1.0)
        t2_cont = t2.view(B, 1, 1)

        x_t2_true = (1.0 - t2_cont) * noise + t2_cont * x_start
        # Taylor expansion prediction: x_t2 ≈ x_t + (t2 - t) * v
        x_t2_pred = x_t + (t2_cont - t_cont) * model_out
        loss_cons = 0.1 * self.get_loss(x_t2_pred, x_t2_true, mean=False).mean(dim=[1, 2])

        loss_smooth = loss_x_smooth + loss_cons

        loss = loss_simple + loss_smooth

        # Logging
        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_simple': loss_simple.mean()})
        loss_dict.update({f'{log_prefix}/loss': loss.mean()})
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_smooth.mean()})

        return loss.mean(), loss_dict

    def forward(self, x, cond_a, cond_t=None, mask=None, t_float=None):
        """
        Main entry point.
        """
        if mask is not None:
            return self.p_losses_mix(x, cond_a, cond_t, mask, t_float=t_float)
        else:
            raise NotImplementedError("Flow Matching Triplet requires mask input.")

    # ------------------------------------------ Inference Helper ------------------------------------------#

    def _model_forward_wrapper(self, x, t_float, cond_audio, cond_text, mask, guidance_weight=1.0, is_cfg=False):
        """
        Wraps model forward to handle continuous time mapping and CFG for Triplet Model.
        """
        B = x.shape[0]
        device = x.device

        if not torch.is_tensor(t_float):
            t_float = torch.full((B,), float(t_float), device=device)
        else:
            t_float = t_float.to(device)

        # Map to discrete time index [0, 999]
        t_idx = torch.round(t_float * (self.num_timesteps - 1)).long()
        t_idx = torch.clamp(t_idx, 0, self.num_timesteps - 1)

        if is_cfg and guidance_weight != 1.0:
            # Unconditional forward (cfg_drop_prob=1.0 -> drop all conditions)
            no_cond_out = self.model(
                inputs=x,
                cond_audio=cond_audio,
                cond_text=cond_text,
                mask=mask,
                time=t_idx,
                cfg_drop_prob=1.0
            )
            # Conditional forward (cfg_drop_prob=0.0 -> keep all conditions)
            cond_out = self.model(
                inputs=x,
                cond_audio=cond_audio,
                cond_text=cond_text,
                mask=mask,
                time=t_idx,
                cfg_drop_prob=0.0
            )
            # Extrapolate: v_pred = v_uncond + w * (v_cond - v_uncond)
            return no_cond_out + (cond_out - no_cond_out) * guidance_weight
        else:
            # Standard forward (cfg_drop_prob=0.0)
            return self.model(
                inputs=x,
                cond_audio=cond_audio,
                cond_text=cond_text,
                mask=mask,
                time=t_idx,
                cfg_drop_prob=0.0
            )

    # ------------------------------------------ Inference (Euler) ------------------------------------------#

    @torch.inference_mode()
    def ddim_sample_loop_mix(self,
                           cond_audio,
                           cond_text,
                           mask,
                           shape,
                           sample_steps=50,
                           guidance_weight=3,
                           is_cfg=False,
                           return_intermediates=False):
        """
        Flow Matching Euler Sampler for Triplet Model.
        Replaces ddim_sample_loop_mix.
        """
        print(f"using flow-matching sampler (Euler Triplet), steps={sample_steps}, cfg={guidance_weight if is_cfg else 'Off'}")
        batch, device = shape[0], self.betas.device

        # Initialize from Normal(0, 1) - Corresponds to t=0 in Flow Matching (Conceptually noise)
        motion = torch.randn(shape, device=device)

        cond_audio = cond_audio.to(device)
        cond_text = cond_text.to(device)
        mask = mask.to(device)

        if return_intermediates:
            motion_intermediates = [motion]

        dt = 1.0 / sample_steps

        for i in tqdm(range(sample_steps), desc='Flow Euler Triplet'):
            t_curr = i * dt

            # Step 1: Predict velocity field v
            v = self._model_forward_wrapper(
                x=motion,
                t_float=t_curr,
                cond_audio=cond_audio,
                cond_text=cond_text,
                mask=mask,
                guidance_weight=guidance_weight,
                is_cfg=is_cfg
            )

            # Step 2: Euler Update -> x_{t+1} = x_t + dt * v
            motion = motion + dt * v

            # Optional: Clip (if strictly bounded, usually FM doesn't need this as much as DDPM)
            if self.clip_denoised:
                motion = motion.clamp(-1.0, 1.0)

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == sample_steps - 1:
                    motion_intermediates.append(motion)

        # Post-process: Temporal Smooth
        motion = self._temporal_gaussian_smooth(motion, kernel_size=9, sigma=1.0)

        if return_intermediates:
            return motion, motion_intermediates
        return motion

    @torch.inference_mode()
    def fm_sample_loop_streaming_mix(self,
                                     cond_audio,
                                     cond_text,
                                     mask,
                                     shape,
                                     previous_motion_last_half,
                                     sample_steps=50,
                                     guidance_weight=3,
                                     is_cfg=False,
                                     return_intermediates=True):
        """
        Streaming Flow Matching Sampler for Triplet Model.
        Handles stitching with previous motion.
        """
        batch, device = shape[0], self.betas.device
        assert batch == 1

        switch_length = 0
        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            previous_motion_last_half = previous_motion_last_half.to(device)
            print("using flow-matching sampler (Euler Triplet Streaming) with prior")
        else:
            print("using flow-matching sampler (Euler Triplet Streaming)")

        motion = torch.randn(shape, device=device)

        # Init overlapping part with previous motion
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond_audio = cond_audio.to(device)
        cond_text = cond_text.to(device)
        mask = mask.to(device)

        if return_intermediates:
            motion_intermediates = [motion.clone()]

        dt = 1.0 / sample_steps

        for i in tqdm(range(sample_steps), desc='Flow Euler Triplet Streaming'):
            t_curr = i * dt

            # 1. Predict velocity
            v = self._model_forward_wrapper(
                x=motion,
                t_float=t_curr,
                cond_audio=cond_audio,
                cond_text=cond_text,
                mask=mask,
                guidance_weight=guidance_weight,
                is_cfg=is_cfg
            )

            # 2. Fix velocity in the overlapping region to 0 (Keep previous motion static relative to flow)
            # Alternatively, we can let it flow and overwrite later, but setting v=0 helps stability
            if previous_motion_last_half is not None:
                v[:, :switch_length] = 0.0

            # 3. Update
            motion = motion + dt * v

            # 4. Hard enforce stitching: Overwrite the prefix with the ground truth from previous window
            if previous_motion_last_half is not None:
                # In Flow Matching, the trajectory is straight.
                # Heuristic: Inject the end result of the previous chunk?
                # A simple strategy used in the reference:
                idx = min(i + 1, previous_motion_last_half.shape[0] - 1)
                # Note: previous_motion_last_half usually comes from a list of intermediate steps
                # If previous_motion_last_half is just the final result (shape [1, T, D]), we just clamp to it.
                # Assuming previous_motion_last_half is [1, T, D] (Fixed Target):
                if len(previous_motion_last_half.shape) == 3:
                     motion[0, :switch_length] = previous_motion_last_half[0]
                else:
                    # If it provides trajectory [Steps, T, D]
                     motion[0, :switch_length] = previous_motion_last_half[idx]

            if return_intermediates:
                motion_intermediates.append(motion.clone())

        # Post-process
        motion = self._temporal_gaussian_smooth(motion, kernel_size=9, sigma=1.0)

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0).squeeze(1)
            return motion, motion_intermediates

        return motion
