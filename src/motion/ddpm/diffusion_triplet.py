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

from .utils.utils import extract, noise_like

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

def cosine_time_schedule(total_timesteps: int,
                         sampling_timesteps: int,
                         kind: str = "cosine_in") -> list:
    """
    Returns ascending discrete time indices: [-1, t0, t1, ..., T-1]
    """
    T = int(total_timesteps)
    S = int(sampling_timesteps)
    assert T >= 2 and S >= 1

    u = torch.linspace(0.0, 1.0, steps=S, dtype=torch.float64)

    if kind == "linear":
        g = u
    elif kind == "cosine_in":
        g = 1.0 - torch.cos(u * math.pi / 2.0)
    else:
        raise ValueError(f"unknown kind={kind}")

    t = torch.floor(g * (T - 1)).to(torch.long)

    if t.numel() > 0:
        t[0] = 0
        for i in range(1, t.numel()):
            if t[i] <= t[i - 1]:
                t[i] = min(t[i - 1] + 1, T - 1)
        t[-1] = T - 1

    times = torch.cat([t.new_tensor([-1], dtype=torch.long), t], dim=0)
    return times.tolist()

torch.compiler.cudagraph_mark_step_begin()

class Gesture_Diffusion_double(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 schedule_type: str = "linear",
                 predict_type: str = "eps",
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

        self.register_schedule(schedule_type, n_timestep, linear_start, linear_end, cosine_s)

        self.use_amp = False
        self.model = self.model.half() if self.use_amp else self.model

    def register_schedule(self, schedule_type, n_timestep, linear_start, linear_end, cosine_s):
        betas = make_beta_schedule(schedule_type=schedule_type, n_timestep=n_timestep, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", to_torch(posterior_variance))

        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))

        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        if self.predict_type == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.predict_type == "x0":
            lvlb_weights = 0.5 * to_torch(np.sqrt(alphas_cumprod)) / (2. * 1 - self.alphas_cumprod)
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_type == "eps":
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        elif self.predict_type == "x0":
            return noise

    def predict_noise_from_start(self, x_t, t, x0):
        if self.predict_type == "x0":
            return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            )
        elif self.predict_type == "eps":
            return x0

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return x_t

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == "huber":
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # ------------------------------------------ New Mixed Training ------------------------------------------#

    def p_losses_mix(self, x_start, cond_audio, cond_text, mask, timestep, noise=None):
        """
        New loss function for Mix_DiT.
        cond_audio: [B, T, D_a]
        cond_text: [B, 77, D_t]
        mask: [B, 3]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=timestep, noise=noise)

        # Use the first element of the drop prob tuple for training (typically 0.1)
        train_drop_prob = self.cfg_cond_drop_prob[0]

        model_out = self.model(
            inputs=x_t,
            cond_audio=cond_audio,
            cond_text=cond_text,
            mask=mask,
            time=timestep,
            cfg_drop_prob=train_drop_prob
        )

        loss_dict = {}
        if self.predict_type == "eps":
            target = noise
        elif self.predict_type == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"predict_type {self.predict_type} not yet supported")

        loss = self.get_loss(model_out, target, mean=False)
        loss = loss.mean(dim=[1, 2])  # [B,]

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[timestep] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def loss_function_mix(self, x, cond_audio, cond_text, mask):
        """
        Main entry point for calculating loss in MixTask_trainer
        """
        B, _, _ = x.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=x.device).long()
        return self.p_losses_mix(x, cond_audio, cond_text, mask, t)

    def forward(self, x, cond_a, cond_t=None, mask=None):
        # Dispatcher
        if mask is not None:
            # New Mix Model Path
            return self.loss_function_mix(x, cond_a, cond_t, mask)
        else:
            raise NotImplementedError("Legacy forward only supports non-mix model. Use p_losses_mix for mix model training.")

    # ------------------------------------------ New Mixed Inference ------------------------------------------#

    @torch.inference_mode()
    def p_sample_loop_mix(self, cond_audio, cond_text, mask, shape, return_intermediates=False):
        """
        Standard DDPM Sampling for Mix Model
        """
        device = self.betas.device
        b = shape[0]
        motion = torch.randn(shape, device=device)

        if return_intermediates:
            motion_intermediates = [motion]

        for i in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps, leave=False, position=1, desc='Sample T'):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            # Predict noise/x0
            # For standard p_sample, we usually don't use CFG (guidance=1), so drop_prob=0
            model_out = self.model(motion, cond_audio, cond_text, mask, t, cfg_drop_prob=0.0)

            if self.predict_type == "eps":
                x_recon = self.predict_start_from_noise(motion, t=t, noise=model_out)
                pred_noise = model_out
            elif self.predict_type == "x0":
                x_recon = model_out
                pred_noise = self.predict_noise_from_start(motion, t=t, x0=x_recon)

            if self.clip_denoised:
                x_recon.clamp_(-1., 1.)

            # Get posterior mean/var
            model_mean, _, model_log_variance = self.q_posterior(x_start=x_recon, x_t=motion, t=t)
            noise = torch.randn_like(motion) if i > 0 else 0.

            motion = model_mean + (0.5 * model_log_variance).exp() * noise

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == self.num_timesteps - 1:
                    motion_intermediates.append(motion)

        if return_intermediates:
            return motion, motion_intermediates
        return motion

    @torch.inference_mode()
    def ddim_sample_loop_mix(self,
                             cond_audio,
                             cond_text,
                             mask,
                             shape,
                             sample_steps=50,
                             guidance_weight=3,
                             eta=0.,
                             is_cfg=False,
                             return_intermediates=False):
        """
        DDIM Sampling for Mix Model (Handles Audio, Text, or Joint based on mask)
        """
        print(f"using ddim_sample_loop_mix (is_cfg={is_cfg})")
        batch, device, total_timesteps, sampling_timesteps = shape[0], self.betas.device, self.num_timesteps, sample_steps

        # Time schedule
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        motion = torch.randn(shape, device=device)

        # Ensure inputs are on device
        cond_audio = cond_audio.to(device)
        cond_text = cond_text.to(device)
        mask = mask.to(device)

        if return_intermediates:
            motion_intermediates = [motion]

        x_recon = None
        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc='sampling loop')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            if is_cfg:
                # Conditional Forward (prob=0.0)
                cond_out = self.model(
                    inputs=motion, cond_audio=cond_audio, cond_text=cond_text, mask=mask,
                    time=time_cond, cfg_drop_prob=0.0
                )
                # Unconditional Forward (prob=1.0)
                uncond_out = self.model(
                    inputs=motion, cond_audio=cond_audio, cond_text=cond_text, mask=mask,
                    time=time_cond, cfg_drop_prob=1.0
                )
                model_out = uncond_out + guidance_weight * (cond_out - uncond_out)
            else:
                model_out = self.model(
                    inputs=motion, cond_audio=cond_audio, cond_text=cond_text, mask=mask,
                    time=time_cond, cfg_drop_prob=0.0
                )

            if self.predict_type == "eps":
                x_recon = self.predict_start_from_noise(motion, t=time_cond, noise=model_out)
                pred_noise = model_out
            elif self.predict_type == "x0":
                x_recon = model_out
                pred_noise = self.predict_noise_from_start(motion, t=time_cond, x0=x_recon)

            if time_next < 0:
                motion = x_recon
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(motion)

            motion = x_recon * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == self.num_timesteps - 1:
                    motion_intermediates.append(motion)

        if return_intermediates:
            return motion, motion_intermediates
        return motion

    @torch.inference_mode()
    def ddim_sample_loop_streaming_mix(self,
                                       cond_audio,
                                       cond_text,
                                       mask,
                                       shape,
                                       previous_motion_last_half,
                                       sample_steps=50,
                                       guidance_weight=3,
                                       eta=0.,
                                       is_cfg=False,
                                       return_intermediates=True):
        """
        Streaming DDIM for Mix Model
        """
        switch_length = 0
        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            print("using ddim_mix_streaming with motion prior")
        else:
            print("using ddim_mix_streaming")

        batch, device, total_timesteps, sampling_timesteps = shape[0], self.betas.device, self.num_timesteps, sample_steps
        assert batch == 1 # Streaming usually implies single batch for continuity

        # Use Cosine schedule for streaming stability
        times = cosine_time_schedule(total_timesteps, sampling_timesteps, kind="cosine_in")
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond_audio = cond_audio.to(device)
        cond_text = cond_text.to(device)
        mask = mask.to(device)

        if return_intermediates:
            motion_intermediates = [motion.clone()]

        x_recon = None
        for i, (time, time_next) in enumerate(tqdm(time_pairs, desc='streaming sampling')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            if is_cfg:
                cond_out = self.model(
                    inputs=motion, cond_audio=cond_audio, cond_text=cond_text, mask=mask,
                    time=time_cond, cfg_drop_prob=0.0
                )
                uncond_out = self.model(
                    inputs=motion, cond_audio=cond_audio, cond_text=cond_text, mask=mask,
                    time=time_cond, cfg_drop_prob=1.0
                )
                model_out = uncond_out + guidance_weight * (cond_out - uncond_out)
            else:
                model_out = self.model(
                    inputs=motion, cond_audio=cond_audio, cond_text=cond_text, mask=mask,
                    time=time_cond, cfg_drop_prob=0.0
                )

            if self.predict_type == "eps":
                x_recon = self.predict_start_from_noise(motion, t=time_cond, noise=model_out)
                pred_noise = model_out
            elif self.predict_type == "x0":
                x_recon = model_out
                pred_noise = self.predict_noise_from_start(motion, t=time_cond, x0=x_recon)

            # Enforce prior
            if previous_motion_last_half is not None:
                # In epsilon prediction, we can try to force noise to 0 in overlapping regions
                # or just force the motion pixels. Forcing pixels is more stable in streaming.
                pass

            if time_next < 0:
                motion = x_recon
                if return_intermediates:
                    motion_intermediates.append(motion.clone())
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(motion)

            motion = x_recon * alpha_next.sqrt() + c * pred_noise + sigma * noise

            # Re-inject prior motion
            if previous_motion_last_half is not None and i < self.num_timesteps - 1:
                motion[0, :switch_length] = previous_motion_last_half[i+1] # Heuristic injection from previous rollout

            if return_intermediates:
                motion_intermediates.append(motion.clone())

        # Smooth result
        motion = gaussian_filter1d(motion.cpu().numpy(), sigma=1, axis=1)
        motion = torch.from_numpy(motion).to(device)

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0)
            if motion_intermediates.shape[1] == 1:
                motion_intermediates = motion_intermediates.squeeze(1)
            return motion, motion_intermediates

        return motion
