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
        print("New flow-matching based Gesture Diffusion initialized.")
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

        self.mimic_delay_time = 5
        self.step_500_need_ctrl_list = list(range(999, 0, -2))[::2]

        self.register_schedule(schedule_type, n_timestep, linear_start, linear_end, cosine_s)

        self.use_amp = False
        self.model = self.model.half() if self.use_amp else self.model
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
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        prediction_type = "epsilon" if self.predict_type == "eps" else "sample"
        self.noise_scheduler: DDPMScheduler = DDPMScheduler(
            num_train_timesteps=n_timestep,
            trained_betas=betas,
            prediction_type=prediction_type,
            clip_sample=self.clip_denoised,
        )

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))


        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))


        if self.predict_type == "eps":
            # posterior variance is needed for lvlb
            posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
            self.register_buffer("posterior_variance", to_torch(posterior_variance))
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        else:
            lvlb_weights = torch.ones_like(to_torch(betas))

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

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

    # ------------------------------------------ Flow Matching Training ------------------------------------------#

    def p_losses(self, x_start, cond_a, t_float, state=None, noise=None):
        # x_start: [B, T, D_total]
        x_start, cond_m = x_start.chunk(2, dim=-1)
        if noise is None:
            noise = torch.randn_like(x_start)

        B = x_start.shape[0]


        t_cont = t_float.view(B, 1, 1)

        # Conditional Flow Matching (Independent Coupling)


        x_t = (1.0 - t_cont) * noise + t_cont * x_start


        if self.training:
            perturb_noise = torch.randn_like(x_t) * (0.005 * (1.0 - t_cont))
            x_t = x_t + perturb_noise

        target_v = x_start - noise



        time_index = (t_float * (self.num_timesteps - 1)).round().long()
        time_index = time_index.to(x_start.device)


        model_out = self.model(
            inputs=x_t,
            cond_a=cond_a,
            cond_m=cond_m,
            state=state,
            time=time_index,
            processing_state=self.processing_state,
            last_time_stamp=self.num_timesteps,
            cfg_cond_drop_prob=self.cfg_cond_drop_prob,
        )

        loss_dict = {}

        loss_simple = self.get_loss(model_out, target_v, mean=False).mean(dim=[1, 2]) # [B, T, D]

        x0_hat = x_t + (1.0 - t_cont) * model_out  # [B, T, D]
        x_vel = x0_hat[:, 1:] - x0_hat[:, :-1]
        x_acc = x_vel[:, 1:] - x_vel[:, :-1]
        loss_x_smooth = 0.05 * (x_vel ** 2).mean(dim=[1, 2]) + 0.02 * (x_acc ** 2).mean(dim=[1, 2])

        delta = torch.rand_like(t_float) * (2/50)
        t2 = torch.clamp(t_float + delta, max=1.0)
        t2_cont = t2.view(B, 1, 1)
        x_t2_true = (1.0 - t2_cont) * noise + t2_cont * x_start
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

    def p_losses_text(self, x_start, cond_a, cond_t, t_float, state=None, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        B = x_start.shape[0]
        t_cont = t_float.view(B, 1, 1)

        x_t = (1.0 - t_cont) * noise + t_cont * x_start
        target_v = x_start - noise

        time_index = (t_float * (self.num_timesteps - 1)).round().long().to(x_start.device)

        model_out = self.model(
            inputs=x_t,
            cond_a=cond_a,
            cond_t=cond_t,
            state=state,
            time=time_index,
            processing_state=self.processing_state,
            last_time_stamp=self.num_timesteps,
            cfg_cond_drop_prob=self.cfg_cond_drop_prob,
        )

        loss = self.get_loss(model_out, target_v, mean=False).mean(dim=[1, 2])

        log_prefix = 'train' if self.training else 'val'
        loss_dict = {f'{log_prefix}/loss': loss.mean(),
                     f'{log_prefix}/loss_vlb': torch.tensor(0.0, device=x_start.device)}

        return loss.mean(), loss_dict

    def forward(self, x: torch.Tensor, cond_a: torch.Tensor, state=None, cond_t: Optional[torch.Tensor]=None):
        B = x.shape[0]
        device = x.device



        t_float = torch.rand(B, device=device)

        if cond_t is None:
            return self.p_losses(x, cond_a, t_float, state)
        return self.p_losses_text(x, cond_a, cond_t, t_float, state)

    # ------------------------------------------ Inference Helper ------------------------------------------#

    def _model_forward_wrapper(self, x, t_float, cond_a, cond_m=None, cond_t=None, state=None, guidance_weight=1.0, is_cfg=False, base_dit=None):
        B = x.shape[0]
        device = x.device

        if not torch.is_tensor(t_float):

            t_float = torch.full((B,), float(t_float), device=device)
        else:
            t_float = t_float.to(device)

        t_idx = torch.round(t_float * (self.num_timesteps - 1)).long()
        t_idx = torch.clamp(t_idx, 0, self.num_timesteps - 1)
        time_cond = t_idx

        # Select model (wrapper or inner base_dit)
        model_func = base_dit if base_dit is not None else self.model

        # Argument preparation

        common_args = {
            "inputs": x,
            "cond_a": cond_a,
            "state": state,
            "time": time_cond,
            "processing_state": "test",
            "last_time_stamp": self.num_timesteps,
        }


        if cond_t is not None:
            common_args["cond_t"] = cond_t
        else:
            common_args["cond_m"] = cond_m

        if is_cfg and guidance_weight != 1.0:
            # Unconditional forward
            no_cond_out = model_func(
                **common_args,
                cfg_cond_drop_prob=(1, 1, 1)
            )
            # Conditional forward
            cond_out = model_func(
                **common_args,
                cfg_cond_drop_prob=(0, 0, 0)
            )
            return no_cond_out + (cond_out - no_cond_out) * guidance_weight
        else:
            return model_func(
                **common_args,
                cfg_cond_drop_prob=(0, 0, 0)
            )

    # ------------------------------------------ Inference ------------------------------------------#

    @torch.inference_mode()
    def fm_sample_loop(self, cond, object_motion, state, shape, sample_steps=50, guidance_weight=3, eta=0., is_cfg=False, return_intermediates=False):
        print(f"using flow-matching sampler (Euler 1st-order), steps={sample_steps}, cfg={guidance_weight}")
        batch, device = shape[0], self.betas.device


        motion = torch.randn(shape, device=device)
        cond = cond.to(device)
        if object_motion is not None: object_motion = object_motion.to(device)
        if state is not None: state = state.to(device)

        if return_intermediates:
            motion_intermediates = [motion]

        assert motion.shape[1] % 2 == 0
        half = motion.shape[1] // 2

        dt = 1.0 / sample_steps

        for i in tqdm(range(sample_steps), desc='Flow Euler Sampling'):
            t_curr = i * dt

            # Step 1: Predict velocity
            # v = f(x_t, t)
            v = self._model_forward_wrapper(motion, t_curr, cond, cond_m=object_motion, state=state, guidance_weight=guidance_weight, is_cfg=is_cfg)

            # Step 2: Update position
            # x_{t+1} = x_t + dt * v
            motion = motion + dt * v

            # Clip (optional)
            if self.clip_denoised:
                motion = motion.clamp(-1.0, 1.0)

            # Edge stitching
            if i < sample_steps - 1 and batch > 1:
                motion[1:, :half] = motion[:-1, half:]

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == sample_steps - 1:
                    motion_intermediates.append(motion)

        # Post-process
        motion = self._temporal_gaussian_smooth(motion, kernel_size=9, sigma=1.0)

        if return_intermediates:
            return motion, motion_intermediates
        return motion


    @torch.inference_mode()
    def fm_sample_loop_text(self, cond, cond_text, state, shape, sample_steps=50, guidance_weight=3, eta=0., is_cfg=False, return_intermediates=False):
        print(f"using flow-matching sampler (Euler, Text), steps={sample_steps}")
        batch, device = shape[0], self.betas.device
        motion = torch.randn(shape, device=device)

        cond = cond.to(device)
        cond_text = cond_text.to(device)
        if state is not None: state = state.to(device)

        if return_intermediates: motion_intermediates = [motion]

        dt = 1.0 / sample_steps

        for i in tqdm(range(sample_steps), desc='Flow Euler Text'):
            t_curr = i * dt

            # Predict velocity
            v = self._model_forward_wrapper(motion, t_curr, cond, cond_t=cond_text, state=state, guidance_weight=guidance_weight, is_cfg=is_cfg)

            # Update
            motion = motion + dt * v

            if self.clip_denoised: motion = motion.clamp(-1.0, 1.0)

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == sample_steps - 1:
                    motion_intermediates.append(motion)

        motion = self._temporal_gaussian_smooth(motion, kernel_size=9, sigma=1.0)

        if return_intermediates: return motion, motion_intermediates
        return motion


    @torch.inference_mode()
    def fm_sample_loop_streaming(self, cond, shape, previous_motion_last_half, sample_steps=50, guidance_weight=3, eta=0., is_cfg=False, return_intermediates=True):
        batch, device = shape[0], self.betas.device
        assert batch == 1

        switch_length = 0
        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            previous_motion_last_half = previous_motion_last_half.to(device)
            print("using flow-matching sampler (Euler Streaming) with prior")
        else:
            print("using flow-matching sampler (Euler Streaming)")

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond = cond.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates: motion_intermediates = [motion.clone()]

        dt = 1.0 / sample_steps

        for i in tqdm(range(sample_steps), desc='Flow Euler Streaming'):
            t_curr = i * dt

            # Predict velocity
            v = self._model_forward_wrapper(motion, t_curr, cond, cond_m=None, state=state, guidance_weight=guidance_weight, is_cfg=is_cfg)

            # Force prefix fixed
            if previous_motion_last_half is not None: v[:, :switch_length] = 0.0

            # Update
            motion = motion + dt * v

            # Hard enforce prefix
            if previous_motion_last_half is not None:
                idx = min(i + 1, previous_motion_last_half.shape[0] - 1)
                motion[0, :switch_length] = previous_motion_last_half[idx]

            if return_intermediates:
                motion_intermediates.append(motion.clone())

        motion = self._temporal_gaussian_smooth(motion, kernel_size=9, sigma=1.0)

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0).squeeze(1)
            return motion, motion_intermediates

        return motion


    @torch.inference_mode()
    def fm_sample_loop_streaming_controlnet(self, cond, shape, previous_motion_last_half, sample_steps=50, guidance_weight=3, eta=0., is_cfg=True, return_intermediates=True):
        batch, device = shape[0], self.betas.device
        assert batch == 1

        switch_length = 0
        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            previous_motion_last_half = previous_motion_last_half.to(device)

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond = cond.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates: motion_intermediates = [motion]

        dt = 1.0 / sample_steps
        base_dit = self.model.base_dit

        for i in tqdm(range(sample_steps), desc='Flow Euler ControlNet'):
            t_curr = i * dt

            # Predict velocity
            v = self._model_forward_wrapper(motion, t_curr, cond, cond_m=None, state=state, guidance_weight=guidance_weight, is_cfg=is_cfg, base_dit=base_dit)

            if previous_motion_last_half is not None: v[:, :switch_length] = 0.0

            # Update
            motion = motion + dt * v

            if previous_motion_last_half is not None:
                idx = min(i + 1, previous_motion_last_half.shape[0] - 1)
                motion[0, :switch_length] = previous_motion_last_half[idx]

            if return_intermediates:
                motion_intermediates.append(motion)

        motion = self._temporal_gaussian_smooth(motion, kernel_size=9, sigma=1.0)
        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0).squeeze(1)
            return motion, motion_intermediates
        return motion


    @torch.inference_mode()
    def fm_sample_loop_streaming_text(self, cond, cond_text, shape, previous_motion_last_half, sample_steps=50, guidance_weight=3, eta=1., is_cfg=True, return_intermediates=True):
        """
        Text conditional streaming Euler sampler
        """
        batch, device = shape[0], self.betas.device
        assert batch == 1

        switch_length = 0
        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            previous_motion_last_half = previous_motion_last_half.to(device)

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond = cond.to(device)
        cond_text = cond_text.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates: motion_intermediates = [motion]
        dt = 1.0 / sample_steps

        for i in tqdm(range(sample_steps), desc='Flow Euler Text Streaming'):
            t_curr = i * dt

            # Predict velocity
            v = self._model_forward_wrapper(motion, t_curr, cond, cond_t=cond_text, state=state, guidance_weight=guidance_weight, is_cfg=is_cfg)
            if previous_motion_last_half is not None: v[:, :switch_length] = 0.0

            # Update
            motion = motion + dt * v

            if previous_motion_last_half is not None:
                idx = min(i + 1, previous_motion_last_half.shape[0] - 1)
                motion[0, :switch_length] = previous_motion_last_half[idx]

            if return_intermediates:
                motion_intermediates.append(motion)

        motion = self._temporal_gaussian_smooth(motion, kernel_size=9, sigma=1.0)

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0).squeeze(1)
            return motion, motion_intermediates
        return motion
