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
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler
)

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
        self.clip_denoised = False
        self.save_intermediates_every_t = save_intermediates_every_t
        self.cfg_cond_drop_prob = tuple(cfg_cond_drop_prob)


        self.register_schedule(schedule_type, n_timestep, linear_start, linear_end, cosine_s)


        self.use_amp = False
        self.model = self.model.half() if self.use_amp else self.model
        self.use_compile = False
        if hasattr(torch, 'compile') and self.use_compile:
            print("Compiling the model for performance optimization...")
            self.model = torch.compile(self.model, backend="eager")
            self.predict_start_from_noise = torch.compile(self.predict_start_from_noise)


    def register_schedule(self, schedule_type, n_timestep, linear_start, linear_end, cosine_s):

        betas = make_beta_schedule(
            schedule_type=schedule_type,
            n_timestep=n_timestep,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s
        ).astype(np.float32)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])


        prediction_type = "epsilon" if self.predict_type == "eps" else "sample"
        self.noise_scheduler: DDPMScheduler = DDPMScheduler(
            num_train_timesteps=n_timestep,
            trained_betas=betas,
            prediction_type=prediction_type,
            clip_sample=False,
        )


        self._scheduler_config = self.noise_scheduler.config


        self._scheduler_config["clip_sample"] = False
        self._scheduler_config["clip_sample_range"] = 1000.0

        to_torch = partial(torch.tensor, dtype=torch.float32)


        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
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

        assert not torch.isnan(self.lvlb_weights).all()

    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
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

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # t: [B] long
        return self.noise_scheduler.add_noise(x_start, noise, t)

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
    def p_losses(self, x_start, cond_a, timestep, state=None, noise=None):

        x_start, cond_m = x_start.chunk(2, dim=-1)
        # print(x_start.mean(), x_start.std(), x_start.max(), x_start.min())
        # print("x_start max id: ", torch.argmax(x_start, dim=1))
        # print("x_start min id: ", torch.argmin(x_start, dim=1))
        # print(cond_m.mean(), cond_m.std(), cond_m.max(), cond_m.min())
        # print(cond_a.mean(), cond_a.std(), cond_a.max(), cond_a.min())

        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=timestep, noise=noise)
        # print("x_start with noise: ", x_t.mean(), x_t.std(), x_t.max(), x_t.min())

        model_out = self.model(
            inputs=x_t,
            cond_a=cond_a,
            cond_m=cond_m,
            state=state,
            time=timestep,
            processing_state=self.processing_state,
            last_time_stamp=self.num_timesteps,
            cfg_cond_drop_prob=self.cfg_cond_drop_prob,
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

    def p_losses_text(self, x_start, cond_a, cond_t, timestep, state=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=timestep, noise=noise)
        model_out = self.model(
            inputs=x_t,
            cond_a=cond_a,
            cond_t=cond_t,
            state=state,
            time=timestep,
            processing_state=self.processing_state,
            last_time_stamp=self.num_timesteps,
            cfg_cond_drop_prob=self.cfg_cond_drop_prob,
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

    def forward(self, x: torch.Tensor, cond_a: torch.Tensor, state=None, cond_t: Optional[torch.Tensor]=None):
        B, _, _ = x.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=x.device).long()  # [B,] the noise step index

        if cond_t is None:
            return self.p_losses(x, cond_a, t, state)
        return self.p_losses_text(x, cond_a, cond_t, t, state)

    # ------------------------------------------ inference: DDPM ------------------------------------------#
    @torch.no_grad()
    def p_sample(self, x, cond, object_motion, t, clip_denoised=False, repeat_noise=False, state=None):
        b, *_, device = *x.shape, x.device
        scheduler = self.noise_scheduler


        if t.ndim > 0:
            timestep = t[0]
        else:
            timestep = t

        timestep = timestep.to(device)

        time_cond = torch.full((b,), int(timestep.item()), device=device, dtype=torch.long)

        model_out = self.model(
            inputs=x,
            cond_a=cond,
            cond_m=object_motion,
            state=state,
            time=time_cond,
            processing_state="test",
            last_time_stamp=self.num_timesteps - 1,
            cfg_cond_drop_prob=(0, 0, 0),
        )


        step_output = scheduler.step(
            model_output=model_out,
            timestep=timestep,
            sample=x
        )
        x_prev = step_output.prev_sample

        if clip_denoised or self.clip_denoised:
            x_prev = x_prev.clamp(-1.0, 1.0)

        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, cond, object_motion, shape, return_intermediates=False, state=None):
        device = self.betas.device
        b = shape[0]

        assert shape[1] % 2 == 0
        half = shape[1] // 2

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_timesteps, device=device)

        motion = torch.randn(shape, device=device)
        if return_intermediates:
            motion_intermediates = [motion]

        for i, t in enumerate(tqdm(scheduler.timesteps, total=len(scheduler.timesteps), leave=False, position=1, desc='Sample T')):
            time_cond = torch.full((b,), int(t.item()), device=device, dtype=torch.long)

            model_out = self.model(
                inputs=motion,
                cond_a=cond,
                cond_m=object_motion,
                state=state,
                time=time_cond,
                processing_state="test",
                last_time_stamp=self.num_timesteps - 1,
                cfg_cond_drop_prob=(0, 0, 0),
            )

            step_output = scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample

            if self.clip_denoised:
                motion = motion.clamp(-1.0, 1.0)


            if b > 1 and i < len(scheduler.timesteps) - 1:
                motion[1:, :half] = motion[:-1, half:]

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == len(scheduler.timesteps) - 1:
                    motion_intermediates.append(motion)
        if return_intermediates:
            return motion, motion_intermediates
        return motion

    # ------------------------------------------ inference: DDIM ------------------------------------------#
    @torch.no_grad()
    def ddim_sample_loop(self,
                         cond,
                         object_motion,
                         state,
                         shape,
                         sample_steps=50,
                         guidance_weight=3,
                         eta=0.,
                         is_cfg=False,
                         return_intermediates=False):

        print("Using Diffusers DDIM with 'Raw' Configuration")
        batch, device = shape[0], self.betas.device


        scheduler_kwargs = {
            "num_train_timesteps": self.num_timesteps,
            "beta_start": self.noise_scheduler.config.beta_start,
            "beta_end": self.noise_scheduler.config.beta_end,
            "beta_schedule": None,
            "trained_betas": self.betas.cpu().numpy(),
            "prediction_type": "epsilon" if self.predict_type == "eps" else "sample",
            "clip_sample": False,
            "timestep_spacing": "trailing",
            "rescale_betas_zero_snr": False,
        }


        ddim_scheduler = DDIMScheduler(**scheduler_kwargs)
        ddim_scheduler.eta = eta
        ddim_scheduler.set_timesteps(sample_steps, device=device)

        train_alpha_last = self.alphas_cumprod[-1].item()
        infer_alpha_last = ddim_scheduler.alphas_cumprod[-1].item()
        print(f"Debug Schedule Alignment: Train Alpha[T]={train_alpha_last:.6f}, DDIM Alpha[T]={infer_alpha_last:.6f}")

        if abs(train_alpha_last - infer_alpha_last) > 1e-3:
            print(":Alpha Schedule ! alphas_cumprod...")

            ddim_scheduler.alphas_cumprod = self.alphas_cumprod.clone().to("cpu")



        motion = torch.randn(shape, device=device)

        cond = cond.to(device)
        if object_motion is not None:
            object_motion = object_motion.to(device)
        if state is not None:
            state = state.to(device)

        if return_intermediates:
            motion_intermediates = [motion]


        for i, t in enumerate(tqdm(ddim_scheduler.timesteps, desc='DDIM Sampling')):

            time_cond = torch.full((batch,), int(t.item()), device=device, dtype=torch.long)


            if is_cfg:
                no_cond_out = self.model(inputs=motion, cond_a=cond, cond_m=object_motion, state=state, time=time_cond,
                                       processing_state=self.processing_state, last_time_stamp=self.num_timesteps - 1,
                                       cfg_cond_drop_prob=(1, 1, 1))
                cond_out = self.model(inputs=motion, cond_a=cond, cond_m=object_motion, state=state, time=time_cond,
                                    processing_state=self.processing_state, last_time_stamp=self.num_timesteps - 1,
                                    cfg_cond_drop_prob=(0, 0, 0))
                model_out = no_cond_out + (cond_out - no_cond_out) * guidance_weight
            else:
                model_out = self.model(inputs=motion, cond_a=cond, cond_m=object_motion, state=state, time=time_cond,
                                     processing_state=self.processing_state, last_time_stamp=self.num_timesteps - 1,
                                     cfg_cond_drop_prob=(0, 0, 0))





            # 5. Scheduler Step

            step_output = ddim_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
                return_dict=False
            )
            motion = step_output[0]

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == len(ddim_scheduler.timesteps) - 1:
                    motion_intermediates.append(motion)

        if return_intermediates:
            return motion, motion_intermediates
        return motion

    @torch.no_grad()
    def ddim_sample_loop_text(self,
                              cond,
                              cond_text,
                              state,
                              shape,
                              sample_steps=50,
                              guidance_weight=3,
                              eta=0.,
                              is_cfg=False,
                              return_intermediates=False):
        print("using diffusers DDIM sampler (text-conditioned)")
        batch, device = shape[0], self.betas.device

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.detach().cpu().numpy(),
            prediction_type="epsilon" if self.predict_type == "eps" else "sample",
            clip_sample=self.clip_denoised,
        )
        ddim_scheduler.eta = float(eta)
        ddim_scheduler.set_timesteps(sample_steps, device=device)

        motion = torch.randn(shape, device=device)
        cond = cond.to(device)
        cond_text = cond_text.to(device)
        if state is not None:
            state = state.to(device)

        if return_intermediates:
            motion_intermediates = [motion]

        assert batch >= 1

        for i, t in enumerate(tqdm(ddim_scheduler.timesteps, desc='sampling loop time step')):
            time_cond = torch.full((batch,), int(t.item()), device=device, dtype=torch.long)

            if is_cfg:
                no_cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1)
                )
                cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0)
                )
                model_out = no_cond_model_out + (cond_model_out - no_cond_model_out) * guidance_weight
            else:
                model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0)
                )

            step_output = ddim_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample

            if self.clip_denoised:
                motion = motion.clamp(-1.0, 1.0)

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == len(ddim_scheduler.timesteps) - 1:
                    motion_intermediates.append(motion)

        if return_intermediates:
            return motion, motion_intermediates
        return motion

    # ------------------------------------------ inference: streaming DDIM ------------------------------------------#
    @torch.inference_mode()
    def ddim_sample_loop_streaming(self,
                                   cond,
                                   shape,
                                   previous_motion_last_half,
                                   sample_steps=50,
                                   guidance_weight=3,
                                   eta=0.,
                                   is_cfg=False,
                                   return_intermediates=True):
        batch, device, total_timesteps = shape[0], self.betas.device, self.num_timesteps
        assert batch == 1

        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            print("previous motion last half shape: ", previous_motion_last_half.shape)
            assert switch_length < shape[1], "previous motion should be shorter than the current motion shape"
            print("using diffusers DDIM sampler with motion prior")
            previous_motion_last_half = previous_motion_last_half.to(device)
        else:
            print("using diffusers DDIM sampler")

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.detach().cpu().numpy(),
            prediction_type="epsilon" if self.predict_type == "eps" else "sample",
            clip_sample=self.clip_denoised,
        )
        ddim_scheduler.eta = float(eta)
        ddim_scheduler.set_timesteps(sample_steps, device=device)

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond = cond.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates:
            motion_intermediates = [motion.clone()]

        for i, t in enumerate(tqdm(ddim_scheduler.timesteps, desc='sampling loop time step')):
            time_cond = torch.full((batch,), int(t.item()), device=device, dtype=torch.long)

            if is_cfg:
                no_cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1)
                )
                cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0)
                )
                model_out = no_cond_model_out + (cond_model_out - no_cond_model_out) * guidance_weight
            else:
                model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0)
                )


            if previous_motion_last_half is not None:
                model_out[:, :switch_length] = 0.0

            step_output = ddim_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample

            if previous_motion_last_half is not None and i < self.num_timesteps - 1:
                motion[0, :switch_length] = previous_motion_last_half[i+1]

            if return_intermediates:
                motion_intermediates.append(motion.clone())

        # smooth motion with a gaussian filter to remove high frequency jitters
        motion = gaussian_filter1d(motion.cpu().numpy(), sigma=1, axis=1)
        motion = torch.from_numpy(motion).to(device)

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0) # (num_intermediates, B, T, D)
            if motion_intermediates.shape[1] == 1:
                motion_intermediates = motion_intermediates.squeeze(1)
            else:
                raise ValueError("batch size should be 1 when return_intermediates is True")
            return motion, motion_intermediates

        return motion

    @torch.inference_mode()
    def ddim_sample_loop_streaming_controlnet(self,
                                              cond,
                                              shape,
                                              previous_motion_last_half,
                                              sample_steps=50,
                                              guidance_weight=3,
                                              eta=0.,
                                              is_cfg=True,
                                              return_intermediates=True):
        batch, device, total_timesteps = shape[0], self.betas.device, self.num_timesteps
        assert batch == 1

        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            print("previous motion last half shape: ", previous_motion_last_half.shape)
            assert switch_length < shape[1], "previous motion should be shorter than the current motion shape"
            print("using diffusers DDIM sampler with motion prior")
            previous_motion_last_half = previous_motion_last_half.to(device)
        else:
            print("using diffusers DDIM sampler")

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.detach().cpu().numpy(),
            prediction_type="epsilon" if self.predict_type == "eps" else "sample",
            clip_sample=self.clip_denoised,
        )
        ddim_scheduler.eta = float(eta)
        ddim_scheduler.set_timesteps(sample_steps, device=device)

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]
        cond = cond.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates:
            motion_intermediates = [motion]

        for i, t in enumerate(tqdm(ddim_scheduler.timesteps, desc='sampling loop time step')):
            time_cond = torch.full((batch,), int(t.item()), device=device, dtype=torch.long)

            if is_cfg:
                no_cond_model_out = self.model.base_dit(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1)
                )
                cond_model_out = self.model.base_dit(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0)
                )
                model_out = no_cond_model_out + (cond_model_out - no_cond_model_out) * guidance_weight
            else:
                model_out = self.model.base_dit(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0)
                )

            if previous_motion_last_half is not None:
                model_out[:, :switch_length] = 0.0

            step_output = ddim_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample

            if previous_motion_last_half is not None and i < self.num_timesteps - 1:
                motion[0, :switch_length] = previous_motion_last_half[i+1]

            if return_intermediates:
                motion_intermediates.append(motion)

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0) # (num_intermediates, B, T, D)
            if motion_intermediates.shape[1] == 1:
                motion_intermediates = motion_intermediates.squeeze(1)
            else:
                raise ValueError("batch size should be 1 when return_intermediates is True")
            return motion, motion_intermediates

        return motion

    @torch.inference_mode()
    def ddim_sample_loop_streaming_text(self,
                                        cond,
                                        cond_text,
                                        shape,
                                        previous_motion_last_half,
                                        sample_steps=50,
                                        guidance_weight=3,
                                        eta=1.,
                                        is_cfg=True,
                                        return_intermediates=True):
        batch, device, total_timesteps = shape[0], self.betas.device, self.num_timesteps
        assert batch == 1

        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            print("previous motion last half shape: ", previous_motion_last_half.shape)
            assert switch_length < shape[1], "previous motion should be shorter than the current motion shape"
            print("using diffusers DDIM sampler with motion prior")
            previous_motion_last_half = previous_motion_last_half.to(device)
        else:
            print("using diffusers DDIM sampler")

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.detach().cpu().numpy(),
            prediction_type="epsilon" if self.predict_type == "eps" else "sample",
            clip_sample=self.clip_denoised,
        )
        ddim_scheduler.eta = float(eta)
        ddim_scheduler.set_timesteps(sample_steps, device=device)

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond = cond.to(device)
        cond_text = cond_text.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates:
            motion_intermediates = [motion]

        for i, t in enumerate(tqdm(ddim_scheduler.timesteps, desc='sampling loop time step')):
            time_cond = torch.full((batch,), int(t.item()), device=device, dtype=torch.long)

            if is_cfg:
                no_cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1)
                )
                cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0)
                )
                model_out = no_cond_model_out + (cond_model_out - no_cond_model_out) * guidance_weight
            else:
                model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=total_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0)
                )

            if previous_motion_last_half is not None:
                model_out[:, :switch_length] = 0.0

            step_output = ddim_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample

            if previous_motion_last_half is not None and i < self.num_timesteps - 1:
                motion[0, :switch_length] = previous_motion_last_half[i+1]

            if return_intermediates:
                motion_intermediates.append(motion)

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0) # (num_intermediates, B, T, D)
            if motion_intermediates.shape[1] == 1:
                motion_intermediates = motion_intermediates.squeeze(1)
            else:
                raise ValueError("batch size should be 1 when return_intermediates is True")
            return motion, motion_intermediates

        return motion

    @torch.no_grad()
    def unipc_sample_loop(self,
                          cond,
                          object_motion,
                          state,
                          shape,
                          sample_steps: int = 20,
                          guidance_weight: float = 3.0,
                          eta: float = 0.0,
                          is_cfg: bool = False,
                          return_intermediates: bool = False):
        print("using UniPCMultistepScheduler sampler")
        batch, device = shape[0], self.betas.device


        scheduler_kwargs = {
            "num_train_timesteps": self.num_timesteps,
            "beta_start": self.noise_scheduler.config.beta_start,
            "beta_end": self.noise_scheduler.config.beta_end,
            "beta_schedule": None,
            "trained_betas": self.betas.cpu().numpy(),
            "prediction_type": "epsilon" if self.predict_type == "eps" else "sample",
            "timestep_spacing": "trailing",
            "rescale_betas_zero_snr": False,
        }
        unipc_scheduler = UniPCMultistepScheduler(**scheduler_kwargs)
        unipc_scheduler.set_timesteps(sample_steps, device=device)

        motion = torch.randn(shape, device=device)
        cond = cond.to(device)
        if object_motion is not None:
            object_motion = object_motion.to(device)
        if state is not None:
            state = state.to(device)

        if return_intermediates:
            motion_intermediates = [motion]

        assert batch >= 1
        assert motion.shape[1] % 2 == 0
        half = motion.shape[1] // 2

        for i, t in enumerate(tqdm(unipc_scheduler.timesteps, desc='UniPC sampling loop time step')):

            time_cond = torch.full(
                (batch,),
                int(t.item()) if isinstance(t, torch.Tensor) else int(t),
                device=device,
                dtype=torch.long,
            )


            if is_cfg:
                no_cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=object_motion,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1),
                )
                cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=object_motion,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )
                model_out = no_cond_model_out + (cond_model_out - no_cond_model_out) * guidance_weight
            else:
                model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=object_motion,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )


            step_output = unipc_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample


            if i < len(unipc_scheduler.timesteps) - 1 and batch > 1:
                motion[1:, :half] = motion[:-1, half:]

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == len(unipc_scheduler.timesteps) - 1:
                    motion_intermediates.append(motion)

        if return_intermediates:
            return motion, motion_intermediates
        return motion

    @torch.no_grad()
    def unipc_sample_loop_text(self,
                               cond,
                               cond_text,
                               state,
                               shape,
                               sample_steps: int = 20,
                               guidance_weight: float = 3.0,
                               eta: float = 0.0,
                               is_cfg: bool = False,
                               return_intermediates: bool = False):
        print("using UniPCMultistepScheduler sampler (text-conditioned)")
        batch, device = shape[0], self.betas.device

        unipc_scheduler = UniPCMultistepScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.detach().cpu().numpy(),
            prediction_type="epsilon" if self.predict_type == "eps" else "sample",
            solver_order=2,
            solver_type="bh2",
            lower_order_final=True,
        )
        unipc_scheduler.set_timesteps(sample_steps, device=device)

        motion = torch.randn(shape, device=device)
        cond = cond.to(device)
        cond_text = cond_text.to(device)
        if state is not None:
            state = state.to(device)

        if return_intermediates:
            motion_intermediates = [motion]

        assert batch >= 1

        for i, t in enumerate(tqdm(unipc_scheduler.timesteps, desc='UniPC sampling loop time step (text)')):
            time_cond = torch.full(
                (batch,),
                int(t.item()) if isinstance(t, torch.Tensor) else int(t),
                device=device,
                dtype=torch.long,
            )

            if is_cfg:
                no_cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1),
                )
                cond_model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )
                model_out = no_cond_model_out + (cond_model_out - no_cond_model_out) * guidance_weight
            else:
                model_out = self.model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )

            step_output = unipc_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample

            if self.clip_denoised:
                motion = motion.clamp(-1.0, 1.0)

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == len(unipc_scheduler.timesteps) - 1:
                    motion_intermediates.append(motion)

        if return_intermediates:
            return motion, motion_intermediates
        return motion


    @torch.inference_mode()
    def unipc_sample_loop_streaming(
        self,
        cond,
        shape,
        previous_motion_last_half,
        sample_steps: int = 20,
        guidance_weight: float = 3.0,
        eta: float = 0.0,
        is_cfg: bool = True,
        return_intermediates: bool = True,
    ):
        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            print("previous motion last half shape: ", previous_motion_last_half.shape)
            assert switch_length < shape[1], "previous motion should be shorter than the current motion shape"
            print("using UniPC sampler with motion prior")
        else:
            print("using UniPC sampler (no motion prior)")

        batch, device = shape[0], self.betas.device
        assert batch == 1, "streaming  batch=1"


        unipc_scheduler = UniPCMultistepScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.detach().cpu().numpy(),
            prediction_type="epsilon" if self.predict_type == "eps" else "sample",
            solver_order=2,
            solver_type="bh2",
            lower_order_final=True,
        )
        unipc_scheduler.set_timesteps(sample_steps, device=device)

        model = self.model.to(device)
        motion = torch.randn(shape, device=device)

        if previous_motion_last_half is not None:

            motion[0, :switch_length] = previous_motion_last_half[0]

        cond = cond.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates:
            motion_intermediates = [motion.clone()]

        for i, t in enumerate(tqdm(unipc_scheduler.timesteps, desc="UniPC streaming steps")):
            time_cond = torch.full((batch,), t, device=device, dtype=torch.long)

            # ------ classifier-free guidance ------
            if is_cfg:

                no_cond_out = model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1),
                )

                cond_out = model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )
                model_out = no_cond_out + (cond_out - no_cond_out) * guidance_weight
            else:
                model_out = model(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )


            if previous_motion_last_half is not None:
                model_out[:, :switch_length] = 0.0

            step_output = unipc_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample


            if previous_motion_last_half is not None:
                idx = min(i + 1, previous_motion_last_half.shape[0] - 1)
                motion[0, :switch_length] = previous_motion_last_half[idx]

            if return_intermediates:
                motion_intermediates.append(motion.clone())


        motion_np = gaussian_filter1d(motion.cpu().numpy(), sigma=1, axis=1)
        motion = torch.from_numpy(motion_np).to(device)

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0)  # (steps+1, B, T, D)
            if motion_intermediates.shape[1] == 1:
                motion_intermediates = motion_intermediates.squeeze(1)
            else:
                raise ValueError("batch size should be 1 when return_intermediates is True")

            return motion, motion_intermediates

        return motion

    @torch.inference_mode()
    def unipc_sample_loop_streaming_controlnet(
        self,
        cond,
        shape,
        previous_motion_last_half,
        sample_steps: int = 20,
        guidance_weight: float = 3.0,
        eta: float = 0.0,
        is_cfg: bool = True,
        return_intermediates: bool = True,
    ):
        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            print("previous motion last half shape: ", previous_motion_last_half.shape)
            assert switch_length < shape[1], "previous motion should be shorter than the current motion shape"
            print("using UniPC sampler with motion prior (ControlNet)")
        else:
            print("using UniPC sampler (ControlNet, no prior)")

        batch, device = shape[0], self.betas.device
        assert batch == 1, "streaming  batch=1"

        unipc_scheduler = UniPCMultistepScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.detach().cpu().numpy(),
            prediction_type="epsilon" if self.predict_type == "eps" else "sample",
            solver_order=2,
            solver_type="bh2",
            lower_order_final=True,
        )
        unipc_scheduler.set_timesteps(sample_steps, device=device)

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond = cond.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates:
            motion_intermediates = [motion.clone()]

        base_dit = self.model.base_dit

        for i, t in enumerate(tqdm(unipc_scheduler.timesteps, desc="UniPC streaming steps (ControlNet)")):
            time_cond = torch.full((batch,), t, device=device, dtype=torch.long)

            if is_cfg:
                no_cond_out = base_dit(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1),
                )
                cond_out = base_dit(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )
                model_out = no_cond_out + (cond_out - no_cond_out) * guidance_weight
            else:
                model_out = base_dit(
                    inputs=motion,
                    cond_a=cond,
                    cond_m=None,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )

            if previous_motion_last_half is not None:
                model_out[:, :switch_length] = 0.0

            step_output = unipc_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample

            if previous_motion_last_half is not None:
                idx = min(i + 1, previous_motion_last_half.shape[0] - 1)
                motion[0, :switch_length] = previous_motion_last_half[idx]

            if return_intermediates:
                motion_intermediates.append(motion.clone())

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0)
            if motion_intermediates.shape[1] == 1:
                motion_intermediates = motion_intermediates.squeeze(1)
            else:
                raise ValueError("batch size should be 1 when return_intermediates is True")

            return motion, motion_intermediates

        return motion

    @torch.inference_mode()
    def unipc_sample_loop_streaming_text(
        self,
        cond,
        cond_text,
        shape,
        previous_motion_last_half,
        sample_steps: int = 20,
        guidance_weight: float = 3.0,
        eta: float = 0.0,
        is_cfg: bool = True,
        return_intermediates: bool = True,
    ):
        if previous_motion_last_half is not None:
            switch_length = previous_motion_last_half.shape[1]
            print("previous motion last half shape: ", previous_motion_last_half.shape)
            assert switch_length < shape[1], "previous motion should be shorter than the current motion shape"
            print("using UniPC sampler with motion prior (text-cond)")
        else:
            print("using UniPC sampler (text-cond, no prior)")

        batch, device = shape[0], self.betas.device
        assert batch == 1, "streaming  batch=1"

        unipc_scheduler = UniPCMultistepScheduler(
            num_train_timesteps=self.num_timesteps,
            trained_betas=self.betas.detach().cpu().numpy(),
            prediction_type="epsilon" if self.predict_type == "eps" else "sample",
            solver_order=2,
            solver_type="bh2",
            lower_order_final=True,
        )
        unipc_scheduler.set_timesteps(sample_steps, device=device)

        motion = torch.randn(shape, device=device)
        if previous_motion_last_half is not None:
            motion[0, :switch_length] = previous_motion_last_half[0]

        cond = cond.to(device)
        cond_text = cond_text.to(device)
        state = torch.ones((batch, shape[1]), device=device)

        if return_intermediates:
            motion_intermediates = [motion.clone()]

        model = self.model.to(device)

        for i, t in enumerate(tqdm(unipc_scheduler.timesteps, desc="UniPC streaming steps (text)")):
            time_cond = torch.full((batch,), t, device=device, dtype=torch.long)

            if is_cfg:
                no_cond_out = model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(1, 1, 1),
                )
                cond_out = model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )
                model_out = no_cond_out + (cond_out - no_cond_out) * guidance_weight
            else:
                model_out = model(
                    inputs=motion,
                    cond_a=cond,
                    cond_t=cond_text,
                    state=state,
                    time=time_cond,
                    processing_state=self.processing_state,
                    last_time_stamp=self.num_timesteps - 1,
                    cfg_cond_drop_prob=(0, 0, 0),
                )

            if previous_motion_last_half is not None:
                model_out[:, :switch_length] = 0.0

            step_output = unipc_scheduler.step(
                model_output=model_out,
                timestep=t,
                sample=motion,
            )
            motion = step_output.prev_sample

            if previous_motion_last_half is not None:
                idx = min(i + 1, previous_motion_last_half.shape[0] - 1)
                motion[0, :switch_length] = previous_motion_last_half[idx]

            if return_intermediates:
                motion_intermediates.append(motion.clone())

        if return_intermediates:
            motion_intermediates = torch.stack(motion_intermediates, dim=0)
            if motion_intermediates.shape[1] == 1:
                motion_intermediates = motion_intermediates.squeeze(1)
            else:
                raise ValueError("batch size should be 1 when return_intermediates is True")

            return motion, motion_intermediates

        return motion
