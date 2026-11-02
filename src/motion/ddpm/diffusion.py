import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from tqdm import tqdm
import os
import time
from typing import Any, Optional, Union, List
from functools import partial
from contextlib import contextmanager

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


class Gesture_Diffusion(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 schedule_type: str = "linear",
                 predict_type: str = "eps",
                 n_timestep: int = 1000,
                 linear_start: float = 1e-4,
                 linear_end: float = 8e-2,
                 cosine_s: float = 8e-3,
                 cfg_cond_drop_prob: float = 0.3,
                 # for training
                 loss_type: str = "l2",
                 l_simple_weight: float = 1.0,
                 original_elbo_weight: float = 0.0,
                 # for inference
                 clip_denoised: bool = False,
                 save_intermediates_every_t: int = 10
                ):
        super(Gesture_Diffusion, self).__init__()
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
        self.cfg_cond_drop_prob = cfg_cond_drop_prob

        self.register_schedule(schedule_type, n_timestep, linear_start, linear_end, cosine_s)


    def register_schedule(self, schedule_type, n_timestep, linear_start, linear_end, cosine_s):
        # beta schedule
        betas = make_beta_schedule(schedule_type=schedule_type, n_timestep=n_timestep, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

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

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_variance", to_torch(posterior_variance))

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))


        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))

        self.register_buffer("posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        # level of noise in the diffusion training process
        if self.predict_type == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.predict_type == "x0":
            lvlb_weights = 0.5 * to_torch(np.sqrt(alphas_cumprod)) / (2. * 1 - self.alphas_cumprod)
        else:
            raise NotImplementedError("mu not supported")

        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

        assert not torch.isnan(self.lvlb_weights).all()


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
                    (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
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


    def p_losses(self, x_start, cond, timestep, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=timestep, noise=noise)
        model_out = self.model(inputs=x_t, cond=cond, time=timestep, processing_state=self.processing_state, last_time_stamp=self.num_timesteps, cfg_cond_drop_prob=self.cfg_cond_drop_prob)

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

    def p_losses_text(self, x_start, cond, cond_text, timestep, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=timestep, noise=noise)
        model_out = self.model(inputs=x_t, cond_a=cond, cond_t=cond_text, time=timestep, processing_state=self.processing_state, last_time_stamp=self.num_timesteps, cfg_cond_drop_prob=self.cfg_cond_drop_prob)

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

    def forward(self, x: torch.Tensor, cond: torch.Tensor, state=None):
        B, _, _ = x.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=x.device).long()  # [B,] the noise step index

        return self.p_losses(x, cond, t)

    def forward_text(self, x: torch.Tensor, cond: torch.Tensor, cond_text: torch.Tensor, state=None):
        B, _, _ = x.shape
        t = torch.randint(0, self.num_timesteps, (B,), device=x.device).long()  # [B,] the noise step index

        return self.p_losses_text(x, cond, cond_text, t)

    # ------------------------------------------ inference ------------------------------------------#

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):

        model_out = self.model(inputs=x, cond=cond, time=t, processing_state="test", last_time_stamp=self.num_timesteps-1)
        if self.predict_type == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.predict_type == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))


        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return x_out


    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]

        assert shape[1] % 2 == 0
        half = shape[1] // 2

        motion = torch.randn(shape, device=device)  # [B, 400, 65]
        if return_intermediates:
            motion_intermediates = [motion]

        for i in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps, leave=False, position=1, desc='Sample T'):
            current_timestamp = torch.full((b,), i, device=device, dtype=torch.long)
            motion = self.p_sample(x=motion, cond=cond, t=current_timestamp, clip_denoised=self.clip_denoised)


            # if b > 1 and i < self.num_timesteps - 1:
            #     # the first half of each sequence is the second half of the previous one
            #     motion[1:, :half] = motion[:-1, half:]

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == self.num_timesteps - 1:
                    motion_intermediates.append(motion)
        if return_intermediates:
            return motion, motion_intermediates

        return motion


    @torch.no_grad()
    def ddim_sample_loop(self,
                        cond,
                        shape,
                        sample_steps=50,
                        guidance_weight=3,
                        eta=0.,
                        return_intermediates=False):
        print("using ddim_edge sampler")
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, sample_steps, eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, guidance_weight * 2, sampling_timesteps), None, guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


        # motion = np.load('./data/interact_select_normal_stand_talk/processed_5s_30fps_flip_add_resample_mirror/feat_30fps_5s_WithExp_waveform_WithStd/test_output_101_310_a_30fps.npz')['clips']
        # motion = torch.from_numpy(motion).to(device)
        # batch = motion.shape[0]
        # time_stamp = torch.full((batch,), self.num_timesteps-1, device=device, dtype=torch.long)
        # print(time_stamp.shape)
        # motion = self.q_sample(x_start=motion, t=time_stamp, noise=None)
        # cond = cond[:batch]

        motion = torch.randn(shape, device=device)
        cond = cond.to(device)

        if return_intermediates:
            motion_intermediates = [motion]

        assert batch >= 1
        assert motion.shape[1] % 2 == 0
        half = motion.shape[1] // 2

        x_recon = None

        # for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
        for i, (time, time_next, weight) in enumerate(tqdm(time_pairs, desc='sampling loop time step')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            no_cond_model_out = self.model(inputs=motion, cond=cond, time=time_cond, processing_state=self.processing_state, last_time_stamp=self.num_timesteps-1, cfg_cond_drop_prob=1)
            cond_model_out = self.model(inputs=motion, cond=cond, time=time_cond, processing_state=self.processing_state, last_time_stamp=self.num_timesteps-1, cfg_cond_drop_prob=0)

            model_out = no_cond_model_out + (cond_model_out - no_cond_model_out) * weight
            if self.predict_type == "eps":
                x_recon = self.predict_start_from_noise(motion, t=time_cond, noise=model_out)
                pred_noise = model_out
            elif self.predict_type == "x0":
                x_recon = model_out
                pred_noise = self.predict_noise_from_start(motion, t=time_cond, x0=x_recon)

            if time_next < 0:
                # print(i, time, "finish ddim sample")
                motion = x_recon
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]


            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(motion)


            # if time_next < 0:
            #     motion = x_recon + sigma * noise
            #     continue

            motion = x_recon * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # print(i, motion.abs().mean())

            if time > 0 and batch > 1:
                # the first half of each sequence is the second half of the previous one
                # print(i, time, "the first half of each sequence is the second half of the previous one")
                motion[1:, :half] = motion[:-1, half:]

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == self.num_timesteps - 1:
                    motion_intermediates.append(motion)

        if return_intermediates:
            return motion, motion_intermediates

        return motion

    @torch.no_grad()
    def ddim_sample_loop_text(self,
                        cond,
                        cond_text,
                        shape,
                        sample_steps=50,
                        guidance_weight=3,
                        eta=0.,
                        return_intermediates=False):
        print("using ddim_edge sampler")
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, sample_steps, eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, guidance_weight * 2, sampling_timesteps), None, guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]


        # motion = np.load('./data/interact_select_normal_stand_talk/processed_5s_30fps_flip_add_resample_mirror/feat_30fps_5s_WithExp_waveform_WithStd/test_output_101_310_a_30fps.npz')['clips']
        # motion = torch.from_numpy(motion).to(device)
        # batch = motion.shape[0]
        # time_stamp = torch.full((batch,), self.num_timesteps-1, device=device, dtype=torch.long)
        # print(time_stamp.shape)
        # motion = self.q_sample(x_start=motion, t=time_stamp, noise=None)
        # cond = cond[:batch]

        motion = torch.randn(shape, device=device)
        cond = cond.to(device)
        cond_text = cond_text.to(device)

        if return_intermediates:
            motion_intermediates = [motion]

        assert batch >= 1
        assert motion.shape[1] % 2 == 0
        half = motion.shape[1] // 2

        x_recon = None

        # for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
        for i, (time, time_next, weight) in enumerate(tqdm(time_pairs, desc='sampling loop time step')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            no_cond_model_out = self.model(inputs=motion, cond_a=cond, cond_t=cond_text, time=time_cond, processing_state=self.processing_state, last_time_stamp=self.num_timesteps-1, cfg_cond_drop_prob=1)
            cond_model_out = self.model(inputs=motion, cond_a=cond, cond_t=cond_text, time=time_cond, processing_state=self.processing_state, last_time_stamp=self.num_timesteps-1, cfg_cond_drop_prob=0)

            model_out = no_cond_model_out + (cond_model_out - no_cond_model_out) * weight
            if self.predict_type == "eps":
                x_recon = self.predict_start_from_noise(motion, t=time_cond, noise=model_out)
                pred_noise = model_out
            elif self.predict_type == "x0":
                x_recon = model_out
                pred_noise = self.predict_noise_from_start(motion, t=time_cond, x0=x_recon)

            if time_next < 0:
                # print(i, time, "finish ddim sample")
                motion = x_recon
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]


            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(motion)


            # if time_next < 0:
            #     motion = x_recon + sigma * noise
            #     continue

            motion = x_recon * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # print(i, motion.abs().mean())

            if time > 0 and batch > 1:
                # the first half of each sequence is the second half of the previous one
                # print(i, time, "the first half of each sequence is the second half of the previous one")
                motion[1:, :half] = motion[:-1, half:]

            if return_intermediates:
                if i % self.save_intermediates_every_t == 0 or i == self.num_timesteps - 1:
                    motion_intermediates.append(motion)

        if return_intermediates:
            return motion, motion_intermediates

        return motion
