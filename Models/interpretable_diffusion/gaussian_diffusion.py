import copy
import math
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from einops import reduce
from tqdm import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract
from Utils.watermark_utils import (
    get_watermarking_mask,
    inject_watermark,
    get_watermarking_pattern,
)


# gaussian diffusion trainer class


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    def __init__(
        self,
        seq_length,
        feature_size,
        n_layer_enc=3,
        n_layer_dec=6,
        d_model=None,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type="l1",
        beta_schedule="cosine",
        n_heads=4,
        mlp_hidden_times=4,
        eta=0.0,
        attn_pd=0.0,
        resid_pd=0.0,
        kernel_size=None,
        padding_size=None,
        use_ff=True,
        reg_weight=None,
        bdia_gamma=0.99,
        **kwargs,
    ):
        super(Diffusion_TS, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)
        self.bdia_gamma = bdia_gamma

        self.model = Transformer(
            n_feat=feature_size,
            n_channel=seq_length,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length,
            n_embd=d_model,
            conv_params=[kernel_size, padding_size],
            **kwargs,
        )

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate reweighting

        register_buffer(
            "loss_weight",
            torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100,
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, t, padding_masks=None):
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    def model_predictions(self, x, t, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(
                x.shape[0], self.seq_length, dtype=bool, device=x.device
            )

        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )
        x_start = self.output(x, t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        _, x_start = self.model_predictions(x, t)
        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def inverse_fast_sample(self, img, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            img.shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )
        img = img.to(device)

        # [0, 1, 2, ..., T-1, T] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = times.int().tolist()
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = time_pairs[1:]
        for time, time_next in tqdm(time_pairs, desc="inverse sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            alpha = self.alphas_cumprod[time]  # t
            alpha_next = self.alphas_cumprod[time_next]  # t+1

            img = (img - (1 - alpha).sqrt() * pred_noise) * (
                alpha_next.sqrt() / alpha.sqrt()
            ) + (1 - alpha_next).sqrt() * pred_noise

        return img

    def step_bdia(self, model_output, timestep, sample, gamma=0.98):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_timesteps // self.sampling_timesteps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.alphas_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # Calculate coefficients a and b
        a_t = (alpha_prod_t_prev**0.5) / (alpha_prod_t**0.5)
        b_t = (1 - alpha_prod_t_prev) ** 0.5 - (
            (alpha_prod_t_prev**0.5) * (beta_prod_t**0.5)
        ) / (alpha_prod_t**0.5)

        next_timestep = timestep + self.num_timesteps // self.sampling_timesteps
        alpha_prod_t_next = self.alphas_cumprod[next_timestep]
        beta_prod_t_next = 1 - alpha_prod_t_next

        a_t_plus_1 = (alpha_prod_t**0.5) / (alpha_prod_t_next**0.5)
        b_t_plus_1 = (1 - alpha_prod_t) ** 0.5 - (
            (alpha_prod_t**0.5) * (beta_prod_t_next**0.5)
        ) / (alpha_prod_t_next**0.5)

        first_term = sample[0]
        second_term = (1 / a_t_plus_1 - 1) * sample[
            1
        ] - b_t_plus_1 / a_t_plus_1 * model_output
        third_term = (a_t - 1) * sample[1] + b_t * model_output
        prev_sample = (
            first_term
            - (1 - gamma) * (sample[0] - sample[1])
            - gamma * second_term
            + third_term
        )
        return prev_sample

    def step_ddim(self, model_output, timestep, sample):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_timesteps // self.sampling_timesteps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.alphas_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        # 5/6. compute "direction pointing to x_t" without additional noise
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        # Calculate coefficients a and b

        return prev_sample

    @torch.no_grad()
    def bdia_sample(self, shape, img, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
        )
        gamma = self.bdia_gamma
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(
            total_timesteps // sampling_timesteps,
            total_timesteps - total_timesteps // sampling_timesteps,
            steps=sampling_timesteps - 1,
        )
        # reverse the time steps
        timesteps = list(reversed(times.int().tolist()))
        x_pairs = [img.clone(), img.clone()]
        for t in tqdm(timesteps, desc="sampling BDIA loop time step"):
            time_cond = torch.full((batch,), t, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                x_pairs[1], time_cond, clip_x_start=clip_denoised
            )
            if t == timesteps[0]:
                x_T_minus_1 = self.step_ddim(pred_noise, t, x_pairs[0])
                x_pairs[1] = x_T_minus_1
            else:
                x_t_minus_1 = self.step_bdia(pred_noise, t, x_pairs, gamma=gamma)
                x_pairs[0] = x_pairs[1]
                x_pairs[1] = x_t_minus_1
        return x_pairs[1]

    @torch.no_grad()
    def inverse_bdia_sample(self, img, clip_denoised=True, aux=None):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            img.shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )
        gamma = self.bdia_gamma
        img = img.to(device)

        # [0, 1, 2, ..., T-1, T] when sampling_timesteps == total_timesteps
        times = torch.linspace(
            0,
            total_timesteps - total_timesteps // sampling_timesteps,
            steps=sampling_timesteps,
        )
        times = times.int().tolist()
        x_pairs = [img.clone(), img.clone()]
        for t in tqdm(times[:-1], desc="inverse BDIA sampling loop time step"):
            if t == times[0]:
                if aux is not None:
                    x_pairs = [aux, x_pairs[1]]
                else:
                    x_pairs = [x_pairs[1], x_pairs[1]]

            else:
                time_cond = torch.full((batch,), t, device=device, dtype=torch.long)
                pred_noise, x_start, *_ = self.model_predictions(
                    x_pairs[1], time_cond, clip_x_start=clip_denoised
                )
                x_t_plus_1 = self.reverse_step_BDIA(pred_noise, t, x_pairs, gamma=gamma)
                x_pairs[0] = x_pairs[1]
                x_pairs[1] = x_t_plus_1
        return x_pairs[1]

    def reverse_step_BDIA(self, model_output, timestep: int, sample, gamma=0.99):
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_timesteps // self.sampling_timesteps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.alphas_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        # Calculate coefficients a and b
        a_t = (alpha_prod_t_prev**0.5) / (alpha_prod_t**0.5)
        b_t = (1 - alpha_prod_t_prev) ** 0.5 - (
            (alpha_prod_t_prev**0.5) * (beta_prod_t**0.5)
        ) / (alpha_prod_t**0.5)

        next_timestep = timestep + self.num_timesteps // self.sampling_timesteps
        alpha_prod_t_next = self.alphas_cumprod[next_timestep]
        beta_prod_t_next = 1 - alpha_prod_t_next

        a_t_plus_1 = (alpha_prod_t**0.5) / (alpha_prod_t_next**0.5)
        b_t_plus_1 = (1 - alpha_prod_t) ** 0.5 - (
            (alpha_prod_t**0.5) * (beta_prod_t_next**0.5)
        ) / (alpha_prod_t_next**0.5)

        first_term = sample[0]
        second_term = a_t * sample[1] + b_t * model_output
        third_term = (1 / a_t_plus_1) * sample[
            1
        ] - b_t_plus_1 / a_t_plus_1 * model_output
        next_sample = first_term / gamma - second_term / gamma + third_term
        return next_sample

    @torch.no_grad()
    def sample(self, shape, img):
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, _ = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def fast_sample(self, shape, img, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]  # t
            alpha_next = self.alphas_cumprod[time_next]  # t-1
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img

    def generate_mts(self, args, batch_size=16, watermark=""):
        shape = (batch_size, self.seq_length, self.feature_size)
        device = self.betas.device

        if watermark in ["TimeWak", "SpatBDIA"]:
            sample_fn = self.bdia_sample if self.fast_sampling else self.sample
        else:
            sample_fn = self.fast_sample if self.fast_sampling else self.sample

        init_latents = torch.randn(shape, device=device)

        if watermark == "TR":
            st0 = torch.get_rng_state()
            torch.manual_seed(217)

            latents = torch.empty((0, shape[1], shape[2]), device=device)
            gt_patches = np.empty([0, 1, shape[1], shape[2]])
            watermarking_masks = np.empty([0, 1, shape[1], shape[2]], dtype=bool)

            for init_latent in init_latents:
                # change from two-dimensional table into watermark size [1, c, l, w]
                init_latent = init_latent.unsqueeze(0).unsqueeze(0)
                init_latent_w = copy.deepcopy(init_latent)
                gt_patch = get_watermarking_pattern(
                    args, device, shape=init_latent_w.shape
                )
                # get watermarking mask
                watermarking_mask = get_watermarking_mask(init_latent_w, args, device)
                # inject watermark
                latent = inject_watermark(
                    init_latent_w, watermarking_mask, gt_patch, args
                )
                latent = latent.squeeze(0)  # (bs, seq_len, feat)

                latents = torch.vstack((latents, latent))
                gt_patches = np.row_stack([gt_patches, gt_patch.detach().cpu().numpy()])
                watermarking_masks = np.row_stack(
                    [watermarking_masks, watermarking_mask.detach().cpu().numpy()]
                )
            latents = latents.to(device)
            torch.set_rng_state(st0)

            return sample_fn(shape, latents), gt_patches, watermarking_masks

        elif watermark == "GS":
            st0 = torch.get_rng_state()
            torch.manual_seed(217)

            latents = torch.zeros_like(init_latents)
            latent_seed = torch.randint(
                0, 2, (init_latents.shape[1], init_latents.shape[2])
            )

            for i in range(init_latents.shape[0]):  # Loop through each sample
                for j in range(init_latents.shape[1]):  # Loop through each time steps
                    for k in range(init_latents.shape[2]):  # Loop through each features
                        if latent_seed[j, k] == 0:
                            # Even index, sample from the left half of the Gaussian distribution
                            while True:
                                sample = torch.randn(1)
                                if sample < 0:
                                    latents[i, j, k] = sample
                                    break
                        else:
                            while True:
                                sample = torch.randn(1)
                                if sample >= 0:
                                    latents[i, j, k] = sample
                                    break

            latents = latents.to(device)
            torch.set_rng_state(st0)

        elif watermark in ["TabWak", "TabWakT"]:
            # Reshape the latents based on token_dim
            if watermark == "TabWakT":
                init_latents = init_latents.permute(0, 2, 1)
            bit_dim = init_latents.shape[2]
            if bit_dim % 2 != 0:
                bit_dim -= 1
                found = False
                while not found:
                    try:
                        init_latents.reshape(init_latents.shape[0], -1, bit_dim)
                        found = True
                    except:
                        bit_dim -= 1
            bit_string = init_latents.reshape(init_latents.shape[0], -1, bit_dim)  # 3D
            bit_string_4bits = (bit_string > 0).int()

            condition_0 = bit_string <= -0.67449
            condition_1 = bit_string >= 0.67449
            condition_2 = (bit_string > -0.67449) & (bit_string < 0)
            condition_3 = (bit_string > 0) & (bit_string < 0.67449)

            # Apply conditions to update bit_string_4bits
            bit_string_4bits[condition_0] = 0
            bit_string_4bits[condition_1] = 1
            bit_string_4bits[condition_2] = 2
            bit_string_4bits[condition_3] = 3

            # Split the bit_string into two equal parts for each row
            split_dim = bit_dim // 2
            bit_string_4bits[:, :, split_dim:] = bit_string_4bits[
                :, :, :split_dim
            ].clone()

            adjusted_bit_string = bit_string_4bits.reshape(
                -1, init_latents.shape[2]
            )  # 2D

            st0 = torch.get_rng_state()
            torch.manual_seed(217)

            permutation = torch.randperm(init_latents.shape[2])
            adjusted_bit_string = adjusted_bit_string[:, permutation]

            # Initialize normal distribution
            normal_dist = torch.distributions.Normal(0, 1)

            # Sample based on adjusted_bit_string values
            random_samples = torch.rand(adjusted_bit_string.shape, device=device)

            # Pre-allocate output tensor
            latents = torch.empty_like(adjusted_bit_string, dtype=torch.float32)

            # Generate quantiles in vectorized manner for each condition
            latents[adjusted_bit_string == 0] = normal_dist.icdf(
                random_samples[adjusted_bit_string == 0] * 0.25
            )
            latents[adjusted_bit_string == 1] = normal_dist.icdf(
                random_samples[adjusted_bit_string == 1] * 0.25 + 0.75
            )
            latents[adjusted_bit_string == 2] = normal_dist.icdf(
                random_samples[adjusted_bit_string == 2] * 0.25 + 0.25
            )
            latents[adjusted_bit_string == 3] = normal_dist.icdf(
                random_samples[adjusted_bit_string == 3] * 0.25 + 0.5
            )
            latents = latents.to(device)
            if watermark == "TabWakT":
                latents = latents.reshape(shape[0], shape[2], shape[1])
                latents = latents.permute(0, 2, 1)
            else:
                latents = latents.reshape(shape)
            torch.set_rng_state(st0)

        elif watermark in ["TimeWak", "TempDDIM", "SpatDDIM", "SpatBDIA"]:
            if watermark in ["SpatDDIM", "SpatBDIA"]:
                init_latents = init_latents.permute(0, 2, 1)
            bits = torch.randint_like(init_latents, args.bits)
            seed_idx = torch.arange(0, bits.shape[1], args.interval)

            for i in range(bits.shape[1]):  # timesteps
                st0 = torch.get_rng_state()
                torch.manual_seed(i)
                permutation = torch.randperm(bits.shape[2])
                if i in seed_idx:
                    bits[:, i, :] = bits[:, i, permutation]
                else:
                    bits[:, i, :] = bits[:, i - 1, permutation]
                torch.set_rng_state(st0)

            for i in range(bits.shape[2]):  # features
                st0 = torch.get_rng_state()
                torch.manual_seed(i)
                permutation = torch.randperm(bits.shape[1])
                bits[:, :, i] = bits[:, permutation, i]
                torch.set_rng_state(st0)

            st0 = torch.get_rng_state()
            torch.manual_seed(217)

            # Initialize normal distribution
            normal_dist = torch.distributions.Normal(0, 1)

            # Sample based on adjusted_bit_string values
            random_samples = torch.rand(bits.shape, device=device)

            # Pre-allocate output tensor
            latents = torch.empty_like(bits, dtype=torch.float32)

            # Generate quantiles in vectorized manner for each condition
            if args.bits == 2:
                latents[bits == 0] = normal_dist.icdf(random_samples[bits == 0] * 0.5)
                latents[bits == 1] = normal_dist.icdf(
                    random_samples[bits == 1] * 0.5 + 0.5
                )

            elif args.bits == 3:
                latents[bits == 0] = normal_dist.icdf(random_samples[bits == 0] * 0.33)
                latents[bits == 1] = normal_dist.icdf(
                    random_samples[bits == 1] * 0.33 + 0.67
                )
                latents[bits == 2] = normal_dist.icdf(
                    random_samples[bits == 2] * 0.33 + 0.33
                )

            elif args.bits == 4:
                latents[bits == 0] = normal_dist.icdf(random_samples[bits == 0] * 0.25)
                latents[bits == 1] = normal_dist.icdf(
                    random_samples[bits == 1] * 0.25 + 0.75
                )
                latents[bits == 2] = normal_dist.icdf(
                    random_samples[bits == 2] * 0.25 + 0.25
                )
                latents[bits == 3] = normal_dist.icdf(
                    random_samples[bits == 3] * 0.25 + 0.5
                )

            else:
                raise NotImplementedError()

            latents = latents.to(device)
            if watermark in ["SpatDDIM", "SpatBDIA"]:
                latents = latents.permute(0, 2, 1)
            torch.set_rng_state(st0)

        else:
            latents = init_latents
        return sample_fn(shape, latents)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
        model_out = self.output(x, t, padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction="none")

        fourier_loss = torch.tensor([0.0])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm="forward")
            fft2 = torch.fft.fft(target.transpose(1, 2), norm="forward")
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(
                torch.real(fft1), torch.real(fft2), reduction="none"
            ) + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction="none")
            train_loss += self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, "b ... -> b (...)", "mean")
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, **kwargs):
        b, c, n, device, feature_size = *x.shape, x.device, self.feature_size
        assert n == feature_size, f"number of variable must be {feature_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size = *x.shape, x.device, self.feature_size
        assert n == feature_size, f"number of variable must be {feature_size}"
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(
        self,
        shape,
        target,
        sampling_timesteps,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        batch, device, total_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.eta,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(
            time_pairs, desc="conditional sampling loop time step"
        ):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(
                sample=img,
                mean=pred_mean,
                sigma=sigma,
                t=time_cond,
                tgt_embs=target,
                partial_mask=partial_mask,
                **model_kwargs,
            )
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape,
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="conditional sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample_infill(
                x=img,
                t=t,
                clip_denoised=clip_denoised,
                target=target,
                partial_mask=partial_mask,
                model_kwargs=model_kwargs,
            )

        img[partial_mask] = target[partial_mask]
        return img

    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(
            sample=pred_img,
            mean=model_mean,
            sigma=sigma,
            t=batched_times,
            tgt_embs=target,
            partial_mask=partial_mask,
            **model_kwargs,
        )

        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.0,
    ):

        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = (
                        coef * ((mean - input_embs_param) ** 2 / 1.0).mean(dim=0).sum()
                    )
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = (
                        coef
                        * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
                    )
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss / sigma.mean()).mean(dim=0).sum()

                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter(
                    (
                        input_embs_param.data + coef_ * sigma.mean().item() * epsilon
                    ).detach()
                )

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample


if __name__ == "__main__":
    pass
