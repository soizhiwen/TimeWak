import copy
import torch
import numpy as np
import hashlib
import random
from tqdm import tqdm


def circle_mask(height=17117, width=44, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = width // 2
    y0 = height // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:height, :width]
    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2


def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == "circle":
        np_mask = circle_mask(
            height=init_latents_w.shape[-2],
            width=init_latents_w.shape[-1],
            r=args.w_radius,
        )
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == "square":
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[
                :,
                :,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
            ] = True
        else:
            watermarking_mask[
                :,
                args.w_channel,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
            ] = True
    elif args.w_mask_shape == "no":
        pass
    else:
        raise NotImplementedError(f"w_mask_shape: {args.w_mask_shape}")

    return watermarking_mask


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(
        torch.fft.fft2(init_latents_w), dim=(-1, -2)
    )
    if args.w_injection == "complex":
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == "seed":
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f"w_injection: {args.w_injection}")

    init_latents_w = torch.fft.ifft2(
        torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))
    ).real

    return init_latents_w


def get_watermarking_pattern(args, device, shape):
    gt_init = torch.randn(shape, device=device)

    if "seed_ring" in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-2], gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif "seed_zeros" in args.w_pattern:
        gt_patch = gt_init * 0
    elif "seed_rand" in args.w_pattern:
        gt_patch = gt_init
    elif "rand" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif "zeros" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif "const" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif "ring" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-2], gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, i, 0].item()

    return gt_patch


def htw_get_smallest_increment(number):
    str_number = str(number)
    if "." in str_number:
        decimal_part = str_number.split(".")[-1]
        increment = 10 ** -len(decimal_part)
    else:
        increment = 1  # Default increment for whole numbers
    return increment


def pseudorandom(x, z, p, precision=10):
    T = p - z
    x_norm = (x - z) / T  # Normalize between 0 and 1

    x_norm_str = f"{x_norm:.{precision}f}"

    seed = hashlib.sha256(x_norm_str.encode()).hexdigest()

    random.seed(seed)
    y = random.random()

    return y


def coin_flip(value):
    return 0 if value <= 0.5 else 1


def htw_watermark_series(xs, theta):
    n = len(xs)
    heads_target = int(n * theta)
    tails_target = n - heads_target

    heads_count = 0
    tails_count = 0

    watermarked_series = []

    z = min(xs)
    p = max(xs)

    for x in xs:
        if x == z or x == p:
            watermarked_series.append(x)
            continue

        pseudo_random_value = pseudorandom(x, z, p)
        heads_or_tails = coin_flip(pseudo_random_value)

        is_heads = heads_or_tails == 0

        if (is_heads and heads_count < heads_target) or (
            not is_heads and tails_count < tails_target
        ):
            watermarked_series.append(x)
            heads_count += 1 if is_heads else 0
            tails_count += 1 if not is_heads else 0
        else:
            # Modify the number to the opposite type
            if is_heads:
                while heads_or_tails == 0:
                    increment = htw_get_smallest_increment(x)
                    x = round(
                        x + increment,
                        (
                            len(str(increment).split(".")[-1])
                            if "." in str(increment)
                            else 0
                        ),
                    )

                    pseudo_random_value = pseudorandom(x, z, p)
                    heads_or_tails = coin_flip(pseudo_random_value)
                tails_count += 1
            else:
                while heads_or_tails == 1:
                    increment = htw_get_smallest_increment(x)
                    x = round(
                        x - increment,
                        (
                            len(str(increment).split(".")[-1])
                            if "." in str(increment)
                            else 0
                        ),
                    )

                    pseudo_random_value = pseudorandom(x, z, p)
                    heads_or_tails = coin_flip(pseudo_random_value)
                heads_count += 1
            watermarked_series.append(x)

    return watermarked_series


def htw_watermark_mts(synth_data, theta=1):
    watermarked_samples = np.empty([0, synth_data.shape[1], synth_data.shape[2]])

    for sample in tqdm(synth_data, desc="HTW watermarking"):
        w_series = np.empty([0, sample.shape[0]])
        for series in sample.T:
            htw_watermarked_values = htw_watermark_series(series, theta)
            w_series = np.row_stack([w_series, htw_watermarked_values])

        w_series = w_series.T[np.newaxis, :, :]
        watermarked_samples = np.row_stack([watermarked_samples, w_series])

    return watermarked_samples


def htw_check_distribution(xs):
    z = min(xs)
    p = max(xs)

    heads_count = 0

    for x in xs:
        pseudo_random_value = pseudorandom(x, z, p)
        if coin_flip(pseudo_random_value) == 0:
            heads_count += 1

    tails_count = len(xs) - heads_count
    total = len(xs)
    return heads_count


def htw_calculate_z_score(series):
    n = len(series)
    expected_even_count = 0.5 * n
    std_dev = (n * 0.5 * 0.5) ** 0.5

    actual_even_count = htw_check_distribution(series)

    z_score = (actual_even_count - expected_even_count) / std_dev
    return z_score


def add_attack(data, attack, attack_factor):
    if attack == "offset":
        data = offset_attack(data, attack_factor)
    elif attack == "crop":
        data = crop_attack(data, attack_factor)
    elif attack == "insert":
        data = insert_attack(data, attack_factor)
    else:
        raise NotImplementedError()
    return data


def offset_attack(data, attack_factor):
    offset = data.mean(axis=1) * attack_factor
    offset = offset[:, np.newaxis, :]
    return data + offset


def crop_attack(data, attack_factor):
    bs, height, width = data.shape
    new_width = int(width * (1 - attack_factor))
    new_height = int(height * (1 - attack_factor))
    padded_data = np.zeros_like(data)

    for i in range(bs):
        start_x = np.random.randint(0, width - new_width + 1)
        start_y = np.random.randint(0, height - new_height + 1)
        end_x = start_x + new_width
        end_y = start_y + new_height
        padded_data[i, start_y:end_y, start_x:end_x] = data[
            i, start_y:end_y, start_x:end_x
        ]
    return padded_data


def insert_attack(data, attack_factor):
    n = data.shape[1]  # seq len
    num_replacements = int(n * attack_factor)

    for i, sample in enumerate(data):
        for j in range(sample.shape[1]):  # feat dim
            # Determine min and max of the series
            min_val = np.min(sample[:, j])
            max_val = np.max(sample[:, j])

            # Generate random positions to replace
            positions = np.random.choice(
                sample.shape[0], size=num_replacements, replace=False
            )

            # Replace the values at the chosen positions with random values
            data[i, positions, j] = np.random.uniform(
                min_val, max_val, size=num_replacements
            )
    return data
