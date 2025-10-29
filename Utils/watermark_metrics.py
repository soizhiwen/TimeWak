import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from Utils.metric_utils import write_csv
from Utils.watermark_utils import htw_calculate_z_score


def evaluate_watermark_methods(args, reversed_noise):
    reversed_noise = torch.tensor(reversed_noise)
    if args.watermark == "TR":
        tmp_name = args.name.split("-")[0]
        watermarking_mask = torch.tensor(
            np.load(
                f"./OUTPUT/{tmp_name}-sample-tr-{args.window_size}/watermarking_mask.npy"
            )
        )
        gt_patch = torch.tensor(
            np.load(f"./OUTPUT/{tmp_name}-sample-tr-{args.window_size}/gt_patch.npy")
        )
        metric = eval_TR(args, reversed_noise, watermarking_mask, gt_patch)
    elif args.watermark == "GS":
        metric = eval_GS(args, reversed_noise)
    elif args.watermark == "HTW":
        metric = eval_HTW(args, reversed_noise.numpy())
    elif args.watermark == "TabWak":
        metric = eval_TabWak(args, reversed_noise)
    elif args.watermark == "TabWakT":
        metric = eval_TabWakT(args, reversed_noise)
    elif args.watermark == "TimeWak":
        metric = eval_TimeWak(args, reversed_noise)
    elif args.watermark == "SpatDDIM":
        metric = eval_SpatDDIM(args, reversed_noise)
    elif args.watermark == "SpatBDIA":
        metric = eval_SpatBDIA(args, reversed_noise)
    elif args.watermark == "TempDDIM":
        metric = eval_TempDDIM(args, reversed_noise)
    else:
        raise NotImplementedError()
    return metric


def eval_TR(args, reversed_latents, watermarking_mask, gt_patch):
    metrics = []

    for i in range(len(reversed_latents)):
        reversed_latent = reversed_latents[i].unsqueeze(0).unsqueeze(0)

        if "complex" in args.w_measurement:
            reversed_latent_fft = torch.fft.fftshift(
                torch.fft.fft2(reversed_latent), dim=(-1, -2)
            )
            target_patch = gt_patch[i].unsqueeze(0)
        elif "seed" in args.w_measurement:
            reversed_latent_fft = reversed_latent_fft
            target_patch = gt_patch[i].unsqueeze(0)
        else:
            NotImplementedError(f"w_measurement: {args.w_measurement}")

        if "l1" in args.w_measurement:
            metric = (
                torch.abs(
                    reversed_latent_fft[watermarking_mask[i].unsqueeze(0)]
                    - target_patch[watermarking_mask[i].unsqueeze(0)]
                )
                .mean()
                .item()
            )
            metrics.append(metric)
            write_csv([metric], "tr", args.save_dir)
        else:
            NotImplementedError(f"w_measurement: {args.w_measurement}")

    metrics = np.array(metrics)
    return metrics.mean()


def eval_GS(args, reversed_noise):
    total_elements = (
        reversed_noise.shape[0] * reversed_noise.shape[1] * reversed_noise.shape[2]
    )  # Total number of elements in the noise
    cnt = 0
    # for each row, normalize the noise into gaussian distribution with mean 0 and std 1
    reversed_noise = (reversed_noise - reversed_noise.mean()) / reversed_noise.std()
    # get the random bit sequence of the noise along the row
    torch.manual_seed(217)
    latent_seed = torch.randint(
        0, 2, (reversed_noise.shape[1], reversed_noise.shape[2])
    )
    for row in reversed_noise:
        sign_row = (row > 0).int()
        # check if the sign of the noise is the same as the latent seed
        cnt_row = (sign_row == latent_seed).sum().item()
        cnt += cnt_row
        acc_bit_row = cnt_row / reversed_noise.shape[1] / reversed_noise.shape[2]
        write_csv([acc_bit_row], "gs", args.save_dir)
    proportion = cnt / total_elements
    return proportion


def eval_HTW(args, synth_data):
    all_zscores = []

    for sample in tqdm(synth_data, desc="HTW detection"):
        sample_zscore = 0
        for series in sample.T:
            htw_zscore = htw_calculate_z_score(series)
            sample_zscore += htw_zscore

        score = sample_zscore / sample.shape[1]
        all_zscores.append(score)
        write_csv([score], "htw", args.save_dir)

    all_zscores = np.array(all_zscores)
    return all_zscores.mean()


def eval_TabWak(args, reversed_noise):
    cnt = 0
    correct = 0

    # Flatten reversed_noise along the first two dimensions
    shape = reversed_noise.shape
    reversed_noise = reversed_noise.reshape(-1, reversed_noise.shape[2])

    # Calculate quantiles (vectorized)
    q1 = torch.quantile(reversed_noise, 0.25, dim=1, keepdim=True)
    q2 = torch.quantile(reversed_noise, 0.5, dim=1, keepdim=True)
    q3 = torch.quantile(reversed_noise, 0.75, dim=1, keepdim=True)

    # Assign values based on quantiles (vectorized)
    reversed_noise = torch.where(
        reversed_noise <= q1,
        0,
        torch.where(
            reversed_noise >= q3,
            1,
            torch.where((reversed_noise > q1) & (reversed_noise < q2), 2, 3),
        ),
    )

    # Reshape reversed_noise back
    bit_dim = shape[2]
    if bit_dim % 2 != 0:
        bit_dim -= 1
        found = False
        while not found:
            try:
                reversed_noise.reshape(shape[0], -1, bit_dim)
                found = True
            except:
                bit_dim -= 1
    reversed_noise_sign = reversed_noise.view(shape[0], -1, bit_dim)

    torch.manual_seed(217)
    permutation = torch.randperm(bit_dim)
    inverse_permutation = torch.argsort(permutation)
    reversed_noise_sign = reversed_noise_sign[:, :, inverse_permutation]

    half_dim = bit_dim // 2
    for row in reversed_noise_sign:
        cnt_row = 0
        correct_row = 0
        for bits in row:
            first_half = bits[:half_dim]
            last_half = bits[half_dim:]
            for i in range(half_dim):
                if first_half[i] == 0 or first_half[i] == 1:
                    cnt_row += 1
                if first_half[i] == 0 and (last_half[i] == 0 or last_half[i] == 2):
                    correct_row += 1
                if first_half[i] == 1 and (last_half[i] == 1 or last_half[i] == 3):
                    correct_row += 1

        cnt += cnt_row
        correct += correct_row

        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        write_csv([acc_bit_row], "tabwak", args.save_dir)

    avg_bit_accuracy = correct / cnt
    return avg_bit_accuracy


def eval_TabWakT(args, reversed_noise):
    cnt = 0
    correct = 0

    # Flatten reversed_noise along the first two dimensions
    shape = reversed_noise.shape
    reversed_noise = reversed_noise.permute(0, 2, 1)
    reversed_noise = reversed_noise.reshape(-1, reversed_noise.shape[2])

    # Calculate quantiles (vectorized)
    q1 = torch.quantile(reversed_noise, 0.25, dim=1, keepdim=True)
    q2 = torch.quantile(reversed_noise, 0.5, dim=1, keepdim=True)
    q3 = torch.quantile(reversed_noise, 0.75, dim=1, keepdim=True)

    # Assign values based on quantiles (vectorized)
    reversed_noise = torch.where(
        reversed_noise <= q1,
        0,
        torch.where(
            reversed_noise >= q3,
            1,
            torch.where((reversed_noise > q1) & (reversed_noise < q2), 2, 3),
        ),
    )

    # Reshape reversed_noise back
    bit_dim = shape[1]
    reversed_noise_sign = reversed_noise.view(shape[0], -1, bit_dim)

    torch.manual_seed(217)
    permutation = torch.randperm(bit_dim)
    inverse_permutation = torch.argsort(permutation)
    reversed_noise_sign = reversed_noise_sign[:, :, inverse_permutation]

    half_dim = bit_dim // 2
    for row in reversed_noise_sign:
        cnt_row = 0
        correct_row = 0
        for bits in row:
            first_half = bits[:half_dim]
            last_half = bits[half_dim:]
            for i in range(half_dim):
                if first_half[i] == 0 or first_half[i] == 1:
                    cnt_row += 1
                if first_half[i] == 0 and (last_half[i] == 0 or last_half[i] == 2):
                    correct_row += 1
                if first_half[i] == 1 and (last_half[i] == 1 or last_half[i] == 3):
                    correct_row += 1

        cnt += cnt_row
        correct += correct_row

        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        write_csv([acc_bit_row], "tabwak_t", args.save_dir)

    avg_bit_accuracy = correct / cnt
    return avg_bit_accuracy


def eval_TimeWak(args, reversed_noise):
    cnt = 0
    correct = 0

    if args.bits == 2:
        q1 = torch.quantile(reversed_noise, 0.5, dim=1, keepdim=True)

        # Assign values based on quantiles (vectorized)
        reversed_noise = torch.where(reversed_noise <= q1, 0, 1)

    elif args.bits == 3:
        q1 = torch.quantile(reversed_noise, 0.33, dim=1, keepdim=True)
        q2 = torch.quantile(reversed_noise, 0.5, dim=1, keepdim=True)
        q3 = torch.quantile(reversed_noise, 0.67, dim=1, keepdim=True)

        # Assign values based on quantiles (vectorized)
        reversed_noise = torch.where(
            reversed_noise <= q1,
            0,
            torch.where(
                reversed_noise >= q3,
                1,
                torch.where((reversed_noise > q1) & (reversed_noise < q2), 2, 3),
            ),
        )

    elif args.bits == 4:
        q1 = torch.quantile(reversed_noise, 0.25, dim=1, keepdim=True)
        q2 = torch.quantile(reversed_noise, 0.5, dim=1, keepdim=True)
        q3 = torch.quantile(reversed_noise, 0.75, dim=1, keepdim=True)

        # Assign values based on quantiles (vectorized)
        reversed_noise = torch.where(
            reversed_noise <= q1,
            0,
            torch.where(
                reversed_noise >= q3,
                1,
                torch.where((reversed_noise > q1) & (reversed_noise < q2), 2, 3),
            ),
        )

    else:
        raise NotImplementedError()

    # reversed_noise = reversed_noise.bool()
    seed_idx = torch.arange(0, reversed_noise.shape[1], args.interval)

    for i in range(reversed_noise.shape[2]):  # features
        st0 = torch.get_rng_state()
        torch.manual_seed(i)
        permutation = torch.randperm(reversed_noise.shape[1])
        inverse_permutation = torch.argsort(permutation)
        reversed_noise[:, :, i] = reversed_noise[:, inverse_permutation, i]
        torch.set_rng_state(st0)

    for sample in reversed_noise:
        cnt_row = 0
        correct_row = 0
        for i in range(reversed_noise.shape[1] - 1, 0, -1):  # timesteps
            if i in seed_idx:
                continue
            st0 = torch.get_rng_state()
            torch.manual_seed(i)
            permutation = torch.randperm(reversed_noise.shape[2])
            inverse_permutation = torch.argsort(permutation)

            if args.bits == 2:
                correct_row += (
                    (sample[i, inverse_permutation] == sample[i - 1, :])
                    .int()
                    .sum()
                    .item()
                )
                cnt_row += reversed_noise.shape[2]

            elif args.bits == 3 or args.bits == 4:
                curr = sample[i, inverse_permutation]
                prev = sample[i - 1, :]
                for j in range(sample.shape[1]):  # features
                    if prev[j] == 0 or prev[j] == 1:
                        cnt_row += 1
                    if prev[j] == 0 and (curr[j] == 0 or curr[j] == 2):
                        correct_row += 1
                    if prev[j] == 1 and (curr[j] == 1 or curr[j] == 3):
                        correct_row += 1

            else:
                raise NotImplementedError()

            torch.set_rng_state(st0)

        cnt += cnt_row
        correct += correct_row

        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        write_csv([acc_bit_row], "timewak", args.save_dir)

    avg_bit_accuracy = correct / cnt
    return avg_bit_accuracy


def eval_SpatDDIM(args, reversed_noise):
    cnt = 0
    correct = 0
    reversed_noise = reversed_noise.permute(0, 2, 1)
    q1 = torch.quantile(reversed_noise, 0.5, dim=1, keepdim=True)

    # Assign values based on quantiles (vectorized)
    reversed_noise = torch.where(reversed_noise <= q1, 0, 1)
    reversed_noise = reversed_noise.bool()
    seed_idx = torch.arange(0, reversed_noise.shape[1], args.interval)

    for i in range(reversed_noise.shape[2]):  # features
        st0 = torch.get_rng_state()
        torch.manual_seed(i)
        permutation = torch.randperm(reversed_noise.shape[1])
        inverse_permutation = torch.argsort(permutation)
        reversed_noise[:, :, i] = reversed_noise[:, inverse_permutation, i]
        torch.set_rng_state(st0)

    for sample in reversed_noise:
        cnt_row = 0
        correct_row = 0
        for i in range(reversed_noise.shape[1] - 1, 0, -1):  # timesteps
            if i in seed_idx:
                continue
            st0 = torch.get_rng_state()
            torch.manual_seed(i)
            permutation = torch.randperm(reversed_noise.shape[2])
            inverse_permutation = torch.argsort(permutation)
            correct_row += (
                (sample[i, inverse_permutation] == sample[i - 1, :]).int().sum().item()
            )
            cnt_row += reversed_noise.shape[2]
            torch.set_rng_state(st0)

        cnt += cnt_row
        correct += correct_row

        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        write_csv([acc_bit_row], "spatddim", args.save_dir)

    avg_bit_accuracy = correct / cnt
    return avg_bit_accuracy


def eval_SpatBDIA(args, reversed_noise):
    cnt = 0
    correct = 0
    reversed_noise = reversed_noise.permute(0, 2, 1)
    q1 = torch.quantile(reversed_noise, 0.5, dim=1, keepdim=True)

    # Assign values based on quantiles (vectorized)
    reversed_noise = torch.where(reversed_noise <= q1, 0, 1)
    reversed_noise = reversed_noise.bool()
    seed_idx = torch.arange(0, reversed_noise.shape[1], args.interval)

    for i in range(reversed_noise.shape[2]):  # features
        st0 = torch.get_rng_state()
        torch.manual_seed(i)
        permutation = torch.randperm(reversed_noise.shape[1])
        inverse_permutation = torch.argsort(permutation)
        reversed_noise[:, :, i] = reversed_noise[:, inverse_permutation, i]
        torch.set_rng_state(st0)

    for sample in reversed_noise:
        cnt_row = 0
        correct_row = 0
        for i in range(reversed_noise.shape[1] - 1, 0, -1):  # timesteps
            if i in seed_idx:
                continue
            st0 = torch.get_rng_state()
            torch.manual_seed(i)
            permutation = torch.randperm(reversed_noise.shape[2])
            inverse_permutation = torch.argsort(permutation)
            correct_row += (
                (sample[i, inverse_permutation] == sample[i - 1, :]).int().sum().item()
            )
            cnt_row += reversed_noise.shape[2]
            torch.set_rng_state(st0)

        cnt += cnt_row
        correct += correct_row

        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        write_csv([acc_bit_row], "spatbdia", args.save_dir)

    avg_bit_accuracy = correct / cnt
    return avg_bit_accuracy


def eval_TempDDIM(args, reversed_noise):
    cnt = 0
    correct = 0

    q1 = torch.quantile(reversed_noise, 0.5, dim=1, keepdim=True)

    # Assign values based on quantiles (vectorized)
    reversed_noise = torch.where(reversed_noise <= q1, 0, 1)
    reversed_noise = reversed_noise.bool()
    seed_idx = torch.arange(0, reversed_noise.shape[1], args.interval)

    for i in range(reversed_noise.shape[2]):  # features
        st0 = torch.get_rng_state()
        torch.manual_seed(i)
        permutation = torch.randperm(reversed_noise.shape[1])
        inverse_permutation = torch.argsort(permutation)
        reversed_noise[:, :, i] = reversed_noise[:, inverse_permutation, i]
        torch.set_rng_state(st0)

    for sample in reversed_noise:
        cnt_row = 0
        correct_row = 0
        for i in range(reversed_noise.shape[1] - 1, 0, -1):  # timesteps
            if i in seed_idx:
                continue
            st0 = torch.get_rng_state()
            torch.manual_seed(i)
            permutation = torch.randperm(reversed_noise.shape[2])
            inverse_permutation = torch.argsort(permutation)
            correct_row += (
                (sample[i, inverse_permutation] == sample[i - 1, :]).int().sum().item()
            )
            cnt_row += reversed_noise.shape[2]
            torch.set_rng_state(st0)

        cnt += cnt_row
        correct += correct_row

        acc_bit_row = correct_row / cnt_row if cnt_row != 0 else 0.5
        write_csv([acc_bit_row], "tempddim", args.save_dir)

    avg_bit_accuracy = correct / cnt
    return avg_bit_accuracy


def get_zscore(method, wo_data, w_data, num_sample=1000, p_value=1e-3):
    mean1 = wo_data.mean()
    std1 = wo_data.std()
    z_threshold = norm.ppf(1 - p_value)
    z_score_samples = []
    cnt = 0

    for _ in range(100):
        sample = np.random.choice(w_data.reshape(-1), num_sample, replace=False)
        mean_sample = sample.mean()
        if method == "tr":
            z_score_sample = abs((mean1 - mean_sample) / std1)
        else:
            z_score_sample = (mean_sample - mean1) / (std1 / (num_sample**0.5))

        if z_score_sample > z_threshold:
            cnt += 1

        z_score_samples.append(z_score_sample)

    z_score_samples = np.array(z_score_samples)
    mean_z = z_score_samples.mean()
    std_z = z_score_samples.std()
    return mean_z, std_z, cnt / 100
