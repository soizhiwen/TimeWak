import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.watermark_metrics import evaluate_watermark_methods
from Utils.watermark_utils import htw_watermark_mts, add_attack
from Utils.io_utils import (
    load_yaml_config,
    seed_everything,
    merge_opts_to_config,
    instantiate_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument("--name", type=str, default=None)

    parser.add_argument(
        "--config_file", type=str, default=None, help="path of config file"
    )
    parser.add_argument(
        "--output", type=str, default="OUTPUT", help="directory to save the results"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="use tensorboard for logging"
    )

    # args for random

    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        default=False,
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="seed for initializing training."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id to use. If given, only the specific gpu will be"
        " used, and ddp will be disabled",
    )

    # args for training
    parser.add_argument(
        "--train", action="store_true", default=False, help="Train or Test."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        choices=[0, 1],
        help="Condition or Uncondition.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=24,
        choices=[24, 64, 128, 256],
        help="Time series window size.",
    )
    parser.add_argument(
        "--detect", type=str, default="", help="Path to detect watermarking."
    )
    parser.add_argument(
        "--post_wm", type=str, default="", help="Path to post-watermarking."
    )
    parser.add_argument(
        "--mode", type=str, default="infill", help="Infilling or Forecasting."
    )
    parser.add_argument(
        "--watermark",
        type=str,
        default="",
        choices=[
            "TR",
            "GS",
            "HTW",
            "TabWak",
            "TabWakT",
            "TimeWak",
            "SpatDDIM",
            "SpatBDIA",
            "TempDDIM",
        ],
        help="Watermarking during sampling.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="",
        choices=["offset", "crop", "insert"],
        help="Attack on data domain.",
    )
    parser.add_argument(
        "--attack_factor",
        type=float,
        default=1.0,
        help="Factor to manipulate the data.",
    )
    parser.add_argument("--milestone", type=int, default=10)
    parser.add_argument(
        "--missing_ratio", type=float, default=0.0, help="Ratio of Missing Values."
    )
    parser.add_argument(
        "--interval", type=int, default=2, help="Interval of the watermarking."
    )
    parser.add_argument(
        "--bits", type=int, default=2, help="Bits of the watermarking."
    )
    parser.add_argument(
        "--pred_len", type=int, default=0, help="Length of Predictions."
    )

    # args for TR watermarking
    # parser.add_argument("--w_radius", default=100, type=int)
    parser.add_argument("--w_channel", default=-1, type=int)
    parser.add_argument("--w_mask_shape", default="circle")
    parser.add_argument("--w_measurement", default="l1_complex")
    parser.add_argument("--w_injection", default="complex")
    parser.add_argument("--w_pattern", default="ring")
    parser.add_argument("--w_pattern_const", default=0, type=float)

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.w_radius = args.window_size - 1
    args.save_dir = os.path.join(args.output, f"{args.name}")

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    config["model"]["params"]["seq_length"] = args.window_size
    config["dataloader"]["train_dataset"]["params"]["window"] = args.window_size
    config["dataloader"]["test_dataset"]["params"]["window"] = args.window_size

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config["model"]).cuda()
    if args.sample == 1 and args.mode in ["infill", "predict"]:
        test_dataloader_info = build_dataloader_cond(config, args)
    if args.detect:
        config["dataloader"]["train_dataset"]["params"]["dataset"] = args.detect
        config["dataloader"]["train_dataset"]["params"]["proportion"] = 1.0
        config["dataloader"]["train_dataset"]["params"]["shuffle"] = False
        config["dataloader"]["train_dataset"]["params"]["attack"] = args.attack
        config["dataloader"]["train_dataset"]["params"][
            "attack_factor"
        ] = args.attack_factor
        config["dataloader"]["batch_size"] = 2000
        config["dataloader"]["shuffle"] = False
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(
        config=config, args=args, model=model, dataloader=dataloader_info, logger=logger
    )

    if args.train:
        trainer.train()

    elif args.detect:
        if args.watermark == "HTW":
            noises = np.load(args.detect)
            if args.attack:
                st0 = np.random.get_state()
                np.random.seed(123)
                noises = add_attack(noises, args.attack, args.attack_factor)
                np.random.set_state(st0)
        else:
            trainer.load(args.milestone)
            dataloader, dataset = (
                dataloader_info["dataloader"],
                dataloader_info["dataset"],
            )
            noises = trainer.detect(dataloader, [dataset.window, dataset.var_num])
            np.save(os.path.join(args.save_dir, f"ddpm_noise_{args.name}.npy"), noises)
        metric = evaluate_watermark_methods(args, noises)
        print(f"Metric: {metric}")

    elif args.sample == 1 and args.mode in ["infill", "predict"]:
        trainer.load(args.milestone)
        dataloader, dataset = (
            test_dataloader_info["dataloader"],
            test_dataloader_info["dataset"],
        )
        coef = config["dataloader"]["test_dataset"]["coefficient"]
        stepsize = config["dataloader"]["test_dataset"]["step_size"]
        sampling_steps = config["dataloader"]["test_dataset"]["sampling_steps"]
        samples, *_ = trainer.restore(
            dataloader,
            [dataset.window, dataset.var_num],
            coef,
            stepsize,
            sampling_steps,
        )
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
        np.save(
            os.path.join(args.save_dir, f"ddpm_norm_{args.mode}_{args.name}.npy"),
            samples,
        )
        samples = dataset.scaler.inverse_transform(
            samples.reshape(-1, samples.shape[-1])
        ).reshape(samples.shape)
        np.save(
            os.path.join(args.save_dir, f"ddpm_{args.mode}_{args.name}.npy"), samples
        )

    else:
        if args.watermark == "HTW":
            dataset = dataloader_info["dataset"]
            synth_data = np.load(args.post_wm)
            samples = htw_watermark_mts(synth_data)

            np.save(os.path.join(args.save_dir, f"ddpm_fake_{args.name}.npy"), samples)

            samples = dataset.scaler.transform(
                samples.reshape(-1, samples.shape[-1])
            ).reshape(samples.shape)

            np.save(
                os.path.join(args.save_dir, f"ddpm_fake_norm_{args.name}.npy"), samples
            )

        else:
            trainer.load(args.milestone)
            dataset = dataloader_info["dataset"]
            samples = trainer.sample(
                num=10000, size_every=2000, shape=[dataset.window, dataset.var_num]
            )
            if dataset.auto_norm:
                samples = unnormalize_to_zero_to_one(samples)
            np.save(
                os.path.join(args.save_dir, f"ddpm_fake_norm_{args.name}.npy"), samples
            )

            samples = dataset.scaler.inverse_transform(
                samples.reshape(-1, samples.shape[-1])
            ).reshape(samples.shape)

            np.save(os.path.join(args.save_dir, f"ddpm_fake_{args.name}.npy"), samples)


if __name__ == "__main__":
    main()
