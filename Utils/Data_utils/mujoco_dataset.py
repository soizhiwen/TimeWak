import os
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from Utils.masking_utils import noise_mask
from Utils.watermark_utils import add_attack


class MuJoCoDataset(Dataset):
    def __init__(
        self,
        dataset=None,
        window=128,
        num=30000,
        dim=12,
        proportion=0.8,
        save2npy=True,
        neg_one_to_one=True,
        shuffle=True,
        seed=123,
        scalar=None,
        period="train",
        output_dir="./OUTPUT",
        attack=None,
        attack_factor=None,
        predict_length=None,
        missing_ratio=None,
        style="separate",
        distribution="geometric",
        mean_mask_length=3,
    ):
        super(MuJoCoDataset, self).__init__()
        assert period in ["train", "test"], "period must be train or test."
        if period == "train":
            assert ~(predict_length is not None or missing_ratio is not None), ""

        self.window, self.var_num = window, dim
        self.auto_norm = neg_one_to_one
        self.shuffle = shuffle
        self.attack, self.attack_factor = attack, attack_factor
        self.dir = os.path.join(output_dir, "samples")
        os.makedirs(self.dir, exist_ok=True)
        self.pred_len, self.missing_ratio = predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = (
            style,
            distribution,
            mean_mask_length,
        )
        self.dataset = dataset

        if self.dataset is not None:
            self.rawdata, self.scaler = self.fit_data(self.dataset, seed)
        else:
            self.rawdata, self.scaler = self._generate_random_trajectories(
                n_samples=num, seed=seed
            )
        if scalar is not None:
            self.scaler = scalar

        self.period, self.save2npy = period, save2npy
        self.data = self.normalize(self.rawdata)
        train, inference = self.__getsamples(self.data, proportion, seed)

        self.samples = train if period == "train" else inference
        self.sample_num = self.samples.shape[0]

        if period == "test":
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()

    def __getsamples(self, data, proportion, seed):
        train_data, test_data = self.divide(data, proportion, self.shuffle, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(
                    os.path.join(
                        self.dir, f"mujoco_ground_truth_{self.window}_test.npy"
                    ),
                    self.unnormalize(test_data),
                )
            np.save(
                os.path.join(self.dir, f"mujoco_ground_truth_{self.window}_train.npy"),
                self.unnormalize(train_data),
            )
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(
                        os.path.join(
                            self.dir, f"mujoco_norm_truth_{self.window}_test.npy"
                        ),
                        unnormalize_to_zero_to_one(test_data),
                    )
                np.save(
                    os.path.join(
                        self.dir, f"mujoco_norm_truth_{self.window}_train.npy"
                    ),
                    unnormalize_to_zero_to_one(train_data),
                )
            else:
                if 1 - proportion > 0:
                    np.save(
                        os.path.join(
                            self.dir, f"mujoco_norm_truth_{self.window}_test.npy"
                        ),
                        test_data,
                    )
                np.save(
                    os.path.join(
                        self.dir, f"mujoco_norm_truth_{self.window}_train.npy"
                    ),
                    train_data,
                )

        return train_data, test_data

    def _generate_random_trajectories(self, n_samples, seed=123):
        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception(
                "Deepmind Control Suite is required to generate the dataset."
            ) from e

        env = suite.load("hopper", "stand")
        physics = env.physics

        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        data = np.zeros((n_samples, self.window, self.var_num))
        for i in range(n_samples):
            with physics.reset_context():
                # x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
                physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
                physics.data.qpos[2:] = np.random.uniform(
                    -2, 2, size=physics.data.qpos[2:].shape
                )
                physics.data.qvel[:] = np.random.uniform(
                    -5, 5, size=physics.data.qvel.shape
                )

            for t in range(self.window):
                data[i, t, : self.var_num // 2] = physics.data.qpos
                data[i, t, self.var_num // 2 :] = physics.data.qvel
                physics.step()

        # Restore RNG.
        np.random.set_state(st0)

        scaler = MinMaxScaler()
        scaler = scaler.fit(data.reshape(-1, self.var_num))
        return data, scaler

    def fit_data(self, filepath, seed):
        """Transform data to MinMaxScaler"""
        data = np.load(filepath)
        if self.attack:
            st0 = np.random.get_state()
            np.random.seed(seed)
            data = add_attack(data, self.attack, self.attack_factor)
            np.random.set_state(st0)
        scaler = MinMaxScaler()
        scaler = scaler.fit(data.reshape(-1, self.var_num))
        return data, scaler

    @staticmethod
    def divide(data, ratio, shuffle, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size) if shuffle else np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(
                x,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution,
            )  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"mujoco_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == "test":
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
