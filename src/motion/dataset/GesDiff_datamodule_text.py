import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import joblib as jl
from typing import Optional, List


# ----------------------------
# Datasets: Motion + Text
# ----------------------------
class MotionTextDataset(Dataset):
    def __init__(
        self,
        cond_data: np.ndarray,
        motion_data: np.ndarray,
        dropout: float = 0.0,
    ):
        assert cond_data.ndim == 4 and motion_data.ndim == 4, "expect [V, N, T, D]"
        assert cond_data.shape[:2] == motion_data.shape[:2], "text/motion must share [V, N]"
        self.dropout = float(dropout)

        Vc, Nc, Tc_text, Dc = cond_data.shape
        Vm, Nm, Tm_mot,  Dm = motion_data.shape

        self.cond   = cond_data.reshape(Vc * Nc, Tc_text, Dc).astype(np.float32)   # [M, 77,  D_text]
        self.motion = motion_data.reshape(Vm * Nm, Tm_mot,  Dm).astype(np.float32) # [M, 150, D_motion]

        self._len = self.cond.shape[0]

        assert self._len == self.motion.shape[0], "flattened batch size mismatch between text and motion"

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        cond = self.cond[idx]    # [T_text, D_text]
        mot  = self.motion[idx]  # [T_mot,  D_motion]

        if self.dropout > 0.0 and np.random.rand() < self.dropout:
            cond = np.zeros_like(cond, dtype=np.float32)

        return {
            "cond_text": torch.from_numpy(cond),     # [77,  D_text]
            "motion_data": torch.from_numpy(mot),
        }

class TestTextDataset(Dataset):
    def __init__(self, cond_data: np.ndarray):
        assert cond_data.ndim == 4, "expect [V, N, T, D]"
        V, N, T, D = cond_data.shape
        self.cond = cond_data.reshape(V * N, T, D).astype(np.float32)
        self._len = self.cond.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        cond = self.cond[idx]
        return {"cond_text": torch.from_numpy(cond)}  # [T, D_text]


# ----------------------------
# DataModule: BVH + TEXT
# ----------------------------
class GesDiffDataModule_BVHText(nn.Module):

    def __init__(self,
                 data_root: str,
                 framerate: int = 30,
                 seqlen: int = 10,
                 dropout: float = 0.3,
                 test_style: List[str] = [],
                 batch_size: int = 32,
                 pin_memory: bool = True,
                 num_workers: int = 22,
                 drop_last: bool = False,
                 cond_feature_dim: int = 1024,
                 motion_feature_dim: int = 166,
                 local_rank: int = 0,
                 world_size: int = 4):
        super().__init__()

        self.train_dataset = None
        self.validation_dataset = None
        self.test_datasets = None

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.data_pipe = None
        self.input_scaler = None
        self.output_scaler = None

        self.data_root = data_root
        self.framerate = framerate
        self.seqlen = seqlen
        self.dropout = dropout
        self.test_style = test_style
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.local_rank = local_rank
        self.world_size = world_size
        self.motion_feature_dim = motion_feature_dim
        self.cond_feature_dim = cond_feature_dim

        self.is_inited_with_std = True

        print(self.local_rank, self.world_size)


    def setup(self, stage: Optional[str] = None) -> None:

        if self.is_inited_with_std:
            in_scaler_path = os.path.join(self.data_root, 'input_scaler.sav')
            out_scaler_path = os.path.join(self.data_root, 'output_scaler.sav')
            if os.path.exists(in_scaler_path):
                self.input_scaler = jl.load(in_scaler_path)
            if os.path.exists(out_scaler_path):
                self.output_scaler = jl.load(out_scaler_path)

        # self.data_pipe = jl.load(os.path.join(self.data_root, 'data_pipe_' + str(self.framerate) + 'fps.sav'))

        if stage == 'train':

            self.train_input = np.load(
                os.path.join(self.data_root, f'train_input_{self.framerate}fps.npz')
            )['clips'].astype(np.float32)    # [3, N, T, D_text]

            self.train_output = np.load(
                os.path.join(self.data_root, f'train_output_{self.framerate}fps.npz')
            )['clips'].astype(np.float32)    # [3, N, T, D_motion]

            self.val_input = np.load(
                os.path.join(self.data_root, f'val_input_{self.framerate}fps.npz')
            )['clips'].astype(np.float32)

            self.val_output = np.load(
                os.path.join(self.data_root, f'val_output_{self.framerate}fps.npz')
            )['clips'].astype(np.float32)

            self.train_dataset = MotionTextDataset(
                cond_data=self.train_input,
                motion_data=self.train_output,
                dropout=self.dropout,
            )
            self.validation_dataset = MotionTextDataset(
                cond_data=self.val_input,
                motion_data=self.val_output,
                dropout=0.0,
            )

        if stage == 'train' or stage == 'inference':

            self.test_inputs = [
                np.load(os.path.join(self.data_root, f'test_input_{one_test_style}_{self.framerate}fps.npz'))['clips'].astype(np.float32)
                for one_test_style in self.test_style
            ]
            self.test_datasets = [TestTextDataset(test_input) for test_input in self.test_inputs]


    def get_train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True)
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=train_sampler,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
        return train_sampler, train_loader

    def get_val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def get_test_dataloader(self):
        test_loaders = [
            DataLoader(
                dataset=dataset,
                batch_size=dataset.__len__(),
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                drop_last=False
            )
            for dataset in self.test_datasets
        ]
        return test_loaders


    def prepare_dataloaders(self, stage: Optional[str] = None):
        self.setup(stage)
        if stage == 'train':
            self.train_sampler, self.train_dataloader = self.get_train_dataloader()
            self.val_dataloader = self.get_val_dataloader()
            self.test_dataloader = self.get_test_dataloader()
            return self.train_sampler, self.train_dataloader, self.val_dataloader, self.test_dataloader
        elif stage == 'inference':
            self.test_dataloader = self.get_test_dataloader()
            return self.test_dataloader
        else:
            raise ValueError("Invalid mode")

    def get_scalers(self):
        return self.input_scaler, self.output_scaler

    def get_data_pipe(self):
        return self.data_pipe

    def channels_dim(self):
        return self.motion_feature_dim, self.cond_feature_dim
