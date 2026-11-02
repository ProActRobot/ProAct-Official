import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import joblib as jl
from typing import Optional, List

class UnifiedTripletDataset(Dataset):
    def __init__(self, motion, audio, text, state, mask, dropout: float = 0.0):
        self.motion = motion # [N, T, Dm]
        self.audio = audio   # [N, T, Da]
        self.text = text     # [N, 77, Dt]
        self.state = state   # [N, T]
        self.mask = mask     # [N, 3] (Motion, Audio, Text)
        self.dropout = dropout

        print(f"audio shape: {self.audio.shape}, text shape: {self.text.shape}")

    def __len__(self):
        return self.motion.shape[0]

    def __getitem__(self, idx):

        m = self.motion[idx]
        a = self.audio[idx]
        t = self.text[idx]
        s = self.state[idx]
        msk = self.mask[idx].copy() # [1, 1, 0] or [1, 0, 1]



        if self.dropout > 0 and np.random.rand() < self.dropout:


            pass

        return {
            "motion": torch.from_numpy(m),
            "audio": torch.from_numpy(a),
            "text": torch.from_numpy(t),   # [77, D]
            "state": torch.from_numpy(s),
            "mask": torch.from_numpy(msk)
        }

class GesDiffDataModule_Unified(nn.Module):
    def __init__(self,
                 data_root: str,
                 framerate: int = 30,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 local_rank: int = 0,
                 world_size: int = 1,
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.framerate = framerate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.local_rank = local_rank
        self.world_size = world_size


        self.train_dataset = None
        self.val_dataset = None
        self.input_scaler_audio = None
        self.output_scaler = None

    def setup(self, stage: Optional[str] = None):
        # Load Scalers
        try:
            self.input_scaler_audio = jl.load(os.path.join(self.data_root, 'input_scaler_audio.sav'))
            self.output_scaler = jl.load(os.path.join(self.data_root, 'output_scaler.sav'))
        except FileNotFoundError:
            print("[WARN] Scalers not found in data_root.")

        if stage == 'train' or stage is None:
            # Load Train
            data_prefix = os.path.join(self.data_root, f'train_')
            self.train_dataset = UnifiedTripletDataset(
                motion = np.load(data_prefix + f'output_{self.framerate}fps.npz')['clips'].astype(np.float32),
                audio  = np.load(data_prefix + f'input_audio_{self.framerate}fps.npz')['clips'].astype(np.float32),
                text   = np.load(data_prefix + f'input_text_{self.framerate}fps.npz')['clips'].astype(np.float32),
                state  = np.load(data_prefix + f'state_{self.framerate}fps.npz')['clips'].astype(np.int32),
                mask   = np.load(data_prefix + f'mask_{self.framerate}fps.npz')['clips'].astype(np.float32)
            )

            # Load Val
            data_prefix = os.path.join(self.data_root, f'val_')
            self.val_dataset = UnifiedTripletDataset(
                motion = np.load(data_prefix + f'output_{self.framerate}fps.npz')['clips'].astype(np.float32),
                audio  = np.load(data_prefix + f'input_audio_{self.framerate}fps.npz')['clips'].astype(np.float32),
                text   = np.load(data_prefix + f'input_text_{self.framerate}fps.npz')['clips'].astype(np.float32),
                state  = np.load(data_prefix + f'state_{self.framerate}fps.npz')['clips'].astype(np.int32),
                mask   = np.load(data_prefix + f'mask_{self.framerate}fps.npz')['clips'].astype(np.float32)
            )
            print(f"Loaded Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def get_train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True)
        shuffle = False

        train_sampler = sampler
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle,
                                      sampler=sampler, num_workers=self.num_workers, pin_memory=True, drop_last=True)
        return train_sampler, train_dataloader

    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def get_scalers(self):
        return self.input_scaler_audio, self.output_scaler

    def prepare_dataloaders(self, stage: Optional[str] = None):
        # output self.train_sampler, self.train_data, self.valid_data, self.test_data
        self.setup(stage)
        self.train_sampler, self.train_dataloader = self.get_train_dataloader()
        self.valid_dataloader = self.get_val_dataloader()
        return self.train_sampler, self.train_dataloader, self.valid_dataloader
