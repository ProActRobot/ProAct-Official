
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import joblib as jl
from typing import Optional


class MotionDataset(Dataset):
    def __init__(self, cond_data, motion_data, state_data, dropout, is_waveform=False):
        self.motion_data = motion_data
        self.control_data = cond_data
        self.state_data = state_data
        self.is_waveform = is_waveform

    def __len__(self):
        return self.motion_data.shape[1]

    def __getitem__(self, idx):
        motion_data_concat = np.concatenate([self.motion_data[0, idx, :, :], self.motion_data[1, idx, :, :]], axis=-1)
        if not self.is_waveform:
            audio_data_concat = np.concatenate([self.control_data[0, idx, :, :], self.control_data[1, idx, :, :]], axis=-1)
            sample = {'motion_data': motion_data_concat, 'cond': audio_data_concat, 'state': self.state_data[0, idx, :]}
        else:
            audio_data_concat = np.concatenate([self.control_data[0, idx, :], self.control_data[1, idx, :]], axis=-1)
            sample = {'motion_data': motion_data_concat, 'cond': audio_data_concat, 'state': self.state_data[0, idx, :]}
        return sample


class TestDataset(Dataset):
    def __init__(self, cond_data, is_waveform=False):
        self.control_data = cond_data
        self.is_waveform = is_waveform

    def __len__(self):
        return self.control_data.shape[1]

    def __getitem__(self, idx):
        if not self.is_waveform:
            audio_data_concat = np.concatenate([self.control_data[0, idx, :, :], self.control_data[1, idx, :, :]], axis=-1)
            sample = {'cond': audio_data_concat}
        else:
            audio_data_concat = np.concatenate([self.control_data[0, idx, :], self.control_data[1, idx, :]], axis=-1)
            sample = {'cond': audio_data_concat}
        return sample


class TestDataset_double(Dataset):
    def __init__(self, cond_data, state, is_waveform=False):
        self.control_data = cond_data
        self.state = state
        self.is_waveform = is_waveform

    def __len__(self):
        return self.control_data.shape[1]

    def __getitem__(self, idx):
        if not self.is_waveform:
            audio_data_concat = np.concatenate([self.control_data[0, idx, :, :], self.control_data[1, idx, :, :]], axis=-1)
            sample = {'cond': audio_data_concat,
                      'state': self.state[idx, :]}
        else:
            audio_data_concat = np.concatenate([self.control_data[0, idx, :], self.control_data[1, idx, :]], axis=-1)
            sample = {'cond': audio_data_concat,
                      'state': self.state[idx, :]}
        return sample


class TestDataset_double_agent(Dataset):
    def __init__(self, cond_data, states, is_waveform=False):
        self.control_data = cond_data
        self.state_1 = states[0]
        self.state_2 = states[1]
        self.is_waveform = is_waveform

    def __len__(self):
        return self.control_data.shape[1]

    def __getitem__(self, idx):
        if not self.is_waveform:
            audio_data_concat_speaker_1 = np.concatenate([self.control_data[0, idx, :, :], self.control_data[1, idx, :, :]], axis=-1)
            audio_data_concat_speaker_2 = np.concatenate([self.control_data[1, idx, :, :], self.control_data[0, idx, :, :]], axis=-1)
            sample = {'cond_1': audio_data_concat_speaker_1,
                      'cond_2': audio_data_concat_speaker_2,
                      'state_1': self.state_1[idx, :],
                      'state_2': self.state_2[idx, :]}
        else:
            audio_data_concat_speaker_1 = np.concatenate([self.control_data[0, idx, :], self.control_data[1, idx, :]], axis=-1)
            audio_data_concat_speaker_2 = np.concatenate([self.control_data[1, idx, :], self.control_data[0, idx, :]], axis=-1)
            sample = {'cond_1': audio_data_concat_speaker_1,
                      'cond_2': audio_data_concat_speaker_2,
                      'state_1': self.state_1[idx, :],
                      'state_2': self.state_2[idx, :]}
        return sample


class GesDiffDataModule_Interaction_with_state_mfcc(nn.Module):

    def __init__(self,
                 data_root: str,
                 framerate: int = 30,
                 seqlen: int = 10,
                 dropout: float = 0.3,
                 is_waveform: bool = True,
                 test_style: list = [],
                 batch_size: int = 32,
                 pin_memory: bool = True,
                 num_workers: int = 22,
                 drop_last: bool = False,
                 cond_feature_dim: int = 1024,
                 motion_feature_dim: int = 166,
                 local_rank: int = 0,
                 world_size: int = 4):
        super(GesDiffDataModule_Interaction_with_state_mfcc, self).__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
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
        self.is_waveform = is_waveform
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
            self.input_scaler = jl.load(os.path.join(self.data_root, 'input_scaler.sav'))
            self.output_scaler = jl.load(os.path.join(self.data_root, 'output_scaler.sav'))

        # self.data_pipe = jl.load(
        #     os.path.join(self.data_root, 'data_pipe_' + str(self.framerate) + 'fps.sav'))
        if stage == 'train':
            self.train_input = \
                np.load(os.path.join(self.data_root, 'train_input_' + str(self.framerate) + 'fps.npz'))['clips'].astype(np.float32)
            self.train_output = \
                np.load(os.path.join(self.data_root, 'train_output_' + str(self.framerate) + 'fps.npz'))['clips'].astype(np.float32)
            self.train_state = \
                np.load(os.path.join(self.data_root, 'train_state_' + str(self.framerate) + 'fps.npz'))['clips'].astype(np.float32)

            self.val_input = \
                np.load(os.path.join(self.data_root, 'val_input_' + str(self.framerate) + 'fps.npz'))['clips'].astype(np.float32)
            self.val_output = \
                np.load(os.path.join(self.data_root, 'val_output_' + str(self.framerate) + 'fps.npz'))['clips'].astype(np.float32)
            self.val_state = \
                np.load(os.path.join(self.data_root, 'val_state_' + str(self.framerate) + 'fps.npz'))['clips'].astype(np.float32)

            self.train_dataset = MotionDataset(cond_data=self.train_input, motion_data=self.train_output, state_data=self.train_state,
                                                dropout=self.dropout, is_waveform=self.is_waveform)
            self.validation_dataset = MotionDataset(cond_data=self.val_input, motion_data=self.val_output, state_data=self.val_state,
                                                    dropout=self.dropout,is_waveform=self.is_waveform)
        if stage == 'train' or stage == 'inference':
            self.test_inputs = [\
                np.load(os.path.join(self.data_root, 'test_input_' + one_test_style + '_' + str(self.framerate) + 'fps.npz'))['clips'].astype(np.float32) for one_test_style in self.test_style]
            self.test_datasets = []
            for test_input in self.test_inputs:
                test_dataset = TestDataset(test_input, is_waveform=self.is_waveform)
                self.test_datasets.append(test_dataset)


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
        test_loaders = [DataLoader(
                dataset=dataset,
                batch_size=dataset.__len__(),
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                drop_last=False
            )  for dataset in self.test_datasets]
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
