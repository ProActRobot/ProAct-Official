"""Based on: https://github.com/simonalexanderson/StyleGestures"""
import numpy as np
import os
from shutil import copyfile
import joblib as jl
# import hydra
import json
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from data_process.extract_features import extract_joint_angles
from data_process.extract_features import extract_melspec, extract_waveform
from data_process.extract_features import process_state_labels
from data_process.pymo.parsers import BVHParser
from data_process.pymo.data import Joint, MocapData
from data_process.pymo.preprocessing import *
from data_process.pymo.writers import *
from data_process.utils.utils import fit_and_standardize, standardize

# def slice_data(data, window_size, overlap):
#     nframes = data.shape[0]
#     overlap_frames = (int)(overlap * window_size)

#     n_sequences = (nframes - overlap_frames) // (window_size - overlap_frames)
#     sliced = np.zeros((n_sequences, window_size, data.shape[1])).astype(np.float32)

#     if n_sequences > 0:

#         # extract sequences from the data
#         for i in range(0, n_sequences):
#             frameIdx = (window_size - overlap_frames) * i
#             sliced[i, :, :] = data[frameIdx:frameIdx + window_size, :].copy()
#     else:
#         print("WARNING: data too small for window")

#     return sliced


def slice_data(motion_data_1, motion_data_2, control_data_1, control_data_2, state_data_1, state_data_2, fps, audio_sample_rate, window_size, overlap):
    nframes = motion_data_1.shape[0]
    overlap_frames = (int)(overlap * window_size)

    n_sequences = (nframes - overlap_frames) // (window_size - overlap_frames)
    # n_sequences_double = n_sequences * 2
    sliced_motion_data = np.zeros((2, n_sequences, window_size, motion_data_1.shape[1])).astype(np.float32)
    control_data_dim = control_data_1.shape[1]
    sliced_control_data = np.zeros((2, n_sequences, window_size, control_data_dim)).astype(np.float32)
    sliced_state_data = np.zeros((2, n_sequences, window_size)).astype(np.float32)

    if n_sequences > 0:
        # extract sequences from the data
        for i in range(0, n_sequences):
            frameIdx = (window_size - overlap_frames) * i
            sliced_motion_data[0, i, :, :] = motion_data_1[frameIdx:frameIdx + window_size, :].copy()
            sliced_motion_data[1, i, :, :] = motion_data_2[frameIdx:frameIdx + window_size, :].copy()
            sliced_control_data[0, i, :] = control_data_1[frameIdx:(frameIdx + window_size)].copy()
            sliced_control_data[1, i, :] = control_data_2[frameIdx:(frameIdx + window_size)].copy()
            sliced_state_data[0, i, :] = state_data_1[frameIdx:frameIdx + window_size].copy()
            sliced_state_data[1, i, :] = state_data_2[frameIdx:frameIdx + window_size].copy()

    else:
        print("WARNING: data too small for window")

    return sliced_motion_data, sliced_control_data, sliced_state_data

# def align(data1, data2):
#     """Truncates to the shortest length and concatenates"""

#     nframes1 = data1.shape[0]
#     nframes2 = data2.shape[0]
#     if nframes1 < nframes2:
#         return np.concatenate((data1, data2[:nframes1, :]), axis=1)
#     else:
#         return np.concatenate((data1[:nframes2, :], data2), axis=1)


def align(motion1, motion2, audio1, audio2, state1, state2, fps: int = 20, audio_sample_rate: int = 16000):
    """Truncates to the shortest length and concatenates"""

    motion1_time = motion1.shape[0]
    motion2_time = motion2.shape[0]
    audio1_time = audio1.shape[0]
    audio2_time = audio2.shape[0]
    state1_time = state1.shape[0]
    state2_time = state2.shape[0]
    min_time = min(motion1_time, motion2_time, audio1_time, audio2_time, state1_time, state2_time)

    return motion1[:min_time, :], motion2[:min_time, :], audio1[:min_time], audio2[:min_time], state1[:min_time], state2[:min_time]


# def import_data(file, motion_path, speech_path, mirror=False, start=0, end=None):
#     """Loads a file and concatenate all features to one [time, features] matrix.
#      NOTE: All sources will be truncated to the shortest length, i.e. we assume they
#      are time synchronized and has the same start time."""

#     suffix = ""
#     if mirror:
#         suffix = "_mirrored"

#     motion_data = np.load(os.path.join(motion_path, file + suffix + '.npz'))['clips'].astype(np.float32)
#     n_motion_feats = motion_data.shape[1]

#     speech_data = np.load(os.path.join(speech_path, file + '.npy')).astype(np.float32)

#     control_data = speech_data
#     concat_data = align(motion_data, control_data)

#     if not end:
#         end = concat_data.shape[0]

#     return concat_data[start:end, :], n_motion_feats

def import_data(file_dict, mirror=False, fps=20, audio_sample_rate=16000, start=0,
                              end=None):
    suffix = ".npz"
    if mirror:
        suffix = "_mirrored.npz"

    motion_paths = file_dict['motion_path']
    audio_paths = file_dict['audio_path']
    state_paths = file_dict['state_label_path']

    motion_data_a = np.load(motion_paths[0]+suffix)['clips'].astype(np.float32)  # [Frame, Feature]
    motion_data_b = np.load(motion_paths[1]+suffix)['clips'].astype(np.float32)  # [Frame, Feature]
    n_motion_feats = motion_data_a.shape[1]

    control_data_a = np.load(audio_paths[0]+'.npy').astype(np.float32)  # [T,]
    control_data_b = np.load(audio_paths[1]+'.npy').astype(np.float32)  # [T,]

    state_data_a = np.load(state_paths[0]+'.npy').astype(int)
    state_data_b = np.load(state_paths[1]+'.npy').astype(int)

    clip_motion_data_1, clip_motion_data_2, clip_control_data_1, clip_control_data_2, clip_state_data_1, clip_state_data_2 = align(motion_data_a, motion_data_b, control_data_a, control_data_b, state_data_a, state_data_b, fps, audio_sample_rate)

    print(clip_motion_data_1.shape, clip_control_data_1.shape, clip_state_data_1.shape)
    if not end:
        end = clip_motion_data_1.shape[0]

    return clip_motion_data_1[start:end, :], clip_motion_data_2[start:end, :], \
        clip_control_data_1[start: end, :], clip_control_data_2[start: end, :], \
        clip_state_data_1[start:end], clip_state_data_2[start:end]



# def import_and_slice(files, motion_path, speech_path, slice_window, slice_overlap, mirror=False, start=0,
#                      end=None):
#     """Imports all features and slices them to samples with equal lenth time [samples, timesteps, features]."""

#     fi = 0
#     for file in files:
#         print(f'import_and_slice: {file}')

#         # slice dataset
#         concat_data, n_motion_feats = import_data(file, motion_path, speech_path, False, start, end)
#         sliced = slice_data(concat_data, slice_window, slice_overlap)

#         if mirror:
#             concat_mirr, nmf = import_data(file, motion_path, speech_path, True, start, end)
#             sliced_mirr = slice_data(concat_mirr, slice_window, slice_overlap)

#             # append to the sliced dataset
#             sliced = np.concatenate((sliced, sliced_mirr), axis=0)

#         if fi == 0:
#             out_data = sliced
#         else:
#             out_data = np.concatenate((out_data, sliced), axis=0)
#         fi = fi + 1

#     return out_data[:, :, :n_motion_feats], out_data[:, :, n_motion_feats:]


def import_and_slice(files, slice_window, slice_overlap, mirror=False, fps=20,
                                   audio_sample_rate=16000, start=0, end=None):
    out_sliced_motion_data = None
    out_sliced_control_data = None
    out_sliced_state_data = None
    fi = 0
    for pair_name, pair_value in files.items():
        print(pair_name, pair_value)
        motion_data_1, motion_data_2, control_data_1, control_data_2, state_data_1, state_data_2 = import_data(file_dict=pair_value, mirror=False,
                                                              fps=fps, audio_sample_rate=audio_sample_rate, start=start,
                                                              end=end)
        # slice dataset
        sliced_motion_data, sliced_control_data, sliced_state_data = slice_data(motion_data_1, motion_data_2, control_data_1, control_data_2, state_data_1, state_data_2,
                                                                           fps, audio_sample_rate, slice_window, slice_overlap)
        if mirror:
            motion_data_1_mir, motion_data_2_mir, control_data_1_mir, control_data_2_mir, state_data_1_mir, state_data_2_mir = import_data(file_dict=pair_value,
                                                                                            mirror=True, fps=fps,
                                                                                            audio_sample_rate=audio_sample_rate,
                                                                                            start=start, end=end)
            sliced_motion_data_mirr, sliced_control_data_mirr, sliced_state_data_mirr = slice_data(motion_data_1_mir, motion_data_2_mir, control_data_1_mir, control_data_2_mir, state_data_1_mir, state_data_2_mir,
                                                                                         fps, audio_sample_rate, slice_window, slice_overlap)

            # append to the sliced dataset
            sliced_motion_data = np.concatenate((sliced_motion_data, sliced_motion_data_mirr), axis=1)
            sliced_control_data = np.concatenate((sliced_control_data, sliced_control_data_mirr), axis=1)
            sliced_state_data = np.concatenate((sliced_state_data, sliced_state_data_mirr), axis=1)

        if fi == 0:
            out_sliced_motion_data = sliced_motion_data
            out_sliced_control_data = sliced_control_data
            out_sliced_state_data = sliced_state_data
        else:
            out_sliced_motion_data = np.concatenate((out_sliced_motion_data, sliced_motion_data), axis=1)
            out_sliced_control_data = np.concatenate((out_sliced_control_data, sliced_control_data), axis=1)
            out_sliced_state_data = np.concatenate((out_sliced_state_data, sliced_state_data), axis=1)
        # if fi == 0:
        #     out_data = sliced
        # else:
        #     out_data = np.concatenate((out_data, sliced), axis=0)
        fi = fi + 1

    return out_sliced_motion_data, out_sliced_control_data, out_sliced_state_data



def slice_data_with_waveform(motion_data_1, motion_data_2, control_data_1, control_data_2, state_data_1, state_data_2, fps, audio_sample_rate, window_size, overlap):
    nframes = motion_data_1.shape[0]
    overlap_frames = (int)(overlap * window_size)

    n_sequences = (nframes - overlap_frames) // (window_size - overlap_frames)
    # n_sequences_double = n_sequences * 2
    sliced_motion_data = np.zeros((2, n_sequences, window_size, motion_data_1.shape[1])).astype(np.float32)
    sliced_control_data = np.zeros((2, n_sequences, window_size // fps * audio_sample_rate)).astype(np.float32)
    sliced_state_data = np.zeros((2, n_sequences, window_size)).astype(np.float32)

    if n_sequences > 0:
        # extract sequences from the data
        for i in range(0, n_sequences):
            frameIdx = (window_size - overlap_frames) * i
            sliced_motion_data[0, i, :, :] = motion_data_1[frameIdx:frameIdx + window_size, :].copy()
            sliced_motion_data[1, i, :, :] = motion_data_2[frameIdx:frameIdx + window_size, :].copy()
            sliced_control_data[0, i, :] = control_data_1[frameIdx // fps * audio_sample_rate:(frameIdx + window_size) // fps * audio_sample_rate].copy()
            sliced_control_data[1, i, :] = control_data_2[frameIdx // fps * audio_sample_rate:(frameIdx + window_size) // fps * audio_sample_rate].copy()
            sliced_state_data[0, i, :] = state_data_1[frameIdx:frameIdx + window_size].copy()
            sliced_state_data[1, i, :] = state_data_2[frameIdx:frameIdx + window_size].copy()

    else:
        print("WARNING: data too small for window")

    return sliced_motion_data, sliced_control_data, sliced_state_data


def align_with_waveform(motion1, motion2, audio1, audio2, state1, state2, fps: int = 20, audio_sample_rate: int = 16000):
    """Truncates to the shortest length and concatenates"""

    motion1_time = motion1.shape[0] // fps
    motion2_time = motion2.shape[0] // fps
    audio1_time = audio1.shape[0] // audio_sample_rate
    audio2_time = audio2.shape[0] // audio_sample_rate
    state1_time = state1.shape[0] // fps
    state2_time = state2.shape[0] // fps
    min_time = min(motion1_time, motion2_time, audio1_time, audio2_time, state1_time, state2_time)

    return motion1[:min_time * fps, :], motion2[:min_time * fps, :], audio1[:min_time * audio_sample_rate], audio2[:min_time * audio_sample_rate], state1[:min_time * fps], state2[:min_time * fps]



def import_data_with_waveform(file_dict, mirror=False, fps=20, audio_sample_rate=16000, start=0,
                              end=None):
    suffix = ".npz"
    if mirror:
        suffix = "_mirrored.npz"

    motion_paths = file_dict['motion_path']
    audio_paths = file_dict['audio_path']
    state_paths = file_dict['state_label_path']

    motion_data_a = np.load(motion_paths[0]+suffix)['clips'].astype(np.float32)  # [Frame, Feature]
    motion_data_b = np.load(motion_paths[1]+suffix)['clips'].astype(np.float32)  # [Frame, Feature]
    n_motion_feats = motion_data_a.shape[1]

    control_data_a = np.load(audio_paths[0]+'.npy').astype(np.float32)  # [T,]
    control_data_b = np.load(audio_paths[1]+'.npy').astype(np.float32)  # [T,]

    state_data_a = np.load(state_paths[0]+'.npy').astype(int)
    state_data_b = np.load(state_paths[1]+'.npy').astype(int)

    clip_motion_data_1, clip_motion_data_2, clip_control_data_1, clip_control_data_2, clip_state_data_1, clip_state_data_2 = align_with_waveform(motion_data_a, motion_data_b, control_data_a, control_data_b, state_data_a, state_data_b, fps, audio_sample_rate)

    print(clip_motion_data_1.shape, clip_control_data_1.shape, clip_state_data_1.shape)
    if not end:
        end = clip_motion_data_1.shape[0]

    return clip_motion_data_1[start:end, :], clip_motion_data_2[start:end, :], \
        clip_control_data_1[start // fps * audio_sample_rate: end // fps * audio_sample_rate], clip_control_data_2[start // fps * audio_sample_rate: end // fps * audio_sample_rate], \
        clip_state_data_1[start:end], clip_state_data_2[start:end]


def import_and_slice_with_waveform(files, slice_window, slice_overlap, mirror=False, fps=20,
                                   audio_sample_rate=16000, start=0, end=None):
    out_sliced_motion_data = None
    out_sliced_control_data = None
    out_sliced_state_data = None
    fi = 0
    for pair_name, pair_value in files.items():
        print(pair_name, pair_value)
        motion_data_1, motion_data_2, control_data_1, control_data_2, state_data_1, state_data_2 = import_data_with_waveform(file_dict=pair_value, mirror=False,
                                                              fps=fps, audio_sample_rate=audio_sample_rate, start=start,
                                                              end=end)
        # slice dataset
        sliced_motion_data, sliced_control_data, sliced_state_data = slice_data_with_waveform(motion_data_1, motion_data_2, control_data_1, control_data_2, state_data_1, state_data_2,
                                                                           fps, audio_sample_rate, slice_window, slice_overlap)
        if mirror:
            motion_data_1_mir, motion_data_2_mir, control_data_1_mir, control_data_2_mir, state_data_1_mir, state_data_2_mir = import_data_with_waveform(file_dict=pair_value,
                                                                                            mirror=True, fps=fps,
                                                                                            audio_sample_rate=audio_sample_rate,
                                                                                            start=start, end=end)
            sliced_motion_data_mirr, sliced_control_data_mirr, sliced_state_data_mirr = slice_data_with_waveform(motion_data_1_mir, motion_data_2_mir, control_data_1_mir, control_data_2_mir, state_data_1_mir, state_data_2_mir,
                                                                                         fps, audio_sample_rate, slice_window, slice_overlap)

            # append to the sliced dataset
            sliced_motion_data = np.concatenate((sliced_motion_data, sliced_motion_data_mirr), axis=1)
            sliced_control_data = np.concatenate((sliced_control_data, sliced_control_data_mirr), axis=1)
            sliced_state_data = np.concatenate((sliced_state_data, sliced_state_data_mirr), axis=1)

        if fi == 0:
            out_sliced_motion_data = sliced_motion_data
            out_sliced_control_data = sliced_control_data
            out_sliced_state_data = sliced_state_data
        else:
            out_sliced_motion_data = np.concatenate((out_sliced_motion_data, sliced_motion_data), axis=1)
            out_sliced_control_data = np.concatenate((out_sliced_control_data, sliced_control_data), axis=1)
            out_sliced_state_data = np.concatenate((out_sliced_state_data, sliced_state_data), axis=1)
        # if fi == 0:
        #     out_data = sliced
        # else:
        #     out_data = np.concatenate((out_data, sliced), axis=0)
        fi = fi + 1

    return out_sliced_motion_data, out_sliced_control_data, out_sliced_state_data


def _scan_data_dir4photoreal_all(data_dir, all_files_dict, testset):
    """ Scan Data Dir & Get File Name, Dir, and Speaker ID. """
    train_files, test_files = {}, {}

    for id, info in all_files_dict.items():
        if id in train_files or id in test_files:
            pass

        motion_path = [info['motion_file_path'], info['motion_file_path']]
        audio_path = [info['audio_1_file_path'], info['audio_2_file_path']]
        state_label_path = [info['state_file_path'], info['state_file_path']]

        if id in testset:
            test_files[id] = {
                "motion_path": motion_path,
                "audio_path": audio_path,
                "state_label_path": state_label_path,
                # "start_frame": start_frame
                # "label_path": [os.path.join(label_dir, f+'.txt') for f in self.interact_data_info[id]["file_name"]],
                # "combined_audio_path": os.path.join(combined_audio_dir, id+'.wav'),
                # "info": self.interact_data_info[id]
            }
        else:
            train_files[id] = {
                            "motion_path": motion_path,
                            "audio_path": audio_path,
                            "state_label_path": state_label_path,
                            # "start_frame": start_frame
                            # "label_path": [os.path.join(label_dir, f+'.txt') for f in self.interact_data_info[id]["file_name"]],
                            # "combined_audio_path": os.path.join(combined_audio_dir, id+'.wav'),
                            # "info": self.interact_data_info[id]
                        }

    return train_files, test_files


def main(cfg_path) -> Optional[float]:

    cfg_para = OmegaConf.load(cfg_path)
    '''
        Converts bvh and wav files into features, slices in equal length intervals and divides the data
        into training, validation and test sets. '''
    # Hardcoded preprocessing params and file structure.
    # Modify these if you want the data in some different format
    train_window_secs = cfg_para.train_window_secs
    test_window_secs = cfg_para.test_window_secs
    window_overlap = cfg_para.window_overlap
    fps = cfg_para.fps
    is_standardize = cfg_para.is_standardize
    map_to_exponential = cfg_para.map_to_exponential
    extract_MFCC = cfg_para.extract_MFCC
    if not extract_MFCC:
        raise ValueError(
            "This preprocessing entry point requires extract_MFCC: true; "
            "the retained training pipeline does not include WavLM features."
        )
    data_root = cfg_para.data_dir

    bvhpath = os.path.join(data_root, 'motion')
    audiopath = os.path.join(data_root, 'audio')
    statepath = os.path.join(data_root, 'state')
    print('........bvhpath = ' + bvhpath)
    print('........audiopath = ' + audiopath)
    print('........statepath = ' + statepath)

    testset = cfg_para.testset
    processed_dir = cfg_para.processed_dir
    audio_sample_rate = cfg_para.audio_sample_rate
    position_smoothing = cfg_para.position_smoothing
    rotation_smoothing = cfg_para.rotation_smoothing
    constant_remover_eps = cfg_para.constant_remover_eps
    need_mirror_data = cfg_para.need_mirror_data

    files = []
    audio_files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(bvhpath):
        for file in sorted(f):
            if '.bvh' in file:
                ff = os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)
    for r, d, f in os.walk(audiopath):
        for file in sorted(f):
            if '.wav' in file:
                ff = os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                audio_files.append(basename)

    # processed data will be organized as following
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    motion_feature = 'joint_rot'
    speech_feature = 'audio'
    state_feature = 'state_label'

    if not map_to_exponential:
        exponential_state = "NoExp"
    else:
        exponential_state = "WithExp"

    if not extract_MFCC:
        audio_feature = "waveform"
    else:
        audio_feature = "MFCC"

    if not is_standardize:
        standardize_state = "NotStd"
    else:
        standardize_state = "WithStd"

    path = os.path.join(processed_dir,
                        f'feat_{fps}fps_{train_window_secs}s_{exponential_state}_{audio_feature}_{standardize_state}')
    motion_path = os.path.join(path, f'{motion_feature}')
    speech_path = os.path.join(path, f'{speech_feature}')
    state_label_path = os.path.join(path, f'{state_feature}')

    if not os.path.exists(path):
        os.makedirs(path)

    all_files_dict = {}
    for file in files:
        # # ! hack
        # file_index = file.split('_')[4]
        file_index = file.split('_')[2]
        print(file, file_index)
        motion_file_path = os.path.join(motion_path, file)
        if 'resample' in file:
            audio_1_file_path = os.path.join(speech_path, file_index+'_audio_left_resample_0_9')
            audio_2_file_path = os.path.join(speech_path, file_index+'_audio_right_resample_0_9')
        else:
            audio_1_file_path = os.path.join(speech_path, file_index+'_audio_left')
            audio_2_file_path = os.path.join(speech_path, file_index+'_audio_right')
        state_file_path = os.path.join(state_label_path, file)
        all_files_dict[file] = {
            'motion_file_path': motion_file_path,
            'audio_1_file_path': audio_1_file_path,
            'audio_2_file_path': audio_2_file_path,
            'state_file_path': state_file_path
        }

    # speech features
    print('Processing speech features...')
    print(os.path.exists(speech_path))
    if not os.path.exists(speech_path):
        os.makedirs(speech_path)
        if extract_MFCC:
            extract_melspec(audiopath, audio_files, speech_path, fps)
        else:   # extract waveform
            extract_waveform(audiopath, audio_files, speech_path, fps, audio_sample_rate)

    # joint angles
    print('Processing motion features...')
    if not os.path.exists(motion_path):
        os.makedirs(motion_path)

        extract_joint_angles(bvhpath, files, motion_path, fps, fullbody=cfg_para.fullbody,
                             map_to_exponential=map_to_exponential,
                             position_smoothing=position_smoothing, rotation_smoothing=rotation_smoothing,
                             constant_remover_eps=constant_remover_eps)

    # state labels
    print('Processing state label features...')
    if not os.path.exists(state_label_path):
        os.makedirs(state_label_path)

        process_state_labels(statepath, files, state_label_path, ds_rate=1)


    # copy pipeline for converting motion features to bvh
    copyfile(os.path.join(motion_path, 'data_pipe.sav'), os.path.join(path, f'data_pipe_{fps}fps.sav'))

    # divide in train, val, dev and test sets. Note that val and dev contains the same data allthough sliced in
    # different ways. - val data is used for logging and sliced the same way as the training data - dev data is
    # sliced in longer sequences and used for visualization (we found that shorter snippets are hard to subjectivly
    # evaluate)
    print("divide in train, val, dev and test sets.\n Preparing datasets...")

    # train_files = [f for f in files if f not in testset_files]
    train_files, testset_files = _scan_data_dir4photoreal_all(data_dir=path, all_files_dict=all_files_dict, testset=testset)

    slice_win_train = int(train_window_secs * fps)
    slice_win_test = int(test_window_secs * fps)
    val_test_split = int(20 * test_window_secs * fps)  # 10
    print(f"slice_win_train: {slice_win_train}, slice_win_test: {slice_win_test}, val_test_split: {val_test_split}")

    if extract_MFCC:
        train_motion, train_ctrl, train_state = import_and_slice(train_files, slice_win_train, window_overlap, mirror=need_mirror_data,
                                                                  fps=fps, audio_sample_rate=audio_sample_rate)
        val_motion, val_ctrl, val_state = import_and_slice(testset_files, slice_win_train, window_overlap, mirror=need_mirror_data,
                                                              fps=fps, audio_sample_rate=audio_sample_rate, start=0,
                                                              end=val_test_split)
    else:
        train_motion, train_ctrl, train_state = import_and_slice_with_waveform(train_files, slice_win_train, window_overlap, mirror=need_mirror_data,
                                                                  fps=fps, audio_sample_rate=audio_sample_rate)
        val_motion, val_ctrl, val_state = import_and_slice_with_waveform(testset_files, slice_win_train, window_overlap, mirror=need_mirror_data,
                                                              fps=fps, audio_sample_rate=audio_sample_rate, start=0,
                                                              end=val_test_split)

    # print("here", train_ctrl.shape)
    if is_standardize:
        print("Standardize Processing...........")
        if not extract_MFCC:
            train_ctrl = np.expand_dims(train_ctrl, axis=3).swapaxes(2, 3)
            val_ctrl = np.expand_dims(val_ctrl, axis=3).swapaxes(2, 3)

        a, b, c, d = train_ctrl.shape
        train_ctrl = train_ctrl.reshape(a * b, c, d)
        train_ctrl, input_scaler = fit_and_standardize(train_ctrl)
        train_ctrl = train_ctrl.reshape(a, b, c, d)

        a, b, c, d = train_motion.shape
        train_motion = train_motion.reshape(a * b, c, d)
        train_motion, output_scaler = fit_and_standardize(train_motion)
        train_motion = train_motion.reshape(a, b, c, d)

        a, b, c, d = val_ctrl.shape
        val_ctrl = val_ctrl.reshape(a * b, c, d)
        val_ctrl = standardize(val_ctrl, input_scaler)
        val_ctrl = val_ctrl.reshape(a, b, c, d)

        a, b, c, d = val_motion.shape
        val_motion = val_motion.reshape(a * b, c, d)
        val_motion = standardize(val_motion, output_scaler)
        val_motion = val_motion.reshape(a, b, c, d)

        if not extract_MFCC:
            for test_file_k, test_file_v in testset_files.items():
                temp_test_file_list = {test_file_k: test_file_v}
                print(f'import_and_slice raw audio: {temp_test_file_list}')
                test_motion, test_ctrl, test_state = import_and_slice_with_waveform(temp_test_file_list, slice_win_test, 0,
                                                                        mirror=False, fps=fps,
                                                                        audio_sample_rate=audio_sample_rate, start=0)
                test_ctrl = np.expand_dims(test_ctrl, axis=3).swapaxes(2, 3)

                a, b, c, d = test_ctrl.shape
                test_ctrl = test_ctrl.reshape(a * b, c, d)
                test_ctrl = standardize(test_ctrl, input_scaler)
                test_ctrl = test_ctrl.reshape(a, b, c, d)

                a, b, c, d = test_motion.shape
                test_motion = test_motion.reshape(a * b, c, d)
                test_motion = standardize(test_motion, output_scaler)
                test_motion = test_motion.reshape(a, b, c, d)

                test_ctrl = np.squeeze(test_ctrl, axis=2)

                np.savez(os.path.join(path, f'test_output_{test_file_k}_{fps}fps.npz'), clips=test_motion)
                np.savez(os.path.join(path, f'test_input_{test_file_k}_{fps}fps.npz'), clips=test_ctrl)
        else:
            for test_file_k, test_file_v in testset_files.items():
                temp_test_file_list = {test_file_k: test_file_v}
                print(f'import_and_slice MFCC: {temp_test_file_list}')
                test_motion, test_ctrl, test_state = import_and_slice(temp_test_file_list, slice_win_test, 0,
                                                                        mirror=False, fps=fps,
                                                                        audio_sample_rate=audio_sample_rate, start=0)

                print(test_ctrl.shape)
                a, b, c, d = test_ctrl.shape
                test_ctrl = test_ctrl.reshape(a * b, c, d)
                test_ctrl = standardize(test_ctrl, input_scaler)
                test_ctrl = test_ctrl.reshape(a, b, c, d)

                a, b, c, d = test_motion.shape
                test_motion = test_motion.reshape(a * b, c, d)
                test_motion = standardize(test_motion, output_scaler)
                test_motion = test_motion.reshape(a, b, c, d)

                np.savez(os.path.join(path, f'test_output_{test_file_k}_{fps}fps.npz'), clips=test_motion)
                np.savez(os.path.join(path, f'test_input_{test_file_k}_{fps}fps.npz'), clips=test_ctrl)

        if not extract_MFCC:
            train_ctrl = np.squeeze(train_ctrl, axis=2)
            val_ctrl = np.squeeze(val_ctrl, axis=2)

        jl.dump(input_scaler, os.path.join(path, f'input_scaler.sav'))
        jl.dump(output_scaler, os.path.join(path, f'output_scaler.sav'))
    else:
        print("Skip Standardizing......")
    print('Save train/val npz!!!')
    np.savez(os.path.join(path, f'train_output_{fps}fps.npz'), clips=train_motion)
    np.savez(os.path.join(path, f'train_input_{fps}fps.npz'), clips=train_ctrl)
    np.savez(os.path.join(path, f'train_state_{fps}fps.npz'), clips=train_state)
    np.savez(os.path.join(path, f'val_output_{fps}fps.npz'), clips=val_motion)
    np.savez(os.path.join(path, f'val_input_{fps}fps.npz'), clips=val_ctrl)
    np.savez(os.path.join(path, f'val_state_{fps}fps.npz'), clips=val_state)

if __name__ == "__main__":
    import sys

    cfg_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else './configs/preprocessing/audio.yaml'
    )
    main(cfg_path)
