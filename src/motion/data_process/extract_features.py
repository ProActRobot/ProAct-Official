"""Based on: https://github.com/simonalexanderson/StyleGestures"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import os
import librosa
import librosa.display
import joblib as jl
import tqdm


from data_process.pymo.parsers import BVHParser
from data_process.pymo.data import Joint, MocapData
from data_process.pymo.preprocessing import *
from data_process.pymo.writers import *
from data_process.utils.audio_feature import AudioFeature


def nan_smooth(data, filt_len):
    win = filt_len // 2

    nframes = len(data)
    out = np.zeros(data.shape)
    for i in range(nframes):
        if i < win:
            st = 0
        else:
            st = i - win
        if i > nframes - win - 1:
            en = nframes
        else:
            en = i + win + 1
        out[i, :] = np.nanmean(data[st:en, :], axis=0)
    return out


def extract_joint_angles(bvh_dir, files, destpath, fps, fullbody=True, map_to_exponential=True,
                         position_smoothing=5, rotation_smoothing=10, constant_remover_eps=1e-6):
    p = BVHParser()

    data_all = list()
    print(f"Importing data... total: {len(files)}")
    for f in files:
        ff = os.path.join(bvh_dir, f + '.bvh')
        data_all.append(p.parse(ff))

    if fullbody:
        # full body
        if map_to_exponential:
            data_pipe = Pipeline([
                ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
                # for zeggs skeleton:
                # ('mir', Mirror(axis='X', append=True)),

                # for interact skeleton:
                # ('mir', Mirror(axis='Z', append=True)),

                # for G1 skeleton:
                ('mir', Mirror_G1(axis='Y', append=True)),

                # for photoreal skeleton:
                # ('mir', Mirror_photoreal(axis='Z', append=True)),

                # for SeG skeleton:
                # ('mir', Mirror(axis='X', append=True)),

                # ZEGGS:
                # ('jtsel', JointSelector(
                #     ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1',
                #      'RightHandThumb2', 'RightHandThumb3', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3',
                #      'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                #      'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
                #      'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'RightUpLeg',
                #      'RightLeg', 'RightFoot', 'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'],
                #     include_root=True)),
                # InterAct:
                # ('jtsel', JointSelector(
                #     ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
                #      'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                #      'RightHandThumb1', 'RightHandThumb2', 'RightHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandMiddle1', 'RightHandMiddle2',
                #      'RightHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandPinky', 'RightHandPinky1', 'RightHandPinky2',
                #      'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                #      'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandMiddle1', 'LeftHandMiddle2',
                #      'LeftHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandPinky', 'LeftHandPinky1', 'LeftHandPinky2',
                #      'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot'],
                #     include_root=True)),
                # photoreal:
                # if G1, not use 'jtsel'
                # ('jtsel', JointSelector(
                #     ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_neck0', 'b_head',  'b_l_shoulder', 'p_l_scap', 'b_l_arm', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3',  'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'p_r_scap', 'b_r_arm', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3',  'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_l_upleg', 'b_l_leg', 'b_l_foot_twist', 'b_l_subtalar', 'b_r_upleg', 'b_r_leg', 'b_r_foot_twist', 'b_r_subtalar'],
                #     include_root=True)),
                ('root', RootTransformer('pos_rot_deltas', position_smoothing=position_smoothing,
                                         rotation_smoothing=rotation_smoothing)),
                ('exp', MocapParameterizer('expmap')),
                ('cnst', ConstantsRemover(eps=constant_remover_eps)),
                ('npf', Numpyfier())
            ])
        else:
            data_pipe = Pipeline([
                ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
                ('mir', Mirror(axis='X', append=True)),
                ('jtsel', JointSelector(
                    ['Spine', 'Spine1', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot',
                     'RightToeBase', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'], include_root=True)),
                ('root', RootTransformer('pos_rot_deltas', position_smoothing=position_smoothing,
                                         rotation_smoothing=rotation_smoothing)),
                ('cnst', ConstantsRemover(eps=constant_remover_eps)),
                ('npf', Numpyfier())
            ])
    else:
        # Upper body
        if map_to_exponential:
            data_pipe = Pipeline([
                ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
                ('root', RootTransformer('hip_centric')),
                ('mir', Mirror(axis='X', append=True)),
                ('jtsel', JointSelector(
                    ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm',
                     'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'],
                    include_root=True)),
                ('exp', MocapParameterizer('expmap')),
                ('cnst', ConstantsRemover()),
                ('np', Numpyfier())
            ])
        else:
            data_pipe = Pipeline([
                ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
                ('root', RootTransformer('hip_centric')),
                ('mir', Mirror(axis='X', append=True)),
                ('jtsel', JointSelector(
                    ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm',
                     'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'],
                    include_root=True)),
                ('cnst', ConstantsRemover()),
                ('np', Numpyfier())
            ])

    print("Processing...")
    out_data = data_pipe.fit_transform(data_all)
    # print(out_data[0, 0, 103:107])

    # the datapipe will append the mirrored files to the end
    assert len(out_data) == 2 * len(files)

    jl.dump(data_pipe, os.path.join(destpath, 'data_pipe.sav'))

    fi = 0
    for f in files:
        ff = os.path.join(destpath, f)

        np.savez(ff + ".npz", clips=out_data[fi])
        print(f"{f} motion shape: {np.shape(out_data[fi])}")
        np.savez(ff + "_mirrored.npz", clips=out_data[len(files) + fi])
        fi = fi + 1


def extract_waveform(audiopath, files, destpath, fps, sample_rate):
    for f in tqdm(files, desc=f'Resampling all the Audios to {sample_rate}'):
        file = os.path.join(audiopath, f+'.wav')
        full_audio_data, sr = librosa.load(file, sr=sample_rate)
        outfile = os.path.join(destpath, f'{f}.npy')
        np.save(outfile, full_audio_data)


def extract_melspec(audio_dir, files, destpath, fps):

    for f in tqdm(files, desc=f'Extract MFCC with {fps}_fps'):
        file = os.path.join(audio_dir, f + '.wav')
        outfile = destpath + '/' + f + '.npy'

        data, fs = librosa.load(file, sr=15360)
        print(fs, fps)
        assert fs % fps == 0

        melspe_db = AudioFeature.get_melspectrogram(data, fs)
        mfcc = AudioFeature.get_mfcc(melspe_db)  # 20, num_frames
        mfcc_delta = AudioFeature.get_mfcc_delta(mfcc)  # 20, num_frames

        audio_harmonic, audio_percussive = AudioFeature.get_hpss(data)
        chroma_cqt = AudioFeature.get_chroma_cqt(audio_harmonic, fs, octave=(7 if fs == 60*512 else 5))  # 12, num_frames. octave is 7 if sr=60*512 else 5

        onset_env = AudioFeature.get_onset_strength(audio_percussive, fs)  # num_frames,

        tempogram = AudioFeature.get_tempogram(onset_env, fs)  # 384, num_frames

        # if chroma_cqt has different num_frames, align it to mfcc
        frames = mfcc.shape[1]
        chroma_cqt = chroma_cqt[:, :frames]

        features = np.concatenate([
            mfcc,
            mfcc_delta,
            chroma_cqt,
            onset_env.reshape(1, -1),
            tempogram
        ], axis=0).transpose()  # num_frames, 437

        feature_table = {
            "mfcc": 20,
            "mfcc_delta": 20,
            "chroma_cqt": 12,
            "onset_env": 1,
            # "tempogram": 384
            "tempogram": 128
        }
        # total shape: 437
        # total shape: 181

        print(f"\n audio feature {f}:=  {np.shape(features)}, saved to {outfile}")
        np.save(outfile, features)


# def extract_melspec(audio_dir, files, destpath, fps, n_mels: int = 20):
#     print('Extract MFCC... ...')
#     for f in tqdm(files, desc=f'Extract MFCC with {fps}_fps_{n_mels}_mels'):
#         file = os.path.join(audio_dir, f + '.wav')
#         outfile = destpath + '/' + f + '.npy'

#         X, fs = librosa.load(file, sr=None)

#         assert fs % fps == 0

#         hop_len = int(fs / fps)

#         n_fft = int(fs * 0.13)
#         C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=2048, hop_length=hop_len, n_mels=n_mels, fmin=0.0,
#                                            fmax=8000)
#         C = np.log(C)
#         np.save(outfile, np.transpose(C))
#         print(f"\n audio feature {f}:=  {np.shape(np.transpose(C))}")


def process_state_labels(statepath, files, state_label_path, ds_rate=1):
    assert ds_rate == 1
    word_to_int = {"stand": 0, "sit": 1, "walk": 2}
    for f in files:
        state_file = os.path.join(statepath, f+'.txt')

        with open(state_file, "r") as file:
            file_content = file.readlines()


        int_array = np.array([word_to_int[word.strip()] for word in file_content], dtype=int)


        outfile = os.path.join(state_label_path, f'{f}.npy')
        np.save(outfile, int_array)
        print(f, int_array.shape)


if __name__ == "__main__":
    bvh_dir = './data/photoreal/motion_x_add_end'
    files = ['pxb_fixed_scene01_body_pose']
    destpath = './data/photoreal/motion_x_add_end_test'
    os.makedirs(destpath, exist_ok=True)
    fps = 30
    extract_joint_angles(bvh_dir, files, destpath, fps)
