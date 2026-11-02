# region Import

import os
import torch
import argparse
import shutil
import numpy as np
import time
import librosa
import joblib as jl

from data_process.utils.utils import inv_standardize, standardize
from Gesture_trainer import to_bvh, MixTask_trainer
from data_process.utils.audio_feature import AudioFeature


import sys
sys.path.append('./data_process')

def streaming_generate_diffusion(data_dir, trainer: MixTask_trainer, solver='DDIM', cached_body_latent=None,
                                 cached_audio_feats=None, sample_steps=50, guidance_weight=2.0, eta=1.0, have_text=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        # [1, T, F_total]
        audio_feats = np.load(os.path.join(data_dir, "audio_features.npy")).astype(np.float32)
        audio_feats = torch.from_numpy(audio_feats).to(device)


        if audio_feats.shape[1] > trainer.config.model.window_size_max_len:
            print(f"[WARNING] audio_feats length {audio_feats.shape[1]} exceeds window_size_max_len ")
            audio_feats = audio_feats[:, :trainer.config.model.window_size_max_len, :]


        if have_text:
            input_text_feats = np.load(os.path.join(data_dir, "text_features.npy")).astype(np.float32)
            input_text_feats = torch.from_numpy(input_text_feats).to(device)
            # print(f"text features shape: {input_text_feats.shape}")
        else:


            input_text_feats = torch.zeros((1, 77, 512), device=device, dtype=torch.float32)


        cache_length = trainer.config.model.window_size_max_len - audio_feats.shape[1]

        if cache_length < trainer.config.model.window_size_max_len and cache_length > 0:
            cached_body_latent = cached_body_latent[:, -cache_length:] if cached_body_latent is not None else None
            input_cached_audio = cached_audio_feats[:, -cache_length:, :] if cached_audio_feats is not None else None
        elif cache_length == 0:
            cached_body_latent = None
            input_cached_audio = None
        else:
            raise ValueError("cache length")


        if input_cached_audio is not None:
            input_audio_feats = torch.cat((input_cached_audio, audio_feats), dim=1)
        else:
            input_audio_feats = audio_feats


        init_body_latent = cached_body_latent if cached_body_latent is not None else None
        solver_name = f'{solver}_edge' if init_body_latent is None else f'{solver}_edge_w_prior'


        current_modality = 'text' if have_text else 'audio'


        body, cache_latent = trainer.render_sample_streaming_g1_mix(
            inference_sovler=solver_name,
            cond_audio=input_audio_feats,
            cond_text=input_text_feats,
            modality=current_modality,
            previous_motion_last_half=init_body_latent,
            cache_length=cache_length,
            sample_steps=sample_steps,
            guidance_weight=guidance_weight,
            eta=eta
        )


    cached_body_latent = cache_latent

    return cached_body_latent, body.squeeze(0), audio_feats


def server_generate_diffusion_mix_generation(audio_feats, trainer: MixTask_trainer, solver='DDIM', cached_body_latent=None,
                                 cached_audio_feats=None, sample_steps=50, guidance_weight=2.0, eta=1.0, have_text=False, text_features=None, mogen_padding=0.0):
    if have_text:
        assert text_features is not None, "When have_text is True, text_features must be provided."
        # print(f"Using text features for generation, shape: {text_features.shape}")
    # else:
        # print("No text features provided, proceeding without text guidance.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        audio_feats = torch.from_numpy(audio_feats).to(device)
        if audio_feats.shape[1] > trainer.config.model.window_size_max_len:
            print(f"[WARNING] audio_feats length {audio_feats.shape[1]} exceeds window_size_max_len ")
            audio_feats = audio_feats[:, :trainer.config.model.window_size_max_len, :]


        if have_text and text_features is not None:
            input_text_feats = torch.from_numpy(text_features).to(device)
            input_text_feats = input_text_feats[0:1] # [1, 77, D]

            assert audio_feats.shape[1] < 121, f"audio_feats length should be less than 121, got {audio_feats.shape[1]}"
            text_padding_length = 121 - audio_feats.shape[1]
            audio_feats = torch.cat((audio_feats, torch.zeros((1, text_padding_length, audio_feats.shape[2]), device=device)), dim=1)
        else:
            input_text_feats = torch.zeros((1, 77, 512), device=device, dtype=torch.float32)


        cache_length = trainer.config.model.window_size_max_len - audio_feats.shape[1]

        if cache_length < trainer.config.model.window_size_max_len and cache_length > 0:
            if cached_body_latent is not None:
                if cached_body_latent.shape[1] >= cache_length:
                    cached_body_latent = cached_body_latent[:, -cache_length:]
                else:
                    repeat_length = cache_length - cached_body_latent.shape[1]
                    repeat_body_latent = cached_body_latent[:, 0:1, :].repeat(1, repeat_length, 1)
                    cached_body_latent = torch.cat((repeat_body_latent, cached_body_latent), dim=1)
            else:
                cached_body_latent = None

            if cached_audio_feats is not None:
                if cached_audio_feats.shape[1] >= cache_length:
                    input_cached_audio = cached_audio_feats[:, -cache_length:, :]
                else:
                    repeat_length = cache_length - cached_audio_feats.shape[1]
                    repeat_cached_audio = cached_audio_feats[:, 0:1, :].repeat(1, repeat_length, 1)
                    input_cached_audio = torch.cat((repeat_cached_audio, cached_audio_feats), dim=1)
            else:
                input_cached_audio = None

        elif cache_length == 0:
            cached_body_latent = None
            input_cached_audio = None
        else:
            raise ValueError("cache length")

        if input_cached_audio is not None:
            input_audio_feats = torch.cat((input_cached_audio, audio_feats), dim=1)
        else:
            input_audio_feats = audio_feats

        init_body_latent = cached_body_latent if cached_body_latent is not None else None
        solver_name = f'{solver}_edge' if init_body_latent is None else f'{solver}_edge_w_prior'

        current_modality = 'text' if (have_text and text_features is not None) else 'audio'


        body, cache_latent = trainer.render_sample_streaming_g1_mix(
            inference_sovler=solver_name,
            cond_audio=input_audio_feats,
            cond_text=input_text_feats,
            modality=current_modality,
            previous_motion_last_half=init_body_latent,
            cache_length=cache_length,
            sample_steps=sample_steps,
            guidance_weight=guidance_weight,
            eta=eta
        )


    motion_padding_after = int(mogen_padding * trainer.config.data.framerate)
    cached_body_latent = cache_latent[:, :-motion_padding_after] if motion_padding_after > 0 else cache_latent
    body = body[:, :-motion_padding_after, :] if motion_padding_after > 0 else body
    audio_feats = audio_feats[:, :-motion_padding_after, :] if motion_padding_after > 0 else audio_feats

    return cached_body_latent, body.squeeze(0), audio_feats


def server_generate_diffusion(audio_feats, trainer: MixTask_trainer, solver='DDIM', cached_body_latent=None,
                                 cached_audio_feats=None, sample_steps=50, guidance_weight=2.0, eta=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        audio_feats = torch.from_numpy(audio_feats).to(device)


        cache_length = trainer.config.model.window_size_max_len - audio_feats.shape[1]

        if cache_length < trainer.config.model.window_size_max_len and cache_length > 0:
            if cached_body_latent is not None:
                if cached_body_latent.shape[1] >= cache_length:
                    cached_body_latent = cached_body_latent[:, -cache_length:]
                else:
                    repeat_length = cache_length - cached_body_latent.shape[1]
                    repeat_body_latent = cached_body_latent[:, 0:1, :].repeat(1, repeat_length, 1)
                    cached_body_latent = torch.cat((repeat_body_latent, cached_body_latent), dim=1)
            else:
                cached_body_latent = None

            if cached_audio_feats is not None:
                if cached_audio_feats.shape[1] >= cache_length:
                    input_cached_audio = cached_audio_feats[:, -cache_length:, :]
                else:
                    repeat_length = cache_length - cached_audio_feats.shape[1]
                    repeat_cached_audio = cached_audio_feats[:, 0:1, :].repeat(1, repeat_length, 1)
                    input_cached_audio = torch.cat((repeat_cached_audio, cached_audio_feats), dim=1)
            else:
                input_cached_audio = None

        elif cache_length == 0:
            cached_body_latent = None
            input_cached_audio = None
        else:
            raise ValueError("cache length")

        if input_cached_audio is not None:
            input_audio_feats = torch.cat((input_cached_audio, audio_feats), dim=1)
        else:
            input_audio_feats = audio_feats

        init_body_latent = cached_body_latent if cached_body_latent is not None else None
        solver_name = f'{solver}_edge' if init_body_latent is None else f'{solver}_edge_w_prior'


        dummy_text = torch.zeros((1, 77, 512), device=device, dtype=torch.float32)


        body, cache_latent = trainer.render_sample_streaming_g1_mix(
            inference_sovler=solver_name,
            cond_audio=input_audio_feats,
            cond_text=dummy_text,
            modality='audio',
            previous_motion_last_half=init_body_latent,
            cache_length=cache_length,
            sample_steps=sample_steps,
            guidance_weight=guidance_weight,
            eta=eta
        )

    cached_body_latent = cache_latent

    return cached_body_latent, body.squeeze(0), audio_feats


def smooth_transition_tensor(total_data, data, k):
    """
    Smoothly interpolate between total_data[-1] and data[:k] to avoid abrupt changes.

    Args:
        total_data (torch.Tensor): Tensor of shape (N, dim), existing cached data.
        data (torch.Tensor): Tensor of shape (M, dim), newly generated data.
        k (int): Number of frames over which to interpolate.

    Returns:
        torch.Tensor: Smoothed data of shape (M, dim), ready for concatenation.
    """
    assert len(total_data.shape) == 2 and len(data.shape) == 2, "Inputs must be 2D tensors"
    if k > data.shape[0]:
        k = data.shape[0]

    last_frame = total_data[-1]  # shape: (dim,)
    weights = torch.linspace(1.0, 0.0, k, device=data.device).view(-1, 1)  # shape: (k, 1)
    interpolated = weights * last_frame + (1 - weights) * data[:k]  # shape: (k, dim)
    smoothed_data = torch.cat([interpolated.clone().detach(), data[k:]], dim=0)  # Concatenate along the first dimension

    return smoothed_data

def _compute_channel_feats(wav_path: str, fps: int, sr: int = 15360) -> np.ndarray:
    data, fs = librosa.load(wav_path, sr=sr)
    assert fs % fps == 0, f"Sample rate {fs} must be divisible by fps={fps}"

    melspe_db = AudioFeature.get_melspectrogram(data, fs)
    mfcc = AudioFeature.get_mfcc(melspe_db)        # (20, T)
    mfcc_delta = AudioFeature.get_mfcc_delta(mfcc) # (20, T)

    audio_harmonic, audio_percussive = AudioFeature.get_hpss(data)
    chroma_cqt = AudioFeature.get_chroma_cqt(audio_harmonic, fs, octave=(7 if fs == 60*512 else 5))  # (12, T)

    onset_env = AudioFeature.get_onset_strength(audio_percussive, fs)  # (T,)
    tempogram = AudioFeature.get_tempogram(onset_env, fs)              # (384, T)

    feats = np.concatenate(
        [mfcc, mfcc_delta, chroma_cqt, onset_env.reshape(1, -1), tempogram],
        axis=0
    ).T  # -> [T, 437]
    return feats.astype(np.float32)


def _standardize_2d_lastdim(feats_2d: np.ndarray, scaler) -> np.ndarray:
    if scaler is None:
        raise ValueError("")
    batched = feats_2d[None, :, :]           # [1, T, F]
    stded = standardize(batched, scaler)
    return stded[0]

if __name__ == '__main__':
    # region Args Parser

    from transformers import CLIPTextModel, CLIPTokenizer
    clip_dir = os.getenv("MOTION_CLIP_MODEL", "openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
    text_encoder = CLIPTextModel.from_pretrained(clip_dir)
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()


    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_path', type=str, required=True, default = '')
    parser.add_argument('--text_path', type=str, required=True, default = '')
    parser.add_argument('--save_dir', type=str, required=True, default = '')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--swap_speaker', action='store_true', default=False)
    parser.add_argument('--ds_rate', type=int, default=1)
    parser.add_argument('--diffusion_config_path', type=str, default='./configs/legacy/triplet.yaml')
    parser.add_argument('--solver', type=str, default='DDIM')

    args = parser.parse_args()

    print("Streaming inference mode is enabled.")
    start_time = time.time()

    print("Preparing data...")
    text_name = []
    audio_name_left = []
    audio_name_right = []
    def extract_index(name):
        base = os.path.basename(name)
        idx_str = base.split('_sample_')[-1].split('_')[0]
        return int(idx_str)

    for text_file in os.listdir(args.text_path):
        if text_file.endswith('.txt'):
            text_name.append(os.path.splitext(text_file)[0])

    trainer = MixTask_trainer(args.diffusion_config_path)

    audio_data_root = trainer.config.data.data_root
    audio_input_scaler = jl.load(os.path.join(audio_data_root, 'input_scaler_audio.sav'))
    output_scaler = jl.load(os.path.join(trainer.config.data.data_root, 'output_scaler.sav'))
    data_pipe = jl.load(os.path.join(trainer.config.data.data_root, 'data_pipe_' + str(trainer.config.data.framerate) + 'fps.sav'))

    print(f"Time taken for initialization: {time.time() - start_time:.2f} seconds")

    for audio_file in os.listdir(args.audio_path):
        if args.swap_speaker:
            if audio_file.endswith('right.wav'):
                audio_name_left.append(os.path.splitext(audio_file)[0])
            if audio_file.endswith('left.wav'):
                audio_name_right.append(os.path.splitext(audio_file)[0])
        else:
            if audio_file.endswith('left.wav'):
                audio_name_left.append(os.path.splitext(audio_file)[0])
            if audio_file.endswith('right.wav'):
                audio_name_right.append(os.path.splitext(audio_file)[0])

    text_name.sort(key=extract_index)
    audio_name_left.sort(key=extract_index)
    audio_name_right.sort(key=extract_index)
    assert len(audio_name_left) == len(audio_name_right), "The number of left and right audio files must be the same."

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu', args.local_rank)
    text_encoder = text_encoder.to(device)

    total_data = None
    cached_z_body = None
    cached_audio_feats = None

    text_i = 0
    for i in range(len(audio_name_left)):
        new_save_dir = os.path.join(args.save_dir, audio_name_left[i])
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(new_save_dir):
            os.mkdir(new_save_dir)

        left_wav_path  = os.path.join(args.audio_path, audio_name_left[i]  + '.wav')
        right_wav_path = os.path.join(args.audio_path, audio_name_right[i] + '.wav')


        feats_left  = _compute_channel_feats(left_wav_path,  fps=trainer.config.data.framerate)   # float32
        feats_right = _compute_channel_feats(right_wav_path, fps=trainer.config.data.framerate)   # float32


        T_min = min(feats_left.shape[0], feats_right.shape[0])
        if feats_left.shape[0] != feats_right.shape[0]:
            print(f"[INFO] L/R length mismatch -> truncate to min T={T_min} (L={feats_left.shape[0]}, R={feats_right.shape[0]})")
        feats_left  = feats_left[:T_min]
        feats_right = feats_right[:T_min]



        feats_left = np.expand_dims(feats_left, axis=0)   # [1, T, 437]
        feats_right = np.expand_dims(feats_right, axis=0) # [1, T, 437]
        feats_all = np.concatenate((feats_left, feats_right), axis=2)  # [1, T, 874]
        audio_features = standardize(feats_all, audio_input_scaler)   # [1, T, 874]

        np.save(os.path.join(new_save_dir, "audio_features.npy"), audio_features)


        have_text = False
        if text_i < len(text_name):
            text_id = int(os.path.basename(text_name[text_i]).split('_sample_')[-1].split('_')[0])
            if text_id == i:
                shutil.copy(os.path.join(args.text_path, text_name[text_i] + '.txt'),
                                os.path.join(new_save_dir, text_name[text_i] + '.txt'))
                text_name_path = os.path.join(new_save_dir, text_name[text_i] + '.txt')
                with open(text_name_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                # Tokenize and encode text
                print("Encoding text:", text_content)
                inputs = tokenizer(text_content, return_tensors="pt", padding='max_length', truncation=True, max_length=77)
                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    text_embeddings = text_encoder(**inputs).last_hidden_state  # [1, 77, 512]
                text_embeddings = text_embeddings.cpu().numpy().astype(np.float32)
                np.save(os.path.join(new_save_dir, "text_features.npy"), text_embeddings)
                text_i += 1
                have_text = True

        print(f"Time taken for {audio_name_left[i]} data preparation: {time.time() - start_time:.2f} seconds")

        print("Generating for {}...".format(audio_name_left[i]))

        current_cached_z_body, data, current_cached_audio_feats = \
            streaming_generate_diffusion(data_dir=new_save_dir,
                                        trainer=trainer,
                                        solver=args.solver,
                                        cached_body_latent=cached_z_body,
                                        cached_audio_feats=cached_audio_feats,
                                        sample_steps=trainer.config.inference.sample_steps,
                                        guidance_weight=trainer.config.inference.guidance_weight,
                                        eta=trainer.config.inference.eta,
                                        have_text=have_text)

        # Update cached_z_body for next iteration
        if cached_z_body is None:
            cached_z_body = current_cached_z_body
        else:
            cached_z_body = torch.cat((cached_z_body, current_cached_z_body), dim=1)

        assert current_cached_audio_feats.shape[2] % 2 == 0, "Feature dim must be even to split L/R."



        if cached_audio_feats is None:
            cached_audio_feats = current_cached_audio_feats
        else:
            cached_audio_feats = torch.cat((cached_audio_feats, current_cached_audio_feats), dim=1)

        # print("cached_z_body shape:", cached_z_body.shape)
        # print("cached_audio_feats shape:", cached_audio_feats.shape)

        if total_data is None:
            total_data = data
        else:
            # print(f"Total data shape before concatenation: {total_data.shape}, current data shape: {data.shape}")
            data = smooth_transition_tensor(total_data, data, k=15)
            total_data = torch.cat((total_data, data), dim=0)

        end_time = time.time()
        print(f"Time taken for {audio_name_left[i]} total generation: {end_time - start_time:.2f} seconds")

    # delete all files in the save_dir
    search_dir = os.path.dirname(new_save_dir)
    for file in os.listdir(search_dir):
        file_path = os.path.join(search_dir, file)
        # print(f"Deleting file: {file_path}")
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

    prefix = 'right' if args.swap_speaker else 'left'
    bvh_save_path = os.path.join(search_dir, os.path.basename(args.audio_path) + f'_{prefix}_text.bvh')
    total_data = total_data.cpu().numpy()
    total_data = inv_standardize(np.expand_dims(total_data, axis=0), output_scaler)
    to_bvh(total_data, bvh_save_path, data_pipe, trainer.config.data.framerate)
