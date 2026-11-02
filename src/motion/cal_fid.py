import os
import torch
import argparse
import shutil
import numpy as np
import time
import librosa
import joblib as jl
from data_process.pymo.parsers import BVHParser
import scipy.linalg

from data_process.utils.utils import inv_standardize, standardize
from Gesture_trainer import to_bvh, Controlnet_trainer, MixTask_trainer
from data_process.utils.audio_feature import AudioFeature


import sys
sys.path.append('./data_process')

def extract_features(motion_list):
    """
    Extract features for evaluation.
    Includes a fix for list-wrapped data from pymo pipelines.
    """
    feats = []
    for motion in motion_list:

        if isinstance(motion, list):
            m = motion[0]
        else:
            m = motion


        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()



        if m.ndim == 3:
            m = m[0]



        feature = np.mean(m, axis=0)
        feats.append(feature)

    return np.array(feats, dtype=np.float32)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("FID calculation produced nan/inf, adding epsilon to diagonal.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_diversity(feats):
    """
    Calculates diversity as the average pairwise Euclidean distance.
    Robust version with explicit type casting.
    """

    if not isinstance(feats, np.ndarray):
        feats = np.array(feats)


    try:
        feats = feats.astype(np.float32)
    except ValueError:
        print("Error: Could not cast features to float32. Check for ragged arrays.")
        return 0.0



    if feats.ndim == 1:
        feats = feats[None, :]

    num_samples = feats.shape[0]
    if num_samples <= 1:
        return 0.0


    # Broadcasting: (N, 1, D) - (1, N, D) -> (N, N, D)
    diff = feats[:, None, :] - feats[None, :, :]



    dist = np.linalg.norm(diff, axis=2)  # (N, N) matrix


    sum_dist = np.sum(np.triu(dist, k=1))


    num_pairs = num_samples * (num_samples - 1) / 2

    return sum_dist / num_pairs

def get_distribution_statistics(feats):
    """
    Calculate mean and covariance.
    Robust to small sample sizes (N < 2).
    """

    if feats.ndim == 1:
        feats = feats[None, :]

    num_samples = feats.shape[0]
    feature_dim = feats.shape[1]


    mu = np.mean(feats, axis=0)



    if num_samples < 2:
        print(f"Warning: Not enough samples to compute Covariance (N={num_samples}). Returning Zero Matrix.")
        sigma = np.zeros((feature_dim, feature_dim))
    else:
        try:

            sigma = np.cov(feats, rowvar=False)
        except Exception as e:
            print(f"Warning: np.cov failed ({e}). Returning Identity Matrix.")
            sigma = np.eye(feature_dim)

    return mu, sigma

def sequence_generate_diffusion(data_dir, trainer, solver='DDIM',
                                 sample_steps=50, guidance_weight=3.0, eta=1.0):
    # print(f"sample_steps: {sample_steps}, guidance_weight: {guidance_weight}, eta: {eta}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        input_text_feats = np.load(os.path.join(data_dir, "text_features.npy")).astype(np.float32)
        input_text_feats = torch.from_numpy(input_text_feats).to(device)
        print(f"text features shape: {input_text_feats.shape}")
        solver_name = f'{solver}_edge'
        body = trainer.render_sample_text(
            solver_name, input_text_feats, sample_steps, guidance_weight, eta
        )
        assert body.shape[0] == 1, "Batch size must be 1 for evaluation."
    return body.squeeze(0)

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

    parser.add_argument('--text_path', type=str, required=True, default = '')
    parser.add_argument('--text_input_scaler', type=str, default=None)
    parser.add_argument('--groundtruth_path', type=str, required=True, default = '')
    parser.add_argument('--all_motion_feats_path', type=str, required=True, default = '')
    parser.add_argument('--save_dir', type=str, required=True, default = '')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--ds_rate', type=int, default=1)
    parser.add_argument('--diffusion_control_config_path', type=str, default='./configs/inference/controlnet.yaml')
    parser.add_argument('--diffusion_mix_config_path', type=str, default='./configs/legacy/triplet.yaml')
    parser.add_argument('--solver', type=str, default='DDIM')
    parser.add_argument('--compare_w_all', action='store_true', default=False)

    args = parser.parse_args()

    print(f"Calculating FID and Diversity. Compare with all? : {args.compare_w_all}")
    start_time = time.time()

    print("Preparing data...")
    text_name = []

    for text_file in os.listdir(args.text_path):
        if text_file.endswith('.txt'):
            text_name.append(os.path.splitext(text_file)[0])

    trainer_control = Controlnet_trainer(args.diffusion_control_config_path)
    trainer_mix = MixTask_trainer(args.diffusion_mix_config_path)

    text_input_scaler = args.text_input_scaler
    output_scaler_control = jl.load(os.path.join(trainer_control.config.data.data_root, 'output_scaler.sav'))
    data_pipe_control = jl.load(os.path.join(trainer_control.config.data.data_root, 'data_pipe_' + str(trainer_control.config.data.framerate) + 'fps.sav'))
    output_scaler_mix = jl.load(os.path.join(trainer_mix.config.data.data_root, 'output_scaler.sav'))
    data_pipe_mix = jl.load(os.path.join(trainer_mix.config.data.data_root, 'data_pipe_' + str(trainer_mix.config.data.framerate) + 'fps.sav'))

    print(f"Time taken for initialization: {time.time() - start_time:.2f} seconds")

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu', args.local_rank)
    text_encoder = text_encoder.to(device)

    p = BVHParser()
    groundtruth_feats_all = list()
    groundtruth_all = list()
    controlnet_all = list()
    mixnet_all = list()

    if args.compare_w_all:
        for file in os.listdir(args.all_motion_feats_path):
            if file.endswith('.npz') and 'mirrored' not in file:
                data = np.load(os.path.join(args.all_motion_feats_path, file))["clips"]
                print(f"Loaded groundtruth features from {file}, shape: {data.shape}")
                groundtruth_feats_all.append(data)

    for text_i in range(len(text_name)):
        new_save_dir = os.path.join(args.save_dir, text_name[text_i])
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(new_save_dir):
            os.mkdir(new_save_dir)

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
            if text_input_scaler is not None:
                text_embeddings = standardize(text_embeddings.cpu().numpy(), text_input_scaler)
            else:
                text_embeddings = text_embeddings.cpu().numpy()
        np.save(os.path.join(new_save_dir, "text_features.npy"), text_embeddings)

        print(f"Time taken for data preparation: {time.time() - start_time:.2f} seconds")

        data_control = sequence_generate_diffusion(data_dir=new_save_dir,
                                            trainer=trainer_control,
                                            solver=args.solver,
                                            sample_steps=trainer_control.config.inference.sample_steps,
                                            guidance_weight=trainer_control.config.inference.guidance_weight,
                                            eta=trainer_control.config.inference.eta,
                                            )
        data_mix = sequence_generate_diffusion(data_dir=new_save_dir,
                                            trainer=trainer_mix,
                                            solver=args.solver,
                                            sample_steps=trainer_mix.config.inference.sample_steps,
                                            guidance_weight=trainer_mix.config.inference.guidance_weight,
                                            eta=trainer_mix.config.inference.eta,
                                            )
        print(f"Generated data shapes - Controlnet: {data_control.shape}, Mixnet: {data_mix.shape}")

        end_time = time.time()
        print(f"Time taken for generation: {end_time - start_time:.2f} seconds, start saving...")

        gt_bvh_path = os.path.join(args.groundtruth_path, text_name[text_i] + '.bvh')
        gt_bvh = list()
        gt_bvh.append(p.parse(gt_bvh_path))
        out_data = data_pipe_control.fit_transform(gt_bvh)
        out_data = out_data[0:1]
        print(f"Groundtruth data shape: {out_data.shape}")
        groundtruth_all.append(out_data)

        data_control = data_control.cpu().numpy()
        data_mix = data_mix.cpu().numpy()

        data_control = np.expand_dims(data_control, axis=0)
        data_control = inv_standardize(data_control, output_scaler_control)
        bvh_save_path_control = os.path.join(new_save_dir, text_name[text_i] + '_controlnet.bvh')
        to_bvh(data_control, bvh_save_path_control, data_pipe_control, trainer_control.config.data.framerate)
        control_bvh = list()
        control_bvh.append(p.parse(bvh_save_path_control))
        data_control = data_pipe_control.fit_transform(control_bvh)
        data_control = data_control[0:1]

        data_mix = np.expand_dims(data_mix, axis=0)
        # data_mix = inv_standardize(data_mix, output_scaler_control)
        data_mix = inv_standardize(data_mix, output_scaler_mix)
        bvh_save_path_mix = os.path.join(new_save_dir, text_name[text_i] + '_mixnet.bvh')
        # to_bvh(data_mix, bvh_save_path_mix, data_pipe_control, trainer_mix.config.data.framerate)
        to_bvh(data_mix, bvh_save_path_mix, data_pipe_mix, trainer_mix.config.data.framerate)
        mix_bvh = list()
        mix_bvh.append(p.parse(bvh_save_path_mix))
        # data_mix = data_pipe_control.fit_transform(mix_bvh)
        data_mix = data_pipe_mix.fit_transform(mix_bvh)
        data_mix = data_mix[0:1]

        shutil.rmtree(new_save_dir)

        controlnet_all.append(data_control)
        mixnet_all.append(data_mix)

    # calculate fid and diversity
    print("\n" + "="*50)
    print("Starting Evaluation (FID & Diversity)...")

    # 1. Extract Features
    print("Extracting features (Placeholder: Temporal Mean)...")
    gt_feats = extract_features(groundtruth_all)
    control_feats = extract_features(controlnet_all)
    mix_feats = extract_features(mixnet_all)

    if args.compare_w_all:
        gt_feats_all = extract_features(groundtruth_feats_all)
        print(f"Feature shapes -> GT_ALL: {gt_feats_all.shape}, GT: {gt_feats.shape}, Control: {control_feats.shape}, Mix: {mix_feats.shape}")
    else:
        print(f"Feature shapes -> GT: {gt_feats.shape}, Control: {control_feats.shape}, Mix: {mix_feats.shape}")

    # 2. Calculate Statistics
    if args.compare_w_all:
        gt_all_mu, gt_all_sigma = get_distribution_statistics(gt_feats_all)
    gt_mu, gt_sigma = get_distribution_statistics(gt_feats)
    control_mu, control_sigma = get_distribution_statistics(control_feats)
    mix_mu, mix_sigma = get_distribution_statistics(mix_feats)

    # 3. Calculate FID
    if args.compare_w_all:
        fid_control = calculate_frechet_distance(gt_all_mu, gt_all_sigma, control_mu, control_sigma)
        fid_mix = calculate_frechet_distance(gt_all_mu, gt_all_sigma, mix_mu, mix_sigma)
        fid_gt = calculate_frechet_distance(gt_all_mu, gt_all_sigma, gt_mu, gt_sigma)
    # else:
        # fid_control = calculate_frechet_distance(gt_mu, gt_sigma, control_mu, control_sigma)
        # fid_mix = calculate_frechet_distance(gt_mu, gt_sigma, mix_mu, mix_sigma)

    fid_control_tiny = calculate_frechet_distance(gt_mu, gt_sigma, control_mu, control_sigma)
    fid_mix_tiny = calculate_frechet_distance(gt_mu, gt_sigma, mix_mu, mix_sigma)

    # 4. Calculate Diversity
    div_gt = calculate_diversity(gt_feats)
    div_control = calculate_diversity(control_feats)
    div_mix = calculate_diversity(mix_feats)

    print("-" * 30)
    print(f"{'Metric':<20} | {'ControlNet':<12} | {'MixNet':<12} | {'GroundTruth':<12}")
    print("-" * 30)
    if args.compare_w_all:
        print(f"{'FID (vs GT)':<20} | {fid_control:.4f}       | {fid_mix:.4f}       | {fid_gt:.4f}")
        print(f"{'FID-Tiny (vs GT)':<20} | {fid_control_tiny:.4f}  | {fid_mix_tiny:.4f}  | {'N/A':<12}")
    else:
        print(f"{'FID (vs GT)':<20} | {fid_control:.4f}       | {fid_mix:.4f}       | {'N/A':<12}")
    print(f"{'Diversity':<20} | {div_control:.4f}       | {div_mix:.4f}       | {div_gt:.4f}")
    print("-" * 30)
    print("Done.")
