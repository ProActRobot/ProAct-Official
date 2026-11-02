# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
from typing import Dict, Tuple, Optional
from shutil import copyfile
import joblib as jl
from omegaconf import OmegaConf
import torch
from transformers import CLIPTextModel, CLIPTokenizer


from data_process.extract_features import extract_joint_angles
from data_process.utils.utils import fit_and_standardize, standardize

_clip_tokenizer = None
_clip_text_encoder = None
_clip_model_path = None


def _get_clip_models(clip_dir: str):
    """Load CLIP only when SeG text features are being generated."""
    global _clip_tokenizer, _clip_text_encoder, _clip_model_path
    if _clip_tokenizer is None or _clip_text_encoder is None or _clip_model_path != clip_dir:
        _clip_tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
        _clip_text_encoder = CLIPTextModel.from_pretrained(clip_dir)
        for parameter in _clip_text_encoder.parameters():
            parameter.requires_grad = False
        _clip_text_encoder.eval()
        _clip_model_path = clip_dir
    return _clip_tokenizer, _clip_text_encoder

# -----------------------------

# -----------------------------
def _sanitize_action_name(s: str) -> str:

    return re.sub(r"[-\s]+", "_", s.strip()).upper()


def _sample_id_from_filename(fname: str) -> str:

    base = os.path.basename(fname)
    return os.path.splitext(base)[0]


def load_sample_texts(text_json_path: str) -> Dict[str, Dict[str, str]]:
    with open(text_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    sample_to_texts: Dict[str, Dict[str, str]] = {}
    if not isinstance(items, list):
        raise ValueError("SeG annotations must be a list of per-clip records")
    for item in items:
        sample_id = _sample_id_from_filename(item.get("file_name", ""))
        if not sample_id:
            raise ValueError("Every SeG annotation needs a nonempty file_name")
        texts = {
            "intent": item.get("annotation_intent", "").strip(),
            "pose": item.get("annotation_structure_pose", "").strip(),
            "motion": item.get("annotation_structure_motion", "").strip(),
        }
        if not all(texts.values()):
            raise ValueError(f"Missing intent, pose or motion annotation for {sample_id}")
        if sample_id in sample_to_texts and sample_to_texts[sample_id] != texts:
            raise ValueError(f"Conflicting annotations for clip {sample_id}")
        sample_to_texts[sample_id] = texts
    return sample_to_texts


@torch.no_grad()
def encode_texts_with_clip(clip_dir: str, texts: Dict[str, str]) -> np.ndarray:
    """Encode the three SeG annotation sentences with the configured CLIP model."""
    tokenizer, text_encoder = _get_clip_models(clip_dir)
    ordered_keys = ["intent", "pose", "motion"]
    feats = []
    for k in ordered_keys:
        t = texts.get(k, "")
        print("text:", t)
        inputs = tokenizer(t, return_tensors="pt", padding='max_length', truncation=True, max_length=77)
        out = text_encoder(**inputs).last_hidden_state.squeeze(0)  # [77, H]
        feats.append(out.cpu().numpy())

    return np.stack(feats, axis=0).astype(np.float32)  # [3, 77, H]


def build_sample_text_features(text_json_path: str, clip_dir: str, save_dir: str) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    sample_to_texts = load_sample_texts(text_json_path)
    sample_to_npz_path: Dict[str, str] = {}
    names = ["intent", "pose", "motion"]
    for sample_id, texts in sample_to_texts.items():
        feats = encode_texts_with_clip(clip_dir, texts)
        npz_path = os.path.join(save_dir, f"{sample_id}.npz")
        np.savez(npz_path, features=feats, names=np.array(names),
                 texts=np.array([texts[name] for name in names]), sample_id=np.array(sample_id))
        sample_to_npz_path[sample_id] = npz_path
    print(f"[TEXT] Built CLIP text features for {len(sample_to_npz_path)} clips at: {save_dir}")
    return sample_to_npz_path


def slice_motion_and_text(
    motion: np.ndarray,          # [T, Dm]
    text3: np.ndarray,
    fps: int,
    window_size: int,
    overlap: float
) -> Tuple[np.ndarray, np.ndarray]:
    assert text3.ndim == 3 and text3.shape[0] == 3, "text3 must be [3, 77, H]"
    nframes = motion.shape[0]
    overlap_frames = int(overlap * window_size)
    step = window_size - overlap_frames
    if step <= 0:
        raise ValueError("window_size * (1 - overlap) must be > 0")

    n_sequences = (nframes - overlap_frames) // step
    if n_sequences <= 0:
        print("[WARN] data too small for window")
        return (np.zeros((3, 0, window_size, motion.shape[1]), dtype=np.float32),
                np.zeros((3, 0, text3.shape[1], text3.shape[2]), dtype=np.float32))

    Dm = motion.shape[1]
    T_text, D_text = text3.shape[1], text3.shape[2]  # 77, H

    sliced_motion = np.zeros((3, n_sequences, window_size, Dm), dtype=np.float32)
    sliced_text   = np.zeros((3, n_sequences, T_text, D_text), dtype=np.float32)

    for i in range(n_sequences):
        base = step * i
        mseg = motion[base:base + window_size, :]  # [window_size, Dm]
        for v in range(3):
            sliced_motion[v, i, :, :] = mseg

            sliced_text[v, i, :, :] = text3[v]

    return sliced_motion, sliced_text


def _scan_bvh_files(bvh_dir: str) -> Tuple[list, Dict[str, Tuple[str, str]]]:
    basenames = []
    meta = {}
    for r, _, files in os.walk(bvh_dir):
        for fn in sorted(files):
            if fn.lower().endswith(".bvh"):
                base = os.path.splitext(fn)[0]

                if "-" not in base:
                    print(f"[WARN] Unexpected bvh name (no '-'): {fn}")
                    continue
                action = base.split("-", 1)[0]
                sample = base.split("-", 1)[1]
                basenames.append(base)
                meta[base] = (action, sample)
    return basenames, meta


def _split_train_test(
    basenames: list,
    testset: list
) -> Tuple[list, list]:
    """Split motion files using the configured held-out basename list."""
    test = [b for b in basenames if b in set(testset)]
    train = [b for b in basenames if b not in set(testset)]
    return train, test


# -----------------------------

# -----------------------------
def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)

    data_root = cfg.data_dir
    bvh_dir   = os.path.join(data_root, "bvh")
    text_json = os.path.join(data_root, "text.json")

    processed_dir = cfg.processed_dir
    fps = cfg.fps
    fullbody = cfg.get("fullbody", True)
    map_to_exponential = cfg.get("map_to_exponential", False)
    position_smoothing = cfg.get("position_smoothing", 0.0)
    rotation_smoothing = cfg.get("rotation_smoothing", 0.0)
    constant_remover_eps = cfg.get("constant_remover_eps", 0.0)

    train_window_secs = cfg.train_window_secs
    test_window_secs  = cfg.test_window_secs
    window_overlap    = cfg.window_overlap
    need_mirror_data  = cfg.get("need_mirror_data", False)
    is_standardize    = cfg.get("is_standardize", True)

    # CLIP
    clip_local_path = cfg.get("clip_local_path", os.getenv("MOTION_CLIP_MODEL", "openai/clip-vit-base-patch32"))


    testset = cfg.get("testset", [])


    path = os.path.join(
        processed_dir,
        f"feat_{fps}fps_{train_window_secs}s_TEXT_{'WithStd' if is_standardize else 'NotStd'}"
    )
    motion_path = os.path.join(path, "joint_rot")
    textfeat_path = os.path.join(path, "text", "by_clip")

    os.makedirs(path, exist_ok=True)
    os.makedirs(motion_path, exist_ok=True)
    os.makedirs(textfeat_path, exist_ok=True)


    basenames, meta = _scan_bvh_files(bvh_dir)
    assert len(basenames) > 0, f"No BVH files found under {bvh_dir}"



    if not any(fn.endswith(".npz") for fn in os.listdir(motion_path)) or cfg.get("force_reextract_motion", False):
        extract_joint_angles(
            bvh_dir,
            basenames,
            motion_path,
            fps,
            fullbody=fullbody,
            map_to_exponential=map_to_exponential,
            position_smoothing=position_smoothing,
            rotation_smoothing=rotation_smoothing,
            constant_remover_eps=constant_remover_eps
        )

        if os.path.exists(os.path.join(motion_path, "data_pipe.sav")):
            copyfile(os.path.join(motion_path, "data_pipe.sav"), os.path.join(path, f"data_pipe_{fps}fps.sav"))
    else:
        if os.path.exists(os.path.join(motion_path, "data_pipe.sav")):
            copyfile(os.path.join(motion_path, "data_pipe.sav"), os.path.join(path, f"data_pipe_{fps}fps.sav"))


    sample_to_npz = build_sample_text_features(text_json, clip_local_path, textfeat_path)


    files: Dict[str, Dict[str, str]] = {}
    missing_annotations = []
    for base in basenames:
        action, _sample = meta[base]
        action_std = _sanitize_action_name(action)
        motion_npz = os.path.join(motion_path, base)
        text_npz   = sample_to_npz.get(base)
        if text_npz is None:
            missing_annotations.append(base)
            continue
        files[base] = {"motion_npz": motion_npz, "text_npz": text_npz, "action": action_std}
    if missing_annotations:
        print(f"[WARN] Skipped {len(missing_annotations)} motions without per-clip annotations: "
              + ", ".join(missing_annotations[:10]))
    if not files:
        raise ValueError("No motion files have matching per-clip annotations")
    train_list, test_list = _split_train_test(list(files.keys()), testset)
    print(f"[SPLIT] Train: {len(train_list)}, Test: {len(test_list)}")


    slice_win_train = int(train_window_secs * fps)
    slice_win_test  = int(test_window_secs  * fps)
    val_test_split_frames = int(20 * test_window_secs * fps)

    def import_and_slice(basenames_subset: list, slice_win: int, overlap: float, start: int = 0, end: Optional[int] = None):
        out_motion = None
        out_text   = None
        first = True
        for base in basenames_subset:
            entry = files[base]

            suffix = ".npz"
            if need_mirror_data:

                raise NotImplementedError("Mirror data handling is done later; this path should not be used here.")
            motion = np.load(entry["motion_npz"] + suffix)["clips"].astype(np.float32)  # [T, Dm]
            if end is None:
                end_idx = motion.shape[0]
            else:
                end_idx = min(end, motion.shape[0])
            motion = motion[start:end_idx, :]


            tpack = np.load(entry["text_npz"])
            text3 = tpack["features"].astype(np.float32)  # [3, Dt]


            sm, st = slice_motion_and_text(motion, text3, fps, slice_win, overlap)  # [3, n_seq, T, Dm], [3, n_seq, T, Dt]


            if need_mirror_data and os.path.exists(entry["motion_npz"] + "_mirrored.npz"):
                motion_m = np.load(entry["motion_npz"] + "_mirrored.npz")["clips"].astype(np.float32)
                motion_m = motion_m[start:end_idx, :]
                sm_m, st_m = slice_motion_and_text(motion_m, text3, fps, slice_win, overlap)
                sm = np.concatenate([sm, sm_m], axis=1)
                st = np.concatenate([st, st_m], axis=1)

            if first:
                out_motion, out_text = sm, st
                first = False
            else:
                out_motion = np.concatenate([out_motion, sm], axis=1)
                out_text   = np.concatenate([out_text,   st], axis=1)

        if first:

            Dm = 0
            Dt = 0
            return (np.zeros((3, 0, slice_win, Dm), dtype=np.float32),
                    np.zeros((3, 0, slice_win, Dt), dtype=np.float32))
        return out_motion, out_text


    train_motion, train_text = import_and_slice(train_list, slice_win_train, window_overlap)


    val_motion, val_text = import_and_slice(test_list, slice_win_train, window_overlap, start=0, end=val_test_split_frames)


    if is_standardize:
        print("[STD] Standardizing motion & text features...")

        # motion: [3, N, T, Dm]
        a, b, c, d = train_motion.shape
        train_motion_r = train_motion.reshape(a * b, c, d)
        train_motion_r, motion_scaler = fit_and_standardize(train_motion_r)
        train_motion = train_motion_r.reshape(a, b, c, d)

        a, b, c, d = val_motion.shape
        val_motion_r = val_motion.reshape(a * b, c, d)
        val_motion_r = standardize(val_motion_r, motion_scaler)
        val_motion = val_motion_r.reshape(a, b, c, d)

        # text: [3, N, T, Dt]
        a, b, c, d = train_text.shape
        train_text_r = train_text.reshape(a * b, c, d)
        train_text_r, text_scaler = fit_and_standardize(train_text_r)
        train_text = train_text_r.reshape(a, b, c, d)

        a, b, c, d = val_text.shape
        val_text_r = val_text.reshape(a * b, c, d)
        val_text_r = standardize(val_text_r, text_scaler)
        val_text = val_text_r.reshape(a, b, c, d)

        jl.dump(motion_scaler, os.path.join(path, "output_scaler.sav"))
        jl.dump(text_scaler,   os.path.join(path, "input_scaler.sav"))
    else:
        print("[STD] Skip standardizing.")


    print("[SAVE] Save train/val npz...")
    np.savez(os.path.join(path, f"train_output_{fps}fps.npz"), clips=train_motion)  # motion
    np.savez(os.path.join(path, f"train_input_{fps}fps.npz"),  clips=train_text)    # text
    np.savez(os.path.join(path, f"val_output_{fps}fps.npz"),   clips=val_motion)
    np.savez(os.path.join(path, f"val_input_{fps}fps.npz"),    clips=val_text)


    print("[SAVE] Save per-test samples...")
    for base in test_list:
        entry = files[base]

        motion = np.load(entry["motion_npz"] + ".npz")["clips"].astype(np.float32)
        motion = motion[:slice_win_test, :]
        tpack = np.load(entry["text_npz"])
        text3 = tpack["features"].astype(np.float32)

        sm, st = slice_motion_and_text(motion, text3, fps, slice_win_test, overlap=0.0)  # [3, 1, T, D*]

        if is_standardize:
            a, b, c, d = sm.shape
            sm_r = sm.reshape(a * b, c, d)
            sm_r = standardize(sm_r, motion_scaler)
            sm   = sm_r.reshape(a, b, c, d)

            a, b, c, d = st.shape
            st_r = st.reshape(a * b, c, d)
            st_r = standardize(st_r, text_scaler)
            st   = st_r.reshape(a, b, c, d)

        np.savez(os.path.join(path, f"test_output_{base}_{fps}fps.npz"), clips=sm)
        np.savez(os.path.join(path, f"test_input_{base}_{fps}fps.npz"),  clips=st)

    print("[DONE] BVH-TEXT preprocessing completed.")
    print(f"Output dir: {path}")


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/preprocessing/seg.yaml"
    main(cfg_path)
