"""Keep per-clip annotations attached to their motion through preprocessing."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src/motion"))
    specification = importlib.util.spec_from_file_location(
        "proact_seg_preparation", ROOT / "src/motion/data_process/prepare_seg_data.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def annotation(sample_id, number):
    return {
        "label": "WAVE",
        "file_name": sample_id + ".bvh",
        "annotation_intent": str(number),
        "annotation_structure_pose": str(number + 10),
        "annotation_structure_motion": str(number + 20),
    }


def test_same_category_retains_each_clips_three_annotations(tmp_path, preparation):
    source = tmp_path / "text.json"
    source.write_text(json.dumps([annotation("WAVE-1", 1), annotation("WAVE-2", 2)]))
    texts = preparation.load_sample_texts(str(source))
    assert texts["WAVE-1"] == {"intent": "1", "pose": "11", "motion": "21"}
    assert texts["WAVE-2"] == {"intent": "2", "pose": "12", "motion": "22"}


@pytest.mark.parametrize("problem", ["filename", "caption", "duplicate"])
def test_invalid_clip_annotations_are_rejected(tmp_path, preparation, problem):
    entries = [annotation("WAVE-1", 1)]
    if problem == "filename":
        entries[0]["file_name"] = ""
    elif problem == "caption":
        entries[0]["annotation_structure_motion"] = ""
    else:
        entries.append(annotation("WAVE-1", 2))
    source = tmp_path / "text.json"
    source.write_text(json.dumps(entries))
    with pytest.raises(ValueError):
        preparation.load_sample_texts(str(source))


def test_dataset_outputs_match_clip_annotations_without_category_fallback(
    tmp_path, preparation, monkeypatch
):
    raw = tmp_path / "raw"
    (raw / "bvh").mkdir(parents=True)
    processed = tmp_path / "processed"
    output = processed / "feat_2fps_2s_TEXT_NotStd"
    (output / "joint_rot").mkdir(parents=True)
    (output / "text").mkdir()
    for number in (1, 2, 3):
        sample_id = f"WAVE-{number}"
        (raw / "bvh" / (sample_id + ".bvh")).write_text("unused motion cache input")
        np.savez(output / "joint_rot" / (sample_id + ".npz"),
                 clips=np.full((88, 31), number, dtype=np.float32))
    (raw / "text.json").write_text(json.dumps([annotation("WAVE-1", 1), annotation("WAVE-2", 2)]))
    np.savez(output / "text/WAVE.npz", features=np.full((3, 77, 2), 99, dtype=np.float32))

    def encode(clip_dir, texts):
        return np.stack([
            np.full((77, 2), float(texts[name]), dtype=np.float32)
            for name in ("intent", "pose", "motion")
        ])

    monkeypatch.setattr(preparation, "encode_texts_with_clip", encode)
    configuration = tmp_path / "preparation.yaml"
    configuration.write_text(json.dumps({
        "data_dir": str(raw), "processed_dir": str(processed), "fps": 2,
        "train_window_secs": 2, "test_window_secs": 2, "window_overlap": 0.5,
        "is_standardize": False, "need_mirror_data": False, "testset": ["WAVE-2"],
    }))
    preparation.main(str(configuration))
    for filename, number in [
        ("train_input_2fps.npz", 1), ("val_input_2fps.npz", 2),
        ("test_input_WAVE-2_2fps.npz", 2),
    ]:
        with np.load(output / filename, allow_pickle=False) as saved:
            for variant, offset in enumerate((0, 10, 20)):
                assert saved["clips"][variant].size
                assert np.all(saved["clips"][variant] == number + offset)
    with np.load(output / "train_output_2fps.npz", allow_pickle=False) as saved:
        assert np.all(saved["clips"] == 1)
    with np.load(output / "text/by_clip/WAVE-2.npz", allow_pickle=False) as saved:
        assert saved["sample_id"].item() == "WAVE-2"
        assert saved["texts"].tolist() == ["2", "12", "22"]
