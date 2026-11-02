import csv
import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian_smooth_motion(data, sigma=1.0, truncate=4.0):
    if len(data) == 0:
        return data
    smoothed_data = np.zeros_like(data)
    for joint_idx in range(data.shape[1]):
        smoothed_data[:, joint_idx] = gaussian_filter1d(
            data[:, joint_idx], sigma=sigma, axis=0, truncate=truncate, mode="nearest"
        )
    return smoothed_data


def load_csv_motion(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            rows.append([float(x) for x in row])
    if not rows:
        return (header, np.array([]), np.array([]))
    data = np.array(rows)
    frame_indices = data[:, 0].astype(int)
    motion_data = data[:, 1:]
    return (header, motion_data, frame_indices)


def save_csv_motion(csv_file, header, motion_data, frame_indices):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, frame_idx in enumerate(frame_indices):
            row = [int(frame_idx)] + motion_data[i].tolist()
            writer.writerow(row)
