import numpy as np
import librosa
import soundfile as sf
import os

from .parsers import BVHParser
from .data import Joint, MocapData
from .preprocessing import *
from .writers import *


def cut_audio(filename, timespan, destpath, starttime=0.0, endtime=-1.0):
    print(f'Cutting AUDIO {filename} into intervals of {timespan}')
    X, fs = librosa.load(filename, sr=None)
    # fs, Y = wav.read(filename)
    if endtime <= 0:
        endtime = len(X) / fs
    suffix = 0
    while (starttime + timespan) <= endtime:
        out_basename = os.path.splitext(os.path.basename(filename))[0]
        # wav_outfile = out_basename + "_" + str(suffix).zfill(3) + '.wav'
        wav_outfile = os.path.join(destpath, out_basename + "_" + str(suffix).zfill(3) + '.wav')
        start_idx = int(np.round(starttime * fs))
        end_idx = int(np.round((starttime + timespan) * fs)) + 1
        if end_idx >= X.shape[0]:
            return

        sf.write(wav_outfile, X[start_idx:end_idx], samplerate=int(fs), subtype='PCM_24')
        # wav.write(wav_outfile, fs, X[start_idx:end_idx])
        starttime += timespan
        suffix += 1


def cut_bvh(filename, timespan, destpath, starttime=0.0, endtime=-1.0):
    print(f'Cutting BVH {filename} into intervals of {timespan}')

    p = BVHParser()
    bvh_data = p.parse(filename)
    if endtime <= 0:
        endtime = bvh_data.framerate * bvh_data.values.shape[0]

    writer = BVHWriter()
    suffix = 0
    while (starttime + timespan) <= endtime:
        out_basename = os.path.splitext(os.path.basename(filename))[0]
        bvh_outfile = os.path.join(destpath, out_basename + "_" + str(suffix).zfill(3) + '.bvh')
        start_idx = int(np.round(starttime / bvh_data.framerate))
        end_idx = int(np.round((starttime + timespan) / bvh_data.framerate)) + 1
        if end_idx >= bvh_data.values.shape[0]:
            return

        with open(bvh_outfile, 'w') as f:
            writer.write(bvh_data, f, start=start_idx, stop=end_idx)

        starttime += timespan
        suffix += 1
