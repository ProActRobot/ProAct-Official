"""
Streaming Audio Feature Extractor

Solves the online/offline feature quality discrepancy by:
1. Maintaining a rolling audio buffer (3-5 seconds context)
2. Extracting features from the full buffer each time
3. Returning only the features corresponding to the new audio chunk

This ensures consistent FFT window behavior and proper context for:
- MFCC delta computation
- Tempogram calculation
- Chroma features

Usage:
    extractor = StreamingAudioFeatureExtractor(sr=15360, fps=30, buffer_seconds=3.0)

    # For each new audio chunk:
    new_features = extractor.process_chunk(audio_chunk)  # Returns [new_T, 181]

    # Reset when session ends:
    extractor.reset()

Author: Auto-generated for Social Robot Baseline
"""

import numpy as np
import librosa
from typing import Optional, Tuple
import warnings


class AudioFeatureComputer:
    """Static methods for computing audio features - matches original AudioFeature class."""

    @staticmethod
    def get_melspectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute mel spectrogram in dB scale."""
        melspe = librosa.feature.melspectrogram(y=audio, sr=sr)
        melspe_db = librosa.power_to_db(melspe, ref=np.max)
        return melspe_db

    @staticmethod
    def get_mfcc(melspe_db: np.ndarray) -> np.ndarray:
        """Compute MFCC from mel spectrogram."""
        return librosa.feature.mfcc(S=melspe_db)

    @staticmethod
    def get_mfcc_delta(mfcc: np.ndarray) -> np.ndarray:
        """Compute MFCC delta (first derivative)."""
        return librosa.feature.delta(mfcc, width=3)

    @staticmethod
    def get_hpss(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Harmonic-Percussive Source Separation."""
        return librosa.effects.hpss(audio)

    @staticmethod
    def get_chroma_cqt(audio_harmonic: np.ndarray, sr: int, n_octaves: int = 5) -> np.ndarray:
        """Compute chroma CQT from harmonic component."""
        return librosa.feature.chroma_cqt(y=audio_harmonic, sr=sr, n_octaves=n_octaves)

    @staticmethod
    def get_onset_strength(audio_percussive: np.ndarray, sr: int) -> np.ndarray:
        """Compute onset strength envelope."""
        return librosa.onset.onset_strength(y=audio_percussive, aggregate=np.median, sr=sr)

    @staticmethod
    def get_tempogram(onset_env: np.ndarray, sr: int) -> np.ndarray:
        """Compute tempogram, truncated to 128 bins."""
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        return tempogram[:128, :]


class StreamingAudioFeatureExtractor:
    """
    Streaming audio feature extractor that maintains context buffer.

    Key insight:
    - Offline: entire audio processed at once → proper context for all frames
    - Online (naive): each chunk processed independently → edge effects, short windows
    - Online (streaming): accumulate audio buffer → extract from buffer → return new frames

    This class implements the streaming approach to match offline quality.

    Enhanced with overlap feature support:
    - Returns overlap frames (e.g., last 1 second) along with new frames
    - Caller can use overlap frames to replace cached feature tails
    - This ensures consistent feature alignment at boundaries
    """

    def __init__(
        self,
        sr: int = 15360,
        fps: int = 30,
        buffer_seconds: float = 5.0,
        hop_length: int = 512,
        n_octaves: int = 5,
        overlap_seconds: float = 1.0,
        mel_ref_mode: str = 'fixed',  # 'running_max', 'fixed', or 'per_buffer'
    ):
        """
        Initialize streaming feature extractor.

        Args:
            sr: Target sample rate (should match training data, typically 15360)
            fps: Frame rate for output features (typically 30)
            buffer_seconds: How many seconds of audio context to maintain
            hop_length: Hop length for STFT (default 512 matches librosa default)
            n_octaves: Number of octaves for chroma_cqt
            overlap_seconds: How many seconds of overlap features to return (for cache replacement)
            mel_ref_mode: Mode for mel spectrogram normalization:
                - 'running_max': Use session running max (recommended, most stable)
                - 'fixed': Use fixed ref=1.0 (absolute dB values, may differ from training)
                - 'per_buffer': Use np.max of current buffer (original behavior, may cause jumps)
        """
        self.sr = sr
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.overlap_seconds = overlap_seconds
        self.mel_ref_mode = mel_ref_mode

        # Calculate overlap frames based on fps
        self.overlap_frames = int(overlap_seconds * fps)

        # Calculate buffer size in samples
        self.buffer_max_samples = int(buffer_seconds * sr)

        # Audio buffer - accumulates over time
        self.audio_buffer = np.empty(0, dtype=np.float32)

        # Track how many feature frames we've already emitted
        self.emitted_frames = 0

        # Minimum samples needed before extraction (at least 1 frame + context)
        # hop_length * 2 ensures at least some overlap
        self.min_samples_for_extraction = hop_length * 2

        # Session running max for mel spectrogram normalization (used when mel_ref_mode='running_max')
        # This ensures consistent ref value throughout the session,
        # even when buffer is trimmed. Mimics batch processing behavior.
        self.session_mel_ref: Optional[float] = None

    def reset(self) -> None:
        """Reset buffer for new session/conversation."""
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.emitted_frames = 0
        self.session_mel_ref = None  # Reset ref for new session

    def _compute_features_full(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute full 181-dim features for audio.

        Feature breakdown (matches original _compute_channel_feats):
        - MFCC: 20
        - MFCC Delta: 20
        - Chroma CQT: 12
        - Onset Envelope: 1
        - Tempogram: 128 (truncated to first 128 bins)
        Total: 181 ✓

        Returns:
            Features array of shape [T, 181] where T = num frames
        """
        with warnings.catch_warnings():
            # Suppress the n_fft warnings for edge frames
            warnings.filterwarnings('ignore', message='.*n_fft.*')
            warnings.filterwarnings('ignore', message='.*empty frequency.*')

            # 1. Mel spectrogram → MFCC
            melspe = librosa.feature.melspectrogram(y=audio, sr=self.sr)

            # Choose ref based on mel_ref_mode
            if self.mel_ref_mode == 'fixed':
                # Use fixed ref=1.0 for absolute dB values
                mel_ref = -500
            elif self.mel_ref_mode == 'per_buffer':
                # Use current buffer's max (original librosa behavior, may cause jumps)
                mel_ref = np.max(melspe)
            else:  # 'running_max' (default)
                # Use session running max for consistent normalization
                current_max = np.max(melspe)
                if self.session_mel_ref is None:
                    self.session_mel_ref = current_max
                else:
                    self.session_mel_ref = max(self.session_mel_ref, current_max)
                mel_ref = self.session_mel_ref

            melspe_db = librosa.power_to_db(melspe, ref=mel_ref)

            mfcc = AudioFeatureComputer.get_mfcc(melspe_db)          # [20, T]
            mfcc_delta = AudioFeatureComputer.get_mfcc_delta(mfcc)   # [20, T]

            # 2. HPSS → Chroma + Onset + Tempogram
            audio_harmonic, audio_percussive = AudioFeatureComputer.get_hpss(audio)
            chroma_cqt = AudioFeatureComputer.get_chroma_cqt(
                audio_harmonic, self.sr, n_octaves=self.n_octaves
            )  # [12, T]

            onset_env = AudioFeatureComputer.get_onset_strength(audio_percussive, self.sr)  # [T]

            # Tempogram - use 128 bins to match original AudioFeature.get_tempogram
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sr)
            tempogram = tempogram[:128, :]  # Truncate to 128 bins

            # Pad tempogram if needed (for very short audio)
            if tempogram.shape[0] < 128:
                pad_size = 128 - tempogram.shape[0]
                tempogram = np.pad(tempogram, ((0, pad_size), (0, 0)), mode='constant')

            # 3. Concatenate all features
            # All should have shape [F, T], we want [T, F]
            T = mfcc.shape[1]

            # Ensure all features have same T dimension
            def align_to_T(feat: np.ndarray, target_T: int) -> np.ndarray:
                if feat.ndim == 1:
                    feat = feat.reshape(1, -1)
                if feat.shape[1] < target_T:
                    # Pad with last value
                    pad_size = target_T - feat.shape[1]
                    feat = np.pad(feat, ((0, 0), (0, pad_size)), mode='edge')
                elif feat.shape[1] > target_T:
                    feat = feat[:, :target_T]
                return feat

            mfcc = align_to_T(mfcc, T)
            mfcc_delta = align_to_T(mfcc_delta, T)
            chroma_cqt = align_to_T(chroma_cqt, T)
            onset_env_2d = align_to_T(onset_env, T)
            tempogram = align_to_T(tempogram, T)

            # Stack: [20 + 20 + 12 + 1 + 128, T] = [181, T]
            features = np.concatenate([
                mfcc,           # 20
                mfcc_delta,     # 20
                chroma_cqt,     # 12
                onset_env_2d,   # 1
                tempogram,      # 128
            ], axis=0)

            # Transpose to [T, 181]
            return features.T.astype(np.float32)

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        return_overlap: bool = False,
        drop_last_frame: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Process a new audio chunk and return features for the NEW frames only.

        This is the key streaming method:
        1. Append new audio to buffer
        2. Extract features from entire buffer
        3. Return only the features we haven't emitted yet (optionally drop last frame)
        4. Trim buffer if it exceeds max size

        Args:
            audio_chunk: New audio samples, shape [N] at self.sr sample rate
            return_overlap: If True, return (new_features, overlap_features) tuple
                           where overlap_features are the last `overlap_frames` frames
                           from the previous emission (for cache replacement)
            drop_last_frame: If True, drop the last frame from new_features to avoid
                            boundary effects. This keeps the last frame in buffer for
                            re-computation with more context in next chunk.
                            - For 1-second chunks: returns 30 frames instead of 31
                            - Recommended to set False for the final chunk

        Returns:
            If return_overlap=False:
                Features for new frames: [new_T, 181] or None if not enough audio
            If return_overlap=True:
                Tuple of (new_features, overlap_features) where:
                - new_features: [new_T, 181] features for new frames
                - overlap_features: [overlap_frames, 181] features to replace cached tail
                                   (None if this is the first chunk)

        Example with drop_last_frame=True (1-second chunks, overlap_frames=30):
            Call 1 (seg1): cache=[1], 31 frames -> emit [0:30], drop frame 30
                          emitted_frames=30, overlap=None
            Call 2 (seg2): cache=[1,2], 61 frames -> emit [30:60], drop frame 60
                          emitted_frames=60, overlap=[0:30] (recomputed seg1)
            Call 3 (seg3): cache=[2,3], 61 frames -> emit [30:60], drop frame 60
                          emitted_frames=60, overlap=[0:30] (recomputed seg2)
            Call 4 (seg4): cache=[3,4], 61 frames -> emit [30:60], drop frame 60
                          emitted_frames=60, overlap=[0:30] (recomputed seg3)
        """
        if audio_chunk.size == 0:
            return None if not return_overlap else (None, None)

        # Ensure audio is float32 and 1D
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=-1)

        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Check if we have enough audio
        if len(self.audio_buffer) < self.min_samples_for_extraction:
            return None if not return_overlap else (None, None)

        # Extract features from full buffer
        all_features = self._compute_features_full(self.audio_buffer)
        total_frames = all_features.shape[0]

        # Calculate how many new frames we have
        new_frames_count = total_frames - self.emitted_frames

        if new_frames_count <= 0:
            # No new frames yet
            return None if not return_overlap else (None, None)

        # Extract overlap features (from the previous emission's tail, re-computed with current buffer)
        # These are the frames BEFORE the new frames, recomputed with more context
        # Caller can use these to replace their cached features for better accuracy
        overlap_features = None
        if return_overlap and self.emitted_frames > 0:
            # overlap_start/end are based on the PREVIOUS emitted_frames position
            # This gives us the last `overlap_frames` from what was emitted before
            overlap_start = max(0, self.emitted_frames - self.overlap_frames)
            overlap_end = self.emitted_frames
            if overlap_end > overlap_start:
                overlap_features = all_features[overlap_start:overlap_end, :]

        # Determine frame range for extraction
        start_frame = self.emitted_frames  # = total_frames - new_frames_count

        # If drop_last_frame is True and we have more than 1 new frame,
        # keep the last frame for re-computation in next chunk
        if drop_last_frame and new_frames_count > 1:
            end_frame = total_frames - 1  # Exclude the boundary frame
            self.emitted_frames = end_frame  # Don't mark the last frame as emitted
        else:
            end_frame = total_frames
            self.emitted_frames = total_frames

        # Extract the new frames
        new_features = all_features[start_frame:end_frame, :]

        # Trim buffer if too long (keep enough context for next extraction)
        if len(self.audio_buffer) > self.buffer_max_samples:
            # Keep the last buffer_max_samples
            trim_samples = len(self.audio_buffer) - self.buffer_max_samples
            self.audio_buffer = self.audio_buffer[trim_samples:]

            # Adjust emitted_frames accordingly
            # We need to figure out how many frames were in the trimmed audio
            frames_in_trimmed = trim_samples // self.hop_length
            self.emitted_frames = max(0, self.emitted_frames - frames_in_trimmed)

        if return_overlap:
            return new_features, overlap_features
        return new_features

    def process_file(self, wav_path: str) -> np.ndarray:
        """
        Process entire audio file (for offline/batch mode compatibility).

        Args:
            wav_path: Path to WAV file

        Returns:
            Features: [T, 181]
        """
        # Reset state
        self.reset()

        # Load audio at target sample rate
        audio, _ = librosa.load(wav_path, sr=self.sr)

        # Process as single chunk
        features = self.process_chunk(audio)

        return features if features is not None else np.empty((0, 181), dtype=np.float32)

    def get_buffer_duration(self) -> float:
        """Return current buffer duration in seconds."""
        return len(self.audio_buffer) / self.sr if self.sr > 0 else 0.0

    def get_buffer_frames(self) -> int:
        """Return approximate number of frames in buffer."""
        return self.emitted_frames


# Convenience function matching original interface
def compute_channel_feats_streaming(
    extractor: StreamingAudioFeatureExtractor,
    audio_chunk: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Drop-in replacement for _compute_channel_feats that uses streaming.

    Args:
        extractor: Initialized StreamingAudioFeatureExtractor
        audio_chunk: Audio samples at extractor's sample rate

    Returns:
        Features [T, 181] or None
    """
    return extractor.process_chunk(audio_chunk)
