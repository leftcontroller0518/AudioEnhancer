#!/usr/bin/env python3
"""
Audio enhancement pipeline with robustness fixes:
- safe_filtfilt to avoid short-signal ValueError
- proper multi-channel handling (any N channels)
- PhaseAlignment shift corrected
- DeReverb: ISTFT length padding/truncation
- DeEsser: envelope smoothing & divide-by-zero safety
- Dynamic/Noise gate: exponential attack/release behavior
- LoudnessNormalizer: use pyloudnorm if available, fallback to RMS
- Output clipping and dtype handling
"""
from typing import List, Tuple, Optional
import numpy as np
import scipy.signal as signal
import soundfile as sf
import warnings

# Try optional pyloudnorm for accurate LUFS measurement
try:
    import pyloudnorm as pyln
    _HAS_PYL = True
except Exception:
    _HAS_PYL = False


# ---------- Utilities ----------
def to_channels(audio: np.ndarray) -> List[np.ndarray]:
    """Return list of channel arrays (1D) regardless of mono/stereo/multi."""
    if audio.ndim == 1:
        return [audio]
    return [audio[:, i] for i in range(audio.shape[1])]


def from_channels(channels: List[np.ndarray]) -> np.ndarray:
    """Reconstruct ndarray from list of channels. If one channel, return 1D array."""
    if len(channels) == 1:
        return channels[0]
    return np.column_stack(channels)


def safe_filtfilt(b, a, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Use filtfilt when input long enough, otherwise fallback to lfilter.
    This avoids ValueError for short signals.
    """
    la = max(len(a), len(b))
    padlen = 3 * la
    n = x.shape[axis]
    if n <= padlen:
        # fallback; warn once
        return signal.lfilter(b, a, x, axis=axis)
    return signal.filtfilt(b, a, x, axis=axis)


def stable_stft_istft_reconstruct(Zxx, original_len: int, fs: int, nperseg: int = 2048) -> np.ndarray:
    """
    ISTFT with length correction: pad or trim to original_len.
    """
    _, x_rec = signal.istft(Zxx, fs=fs, nperseg=nperseg)
    if len(x_rec) < original_len:
        x_rec = np.pad(x_rec, (0, original_len - len(x_rec)))
    else:
        x_rec = x_rec[:original_len]
    return x_rec


def db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)


def compute_attack_release_coeff(ms: float, sr: int) -> float:
    """Time constant based coefficient for smoothing: alpha in y[n] = (1-alpha)*x[n] + alpha*y[n-1]"""
    # Typical approach: tau = ms/1000, alpha = exp(-1/(tau*sr)), clamp if tau very small
    tau = max(ms / 1000.0, 1e-6)
    return np.exp(-1.0 / (tau * sr))


# ---------- Base ----------
class AudioEffect:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate

    def process(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------- Effects ----------
class NoiseGate(AudioEffect):
    def __init__(self, sample_rate: int, threshold_db: float = -40.0, attack_ms: float = 5.0, release_ms: float = 100.0):
        super().__init__(sample_rate)
        self.threshold = db_to_linear(threshold_db)
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def process(self, audio: np.ndarray) -> np.ndarray:
        channels = to_channels(audio)
        out_ch = []
        attack_alpha = compute_attack_release_coeff(self.attack_ms, self.sr)
        release_alpha = compute_attack_release_coeff(self.release_ms, self.sr)

        for ch in channels:
            env = np.abs(ch)
            gain = np.ones_like(ch)
            smoothed = 0.0
            for i, e in enumerate(env):
                target = 1.0 if e >= self.threshold else 0.0
                if target < smoothed:
                    # release (decay toward target)
                    smoothed = release_alpha * smoothed + (1 - release_alpha) * target
                else:
                    # attack (rise toward target)
                    smoothed = attack_alpha * smoothed + (1 - attack_alpha) * target
                gain[i] = smoothed
            out_ch.append(ch * gain)
        return from_channels(out_ch)


class DeReverb(AudioEffect):
    def __init__(self, sample_rate: int, threshold_db: float = -30.0, reduction_db: float = -12.0, nperseg: int = 2048):
        super().__init__(sample_rate)
        self.threshold = db_to_linear(threshold_db)
        self.reduction = db_to_linear(reduction_db)
        self.nperseg = nperseg

    def process(self, audio: np.ndarray) -> np.ndarray:
        channels = to_channels(audio)
        processed = []
        for ch in channels:
            # STFT
            f, t, Zxx = signal.stft(ch, fs=self.sr, nperseg=self.nperseg)
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            # simple temporal decay model across frames
            decay = np.zeros_like(mag)
            decay[:, 1:] = mag[:, :-1] * 0.85
            diff = mag - decay
            # mask: if diff < threshold -> reduce energy
            mask = np.where(diff < self.threshold, self.reduction, 1.0)
            mag_clean = mag * mask
            Zxx_clean = mag_clean * np.exp(1j * phase)
            x_rec = stable_stft_istft_reconstruct(Zxx_clean, len(ch), fs=self.sr, nperseg=self.nperseg)
            processed.append(x_rec.astype(ch.dtype))
        return from_channels(processed)


class PhaseAlignment(AudioEffect):
    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)

    def process(self, audio: np.ndarray) -> np.ndarray:
        # Only act if multi-channel
        if audio.ndim < 2 or audio.shape[1] < 2:
            return audio
        # Align each other channel to the first channel
        channels = to_channels(audio)
        ref = channels[0]
        aligned = [ref]
        for ch in channels[1:]:
            corr = signal.correlate(ref, ch, mode='full')
            lags = signal.correlation_lags(len(ref), len(ch), mode='full')
            lag = lags[np.argmax(corr)]
            if lag > 0:
                # ch is delayed relative to ref: shift left (remove leading samples), pad end with zeros
                ch_al = np.concatenate([ch[lag:], np.zeros(min(lag, len(ch)), dtype=ch.dtype)])
            elif lag < 0:
                # ch leads ref: shift right (prepend zeros)
                pad = -lag
                ch_al = np.concatenate([np.zeros(min(pad, len(ch)), dtype=ch.dtype), ch[:len(ch)-pad]])
            else:
                ch_al = ch
            # ensure same length
            if len(ch_al) < len(ref):
                ch_al = np.pad(ch_al, (0, len(ref) - len(ch_al)))
            else:
                ch_al = ch_al[:len(ref)]
            aligned.append(ch_al)
        return from_channels(aligned)


class DeEsser(AudioEffect):
    def __init__(self, sample_rate: int, freq_range: Tuple[float, float] = (5000.0, 8500.0), threshold_db: float = -20.0, ratio: float = 4.0):
        super().__init__(sample_rate)
        self.freq_range = freq_range
        self.threshold = db_to_linear(threshold_db)
        self.ratio = ratio

    def process(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sr * 0.5
        low = max(0.001, self.freq_range[0] / nyquist)
        high = min(0.999, self.freq_range[1] / nyquist)
        if low >= high:
            # invalid band, bypass
            return audio
        b, a = signal.butter(4, [low, high], btype='band')
        channels = to_channels(audio)
        out = []
        # envelope smoothing kernel (use safe filtfilt)
        for ch in channels:
            # bandpassed sibilance
            s = safe_filtfilt(b, a, ch, axis=0)
            env = np.abs(s)
            # smooth envelope with a short lowpass (to avoid sample-level jitter)
            # design a simple 2nd order lowpass at e.g. 100Hz
            cutoff = 100.0 / nyquist
            if cutoff >= 0.999:
                env_s = env
            else:
                bb, aa = signal.butter(2, cutoff, btype='low')
                env_s = safe_filtfilt(bb, aa, env, axis=0)
            # avoid division by zero
            denom = np.maximum(env_s, 1e-12)
            over = env_s > self.threshold
            gain = np.ones_like(ch)
            # compression curve: above threshold reduce by ratio, smoothly proportional to overshoot
            gain[over] = 1.0 - (1.0 - (1.0 / self.ratio)) * ((env_s[over] - self.threshold) / denom[over])
            gain = np.clip(gain, 0.0, 1.0)
            out.append(ch * gain)
        return from_channels(out)


class ParametricEQ(AudioEffect):
    def __init__(self, sample_rate: int, low_shelf: Tuple[float, float] = (100.0, 0.0), high_shelf: Tuple[float, float] = (10000.0, 0.0)):
        super().__init__(sample_rate)
        self.low_freq, self.low_gain_db = low_shelf
        self.high_freq, self.high_gain_db = high_shelf

    def process(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sr * 0.5
        out = audio.copy().astype(np.float64)
        channels = to_channels(out)
        out_ch = []
        # implement as simple lowpass/highpass boosts using safe_filtfilt (approximate)
        if self.low_gain_db != 0.0:
            w = min(max(self.low_freq / nyquist, 1e-6), 0.999)
            b, a = signal.iirfilter(2, w, btype='lowpass', ftype='butter')
            for ch in channels:
                low_c = safe_filtfilt(b, a, ch, axis=0)
                gain_lin = db_to_linear(self.low_gain_db) - 1.0
                out_ch.append(ch + low_c * gain_lin)
            return from_channels(out_ch)
        if self.high_gain_db != 0.0:
            w = min(max(self.high_freq / nyquist, 1e-6), 0.999)
            b, a = signal.iirfilter(2, w, btype='highpass', ftype='butter')
            for ch in channels:
                high_c = safe_filtfilt(b, a, ch, axis=0)
                gain_lin = db_to_linear(self.high_gain_db) - 1.0
                out_ch.append(ch + high_c * gain_lin)
            return from_channels(out_ch)
        return audio


class TransientShaper(AudioEffect):
    def __init__(self, sample_rate: int, attack_gain: float = 1.5, sustain_gain: float = 1.0):
        super().__init__(sample_rate)
        self.attack_gain = attack_gain
        self.sustain_gain = sustain_gain

    def process(self, audio: np.ndarray) -> np.ndarray:
        channels = to_channels(audio)
        out = []
        # approximate envelope slow (lowpass) using safe_filtfilt
        for ch in channels:
            env_fast = np.abs(ch)
            nyq = self.sr * 0.5
            cutoff = min(20.0 / nyq, 0.999)
            if cutoff <= 0:
                env_slow = np.ones_like(env_fast) * np.mean(env_fast)
            else:
                b_slow, a_slow = signal.butter(2, cutoff, btype='low')
                env_slow = safe_filtfilt(b_slow, a_slow, env_fast, axis=0)
            attack_env = np.maximum(0.0, env_fast - env_slow)
            sustain_env = env_slow
            gain = 1.0 + (attack_env * (self.attack_gain - 1.0)) + (sustain_env * (self.sustain_gain - 1.0))
            out.append(ch * gain)
        return from_channels(out)


class DynamicCompressor(AudioEffect):
    def __init__(self, sample_rate: int, threshold_db: float = -18.0, ratio: float = 3.0, attack_ms: float = 10.0, release_ms: float = 100.0):
        super().__init__(sample_rate)
        self.threshold = db_to_linear(threshold_db)
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def process(self, audio: np.ndarray) -> np.ndarray:
        channels = to_channels(audio)
        out = []
        attack_alpha = compute_attack_release_coeff(self.attack_ms, self.sr)
        release_alpha = compute_attack_release_coeff(self.release_ms, self.sr)
        for ch in channels:
            env = np.abs(ch)
            gain = np.ones_like(ch)
            smoothed_gain = 1.0
            for i, e in enumerate(env):
                if e > self.threshold and e > 1e-12:
                    # desired linear gain to reduce amount above threshold according to ratio:
                    desired = (self.threshold + (e - self.threshold) / self.ratio) / e
                else:
                    desired = 1.0
                # smooth toward desired using attack/release (attack -> reduces gain quickly)
                if desired < smoothed_gain:
                    # reduce gain: use attack (faster)
                    smoothed_gain = attack_alpha * smoothed_gain + (1 - attack_alpha) * desired
                else:
                    # increase gain: use release (slower)
                    smoothed_gain = release_alpha * smoothed_gain + (1 - release_alpha) * desired
                gain[i] = smoothed_gain
            out.append(ch * gain)
        return from_channels(out)


class MultibandCompressor(AudioEffect):
    def __init__(self, sample_rate: int, crossover_freq: float = 500.0):
        super().__init__(sample_rate)
        self.crossover = crossover_freq
        # compressors for bands (they will be reused)
        self.low_comp = DynamicCompressor(sample_rate, threshold_db=-20.0, ratio=2.5)
        self.high_comp = DynamicCompressor(sample_rate, threshold_db=-15.0, ratio=3.5)

    def process(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sr * 0.5
        w = min(max(self.crossover / nyquist, 1e-6), 0.999)
        b_low, a_low = signal.butter(4, w, btype='lowpass')
        b_high, a_high = signal.butter(4, w, btype='highpass')

        # work channel-wise
        channels = to_channels(audio)
        out = []
        for ch in channels:
            low_band = safe_filtfilt(b_low, a_low, ch, axis=0)
            high_band = safe_filtfilt(b_high, a_high, ch, axis=0)
            low_processed = self.low_comp.process(low_band)
            high_processed = self.high_comp.process(high_band)
            # ensure same length
            if isinstance(low_processed, np.ndarray) and isinstance(high_processed, np.ndarray):
                L = max(len(low_processed), len(high_processed))
                if len(low_processed) < L:
                    low_processed = np.pad(low_processed, (0, L - len(low_processed)))
                if len(high_processed) < L:
                    high_processed = np.pad(high_processed, (0, L - len(high_processed)))
                out.append(low_processed + high_processed)
            else:
                out.append(ch)
        return from_channels(out)


class HarmonicExciter(AudioEffect):
    def __init__(self, sample_rate: int, drive: float = 1.5, blend: float = 0.3):
        super().__init__(sample_rate)
        self.drive = drive
        self.blend = blend

    def process(self, audio: np.ndarray) -> np.ndarray:
        driven = audio * self.drive
        saturated = np.tanh(driven)
        return (1.0 - self.blend) * audio + self.blend * saturated


class TapeTubeSimulator(AudioEffect):
    def __init__(self, sample_rate: int, saturation: float = 0.4):
        super().__init__(sample_rate)
        self.saturation = saturation

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = audio * (1.0 + self.saturation)
        # soft-clipping-like curve
        processed = np.where(x > 0, 1.0 - np.exp(-x), -1.0 + np.exp(x))
        nyquist = self.sr * 0.5
        cutoff = min(12000.0 / nyquist, 0.999)
        if cutoff <= 0:
            return processed
        b, a = signal.butter(2, cutoff, btype='low')
        return safe_filtfilt(b, a, processed, axis=0)


class AudioSuperResolution(AudioEffect):
    def __init__(self, sample_rate: int, cutoff_freq: float = 8000.0, exciter_amount: float = 0.2):
        super().__init__(sample_rate)
        self.cutoff = cutoff_freq
        self.amount = exciter_amount

    def process(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sr * 0.5
        w = min(max(self.cutoff / nyquist, 1e-6), 0.999)
        if w <= 0:
            return audio
        b, a = signal.butter(2, w, btype='highpass')
        hf = safe_filtfilt(b, a, audio, axis=0)
        # create harmonic-like enhancement per-channel
        harmonics = np.sign(hf) * (np.abs(hf) ** 1.2)
        # zero-mean per channel
        if harmonics.ndim == 2:
            means = np.mean(harmonics, axis=0, keepdims=True)
            harmonics = harmonics - means
        else:
            harmonics = harmonics - np.mean(harmonics)
        return audio + (harmonics * self.amount)


class StereoImager(AudioEffect):
    def __init__(self, sample_rate: int, width: float = 1.3):
        super().__init__(sample_rate)
        self.width = width

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim < 2 or audio.shape[1] < 2:
            return audio
        left = audio[:, 0]
        right = audio[:, 1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        side = side * self.width
        left_out = mid + side
        right_out = mid - side
        # preserve additional channels if present (append as unchanged)
        if audio.shape[1] > 2:
            extras = audio[:, 2:]
            return np.column_stack([left_out, right_out, extras])
        return np.column_stack([left_out, right_out])


class Limiter(AudioEffect):
    def __init__(self, sample_rate: int, threshold_db: float = -0.5):
        super().__init__(sample_rate)
        self.threshold = db_to_linear(threshold_db)

    def process(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > self.threshold and max_val > 0:
            return audio * (self.threshold / max_val)
        return audio


class LoudnessNormalizer(AudioEffect):
    def __init__(self, sample_rate: int, target_lufs: float = -14.0):
        super().__init__(sample_rate)
        self.target_lufs = target_lufs
        if not _HAS_PYL:
            warnings.warn("pyloudnorm not found: LoudnessNormalizer will use RMS fallback, not true LUFS.")

    def process(self, audio: np.ndarray) -> np.ndarray:
        # audio can be multi-channel; pyloudnorm expects mono or will compute on combined
        if _HAS_PYL:
            meter = pyln.Meter(self.sr)
            # convert to mono for loudness measurement (ITU-R BS.1770 uses K-weighting; pyloudnorm handles it)
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
            loudness = meter.integrated_loudness(mono)
            gain_db = self.target_lufs - loudness
            scaling = db_to_linear(gain_db)
            return audio * scaling
        else:
            # RMS fallback: interpret target_lufs as target RMS dB (approx)
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
            rms = np.sqrt(np.mean(np.square(mono)))
            if rms <= 1e-12:
                return audio
            target_lin = db_to_linear(self.target_lufs)
            scaling = target_lin / rms
            return audio * scaling


# ---------- Pipeline ----------
class AudioEnhancementPipeline:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate
        # chain order kept but can be reconfigured by user
        self.chain: List[AudioEffect] = [
            NoiseGate(sample_rate),
            DeReverb(sample_rate),
            PhaseAlignment(sample_rate),
            DeEsser(sample_rate),
            ParametricEQ(sample_rate, low_shelf=(80.0, -1.0), high_shelf=(12000.0, 1.5)),
            TransientShaper(sample_rate),
            DynamicCompressor(sample_rate),
            MultibandCompressor(sample_rate),
            HarmonicExciter(sample_rate),
            TapeTubeSimulator(sample_rate),
            AudioSuperResolution(sample_rate),
            StereoImager(sample_rate),
            Limiter(sample_rate),
            LoudnessNormalizer(sample_rate)
        ]

    def process_file(self, input_path: str, output_path: str, subtype: Optional[str] = None):
        audio, sr = sf.read(input_path, always_2d=False)
        # convert ints to float in [-1,1] if needed (soundfile usually returns float)
        # ensure float64 for processing, preserve dtype later
        orig_dtype = audio.dtype if hasattr(audio, "dtype") else None
        audio = audio.astype(np.float64)
        if sr != self.sr:
            self.sr = sr
            for effect in self.chain:
                effect.sr = sr

        output = audio
        for effect in self.chain:
            try:
                output = effect.process(output)
            except Exception as e:
                warnings.warn(f"Effect {effect.__class__.__name__} failed: {e}. Bypassing it.")
                # bypass on failure
        # Clip to [-1, 1] and cast to float32 for safe writing unless user wants otherwise
        output = np.clip(output, -1.0, 1.0)
        out_dtype = 'float32'
        sf.write(output_path, output.astype(np.float32), self.sr, subtype=subtype)

# ---------- CLI ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        pipeline = AudioEnhancementPipeline(sample_rate=44100)
        pipeline.process_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python audio_enhancement.py in.wav out.wav")
