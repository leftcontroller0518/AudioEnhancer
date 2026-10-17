import numpy as np
import scipy.signal as signal
import soundfile as sf
from typing import List, Tuple, Optional

class AudioEffect:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate

    def process(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class NoiseGate(AudioEffect):
    def __init__(self, sample_rate: int, threshold_db: float = -40.0, attack_ms: float = 5.0, release_ms: float = 100.0):
        super().__init__(sample_rate)
        self.threshold = 10 ** (threshold_db / 20.0)
        self.attack_samples = int((attack_ms / 1000.0) * sample_rate)
        self.release_samples = int((release_ms / 1000.0) * sample_rate)

    def process(self, audio: np.ndarray) -> np.ndarray:
        is_stereo = audio.ndim == 2
        mono = np.mean(audio, axis=1) if is_stereo else audio
        envelope = np.abs(mono)
        
        gain = np.ones_like(mono)
        current_gain = 1.0
        
        attack_step = 1.0 / max(1, self.attack_samples)
        release_step = 1.0 / max(1, self.release_samples)
        
        for i in range(len(mono)):
            target_gain = 1.0 if envelope[i] >= self.threshold else 0.0
            if current_gain < target_gain:
                current_gain = min(1.0, current_gain + attack_step)
            elif current_gain > target_gain:
                current_gain = max(0.0, current_gain - release_step)
            gain[i] = current_gain
            
        if is_stereo:
            return audio * gain[:, np.newaxis]
        return audio * gain

class DeReverb(AudioEffect):
    def __init__(self, sample_rate: int, threshold_db: float = -30.0, reduction_db: float = -12.0):
        super().__init__(sample_rate)
        self.threshold = 10 ** (threshold_db / 20.0)
        self.reduction = 10 ** (reduction_db / 20.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        is_stereo = audio.ndim == 2
        channels = [audio[:, i] for i in range(2)] if is_stereo else [audio]
        processed_channels = []

        for ch in channels:
            f, t, Zxx = signal.stft(ch, fs=self.sr, nperseg=2048)
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            decay = np.zeros_like(mag)
            decay[:, 1:] = mag[:, :-1] * 0.85
            diff = mag - decay
            
            mask = np.where(diff < self.threshold, self.reduction, 1.0)
            mag_clean = mag * mask
            
            Zxx_clean = mag_clean * np.exp(1j * phase)
            _, x_rec = signal.istft(Zxx_clean, fs=self.sr)
            processed_channels.append(x_rec[:len(ch)])

        if is_stereo:
            return np.column_stack(processed_channels)
        return processed_channels[0]

class PhaseAlignment(AudioEffect):
    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim < 2:
            return audio
        
        ch1, ch2 = audio[:, 0], audio[:, 1]
        correlation = signal.correlate(ch1, ch2, mode='full')
        lags = signal.correlation_lags(len(ch1), len(ch2), mode='full')
        lag = lags[np.argmax(correlation)]
        
        if lag > 0:
            ch2_aligned = np.pad(ch2, (lag, 0))[:len(ch2)]
            return np.column_stack([ch1, ch2_aligned])
        elif lag < 0:
            ch1_aligned = np.pad(ch1, (-lag, 0))[:len(ch1)]
            return np.column_stack([ch1_aligned, ch2])
        return audio

class DeEsser(AudioEffect):
    def __init__(self, sample_rate: int, freq_range: Tuple[float, float] = (5000.0, 8500.0), threshold_db: float = -20.0, ratio: float = 4.0):
        super().__init__(sample_rate)
        self.freq_range = freq_range
        self.threshold = 10 ** (threshold_db / 20.0)
        self.ratio = ratio

    def process(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sr * 0.5
        b, a = signal.butter(4, [self.freq_range[0] / nyquist, self.freq_range[1] / nyquist], btype='band')
        
        is_stereo = audio.ndim == 2
        channels = [audio[:, i] for i in range(2)] if is_stereo else [audio]
        processed_channels = []

        for ch in channels:
            sibilance = signal.filtfilt(b, a, ch)
            env = np.abs(sibilance)
            
            gain = np.ones_like(ch)
            over = env > self.threshold
            gain[over] = 1.0 - (1.0 - (1.0 / self.ratio)) * ((env[over] - self.threshold) / env[over])
            
            processed_channels.append(ch * gain)

        if is_stereo:
            return np.column_stack(processed_channels)
        return processed_channels[0]

class ParametricEQ(AudioEffect):
    def __init__(self, sample_rate: int, low_shelf: Tuple[float, float] = (100.0, 0.0), high_shelf: Tuple[float, float] = (10000.0, 0.0)):
        super().__init__(sample_rate)
        self.low_freq, self.low_gain = low_shelf
        self.high_freq, self.high_gain = high_shelf

    def process(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sr * 0.5
        out = audio.copy()
        
        if self.low_gain != 0.0:
            b, a = signal.iirfilter(2, self.low_freq / nyquist, btype='lowpass', ftype='butter')
            low_component = signal.filtfilt(b, a, out, axis=0)
            out += low_component * ((10 ** (self.low_gain / 20.0)) - 1.0)
            
        if self.high_gain != 0.0:
            b, a = signal.iirfilter(2, self.high_freq / nyquist, btype='highpass', ftype='butter')
            high_component = signal.filtfilt(b, a, out, axis=0)
            out += high_component * ((10 ** (self.high_gain / 20.0)) - 1.0)
            
        return out

class TransientShaper(AudioEffect):
    def __init__(self, sample_rate: int, attack_gain: float = 1.5, sustain_gain: float = 1.0):
        super().__init__(sample_rate)
        self.attack_gain = attack_gain
        self.sustain_gain = sustain_gain

    def process(self, audio: np.ndarray) -> np.ndarray:
        is_stereo = audio.ndim == 2
        mono = np.mean(audio, axis=1) if is_stereo else audio
        
        env_fast = np.abs(mono)
        b_slow, a_slow = signal.butter(2, 20.0 / (self.sr * 0.5), btype='low')
        env_slow = signal.filtfilt(b_slow, a_slow, env_fast)
        
        attack_env = np.maximum(0, env_fast - env_slow)
        sustain_env = env_slow
        
        gain = 1.0 + (attack_env * (self.attack_gain - 1.0)) + (sustain_env * (self.sustain_gain - 1.0))
        
        if is_stereo:
            return audio * gain[:, np.newaxis]
        return audio * gain

class DynamicCompressor(AudioEffect):
    def __init__(self, sample_rate: int, threshold_db: float = -18.0, ratio: float = 3.0, attack_ms: float = 10.0, release_ms: float = 100.0):
        super().__init__(sample_rate)
        self.threshold = 10 ** (threshold_db / 20.0)
        self.ratio = ratio
        self.attack_samples = int((attack_ms / 1000.0) * sample_rate)
        self.release_samples = int((release_ms / 1000.0) * sample_rate)

    def process(self, audio: np.ndarray) -> np.ndarray:
        is_stereo = audio.ndim == 2
        mono = np.mean(audio, axis=1) if is_stereo else audio
        
        env = np.abs(mono)
        gain = np.ones_like(mono)
        
        attack_step = 1.0 / max(1, self.attack_samples)
        release_step = 1.0 / max(1, self.release_samples)
        
        current_gain = 1.0
        for i in range(len(mono)):
            if env[i] > self.threshold:
                target_gain = (self.threshold + (env[i] - self.threshold) / self.ratio) / env[i]
            else:
                target_gain = 1.0
                
            if current_gain > target_gain:
                current_gain = max(target_gain, current_gain - attack_step)
            else:
                current_gain = min(target_gain, current_gain + release_step)
            gain[i] = current_gain

        if is_stereo:
            return audio * gain[:, np.newaxis]
        return audio * gain

class MultibandCompressor(AudioEffect):
    def __init__(self, sample_rate: int, crossover_freq: float = 500.0):
        super().__init__(sample_rate)
        self.crossover = crossover_freq
        self.low_comp = DynamicCompressor(sample_rate, threshold_db=-20.0, ratio=2.5)
        self.high_comp = DynamicCompressor(sample_rate, threshold_db=-15.0, ratio=3.5)

    def process(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sr * 0.5
        b_low, a_low = signal.butter(4, self.crossover / nyquist, btype='lowpass')
        b_high, a_high = signal.butter(4, self.crossover / nyquist, btype='highpass')

        low_band = signal.filtfilt(b_low, a_low, audio, axis=0)
        high_band = signal.filtfilt(b_high, a_high, audio, axis=0)

        low_processed = self.low_comp.process(low_band)
        high_processed = self.high_comp.process(high_band)

        return low_processed + high_processed

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
        processed = np.where(x > 0, 1.0 - np.exp(-x), -1.0 + np.exp(x))
        nyquist = self.sr * 0.5
        b, a = signal.butter(2, 12000.0 / nyquist, btype='lowpass')
        return signal.filtfilt(b, a, processed, axis=0)

class AudioSuperResolution(AudioEffect):
    def __init__(self, sample_rate: int, cutoff_freq: float = 8000.0, exciter_amount: float = 0.2):
        super().__init__(sample_rate)
        self.cutoff = cutoff_freq
        self.amount = exciter_amount

    def process(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sr * 0.5
        b, a = signal.butter(2, self.cutoff / nyquist, btype='highpass')
        high_freqs = signal.filtfilt(b, a, audio, axis=0)
        
        harmonics = np.abs(high_freqs) ** 1.2
        harmonics = harmonics - np.mean(harmonics, axis=0)
        
        return audio + (harmonics * self.amount)

class StereoImager(AudioEffect):
    def __init__(self, sample_rate: int, width: float = 1.3):
        super().__init__(sample_rate)
        self.width = width

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim < 2:
            return audio
        
        mid = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5
        
        side *= self.width
        
        left = mid + side
        right = mid - side
        return np.column_stack([left, right])

class Limiter(AudioEffect):
    def __init__(self, sample_rate: int, threshold_db: float = -0.5):
        super().__init__(sample_rate)
        self.threshold = 10 ** (threshold_db / 20.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > self.threshold:
            return audio * (self.threshold / max_val)
        return audio

class LoudnessNormalizer(AudioEffect):
    def __init__(self, sample_rate: int, target_lufs: float = -14.0):
        super().__init__(sample_rate)
        self.target_gain = 10 ** (target_lufs / 20.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(np.square(audio)))
        if rms == 0:
            return audio
        scaling = self.target_gain / rms
        return audio * scaling

class AudioEnhancementPipeline:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate
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

    def process_file(self, input_path: str, output_path: str):
        audio, sr = sf.read(input_path)
        if sr != self.sr:
            self.sr = sr
            for effect in self.chain:
                effect.sr = sr
        
        output = audio
        for effect in self.chain:
            output = effect.process(output)
            
        sf.write(output_path, output, self.sr)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        pipeline = AudioEnhancementPipeline(sample_rate=44100)
        pipeline.process_file(sys.argv[1], sys.argv[2])
