import numpy as np
from scipy.signal import butter, filtfilt


def analyze_alpha_power(
    eeg_array,
    fs,
    baseline_time,
    stimulation_time,
    baseline_exclude=30,
    alpha_band=(8, 12),
):
    """
    eeg_array: shape (N, 4) [timestamp, ch1, ch2, ch3]
    fs: sampling rate
    baseline_time: seconds of baseline
    stimulation_time: seconds of stimulation
    baseline_exclude: seconds to exclude from start of baseline
    alpha_band: tuple (low, high) in Hz
    """
    # 1. Get time axis
    t = (eeg_array[:, 0] - eeg_array[0, 0]) / 1e6  # us to s

    # 2. Indices for baseline and stimulation
    baseline_start = baseline_exclude
    baseline_end = baseline_time
    stim_start = baseline_time
    stim_end = baseline_time + stimulation_time

    baseline_mask = (t >= baseline_start) & (t < baseline_end)
    stim_mask = (t >= stim_start) & (t < stim_end)

    # 3. Bandpass filter for alpha
    b, a = butter(2, [alpha_band[0] / (fs / 2), alpha_band[1] / (fs / 2)], btype="band")
    eeg_alpha = filtfilt(b, a, eeg_array[:, 1:4], axis=0)

    # 4. Compute 1s windowed power
    win_size = int(fs)  # 1s window
    step = 1  # slide by 1 sample for max resolution, or win_size for non-overlapping

    def windowed_power(data, mask):
        idx = np.where(mask)[0]
        powers = []
        for start in range(idx[0], idx[-1] - win_size + 1, step):
            win = data[start : start + win_size]
            # mean squared amplitude (power)
            powers.append(np.mean(win**2, axis=0))
        return np.array(powers)

    baseline_powers = windowed_power(eeg_alpha, baseline_mask)
    stim_powers = windowed_power(eeg_alpha, stim_mask)

    # 5. Z-score using baseline
    mean = np.mean(baseline_powers, axis=0)
    std = np.std(baseline_powers, axis=0)
    stim_z = (stim_powers - mean) / std
    perc90 = np.percentile(stim_z, 90, axis=0)
    mean_z = np.mean(stim_z, axis=0)
    return perc90, mean, std, stim_z, mean_z
