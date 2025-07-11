import serial
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
from scipy.signal import filtfilt
import os
from datetime import datetime
import re
from collections import defaultdict
from math import degrees
import serial.tools.list_ports
from post_hoc_analysis import analyze_alpha_power
from oscilltrack import OscillTrack

# --- GP-UCB controller imports ----------------------------------------
from tv_gp_ucb import (
    initialise_data,
    build_model,
    acquisition_func,
    evaluate_information,
    replace_next_value,
    uniform_time_increase,
    TestData,
)

"""
Elemind Headband Python Interface with Closed-Loop Control

This script provides real-time EEG monitoring and closed-loop control using the Elemind headband.
The closed-loop control system triggers pink noise audio when the instantaneous phase meets
specific criteria.

CLOSED-LOOP CONTROL FEATURE:
- Monitors instantaneous phase from EEG data
- Triggers pink noise when phase enters target range
- Configurable parameters:
  * target_phase_rad: Target phase value (default: π radians)
  * phase_tolerance: Tolerance around target (default: ±0.2 radians)
  * pink_noise_volume: Audio volume (0.0-1.0, default: 0.5)
  * pink_noise_fade_in_ms: Fade-in time (default: 100ms)
  * pink_noise_fade_out_ms: Fade-out time (default: 100ms)

Example usage:
    headband = ElemindHeadband("COM6", debug=True)
    headband.target_phase_rad = np.pi/2  # 90 degrees
    headband.phase_tolerance = 0.1       # ±5.7 degrees
    headband.pink_noise_volume = 0.3     # 30% volume
"""


class ElemindFilter2ndOrder:
    """2nd order IIR filter (for high-pass)"""

    def __init__(self, b_coeffs: np.ndarray, a_coeffs: np.ndarray):
        self.b = np.array(b_coeffs, dtype=np.float64)
        self.a = np.array(a_coeffs, dtype=np.float64)
        self.w1 = 0.0
        self.w2 = 0.0

    def filter_sample(self, x: float) -> float:
        """Apply 2nd order IIR filter"""
        x = float(x)

        if not np.isfinite(x):
            return 0.0

        y = self.b[0] * x + self.w1
        self.w1 = self.b[1] * x + self.w2 - self.a[1] * y
        self.w2 = self.b[2] * x - self.a[2] * y

        if not np.isfinite(y):
            y = 0.0
            self.w1 = 0.0
            self.w2 = 0.0

        self.w1 = np.clip(self.w1, -1e6, 1e6)
        self.w2 = np.clip(self.w2, -1e6, 1e6)

        return y


class ElemindFilter4thOrder:
    """4th order IIR filter (for bandpass/bandstop)"""

    def __init__(self, b_coeffs: np.ndarray, a_coeffs: np.ndarray):
        self.b = np.array(b_coeffs, dtype=np.float64)
        self.a = np.array(a_coeffs, dtype=np.float64)
        self.w1 = 0.0
        self.w2 = 0.0
        self.w3 = 0.0
        self.w4 = 0.0

    def filter_sample(self, x: float) -> float:
        """Apply 4th order IIR filter"""
        x = float(x)

        if not np.isfinite(x):
            return 0.0

        y = self.b[0] * x + self.w1
        self.w1 = self.b[1] * x + self.w2 - self.a[1] * y
        self.w2 = self.b[2] * x + self.w3 - self.a[2] * y
        self.w3 = self.b[3] * x + self.w4 - self.a[3] * y
        self.w4 = self.b[4] * x - self.a[4] * y

        if not np.isfinite(y):
            y = 0.0
            self.w1 = 0.0
            self.w2 = 0.0
            self.w3 = 0.0
            self.w4 = 0.0

        self.w1 = np.clip(self.w1, -1e6, 1e6)
        self.w2 = np.clip(self.w2, -1e6, 1e6)
        self.w3 = np.clip(self.w3, -1e6, 1e6)
        self.w4 = np.clip(self.w4, -1e6, 1e6)

        return y


class ElemindHeadband:
    """Main class for Elemind Headband interface"""

    def __init__(self, port: str, baudrate: int = 115200, debug: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.debug_mode = debug

        # Recording parameters
        self.stimulation_time = 2 * 60  # Duration of active stimulation
        self.baseline_time = 60  # baseline before stimulation starts
        self.sampling_duration_secs = (
            self.stimulation_time + self.baseline_time
        )  # Total recording time
        self.time_start = self.baseline_time  # Start processing after 20 seconds
        self.time_end = self.sampling_duration_secs  # End 20 seconds before end

        # Data processing parameters
        self.fs = 250  # Sampling rate
        self.ts = 1 / self.fs
        self.bandpass_centre_freq = 10

        self.sample_count = 0  # j in MATLAB

        # Control flags
        self.enable_real_time_plotting = False

        # Setup filters
        self._setup_filters()

        # Data storage for post-processing
        self.log_file = None
        self.abs_log_path = None
        self.raw_eeg_data = []  # Store raw EEG for post-processing
        self.inst_amp_phs_data = []  # Store instantaneous amplitude/phase data

        # Threading
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Conversion factors
        self.eeg_adc2volts_numerator = 4.5 / (8388608.0 - 1)
        self.eeg_gain_default = 24
        self.eeg_adc2volts = self.eeg_adc2volts_numerator / self.eeg_gain_default
        self.accel_raw2g = 1 / (2**14)

        # Plotting buffers
        self.plot_update_counter = 0
        self.plot_update_interval = 250  # Update every 250 samples (1 second)
        self.simple_eeg_buffer = np.zeros((1000, 3))  # 4 seconds of data
        self.buffer_t = self.ts * np.linspace(-1000, 0, 1000)

        # Session timing
        self.session_start_time = None

        # New buffers for amplitude and phase
        self.inst_amp_buffer = np.zeros(1000)  # Last 4 seconds of amplitude
        self.inst_phase_buffer = np.zeros(1000)  # Last 4 seconds of phase

        # Buffer for rolling 1s average alpha amplitude (last 4 seconds)
        self.avg_alpha_amp_buffer = np.zeros(1000)

        # ----- rolling average of alpha-band amplitude -----
        self.avg_alpha_amp_last_sec = 0.0  # most-recent 1-s mean
        self.alpha_amp_history = []  # (timestamp, mean) log
        self._amp_sample_counter = 0  # counts samples since last update

        # Closed-loop control parameters
        self.target_phase_rad = 0  # [np.pi/3, 5*np.pi/6, 4*np.pi/3, 11*np.pi/6]  # Target phase value (modify by James Bayes Optimisation)
        self.phase_tolerance = 0.1  # Tolerance around target phase (radians)
        self.pink_noise_volume = 1.0  # Pink noise volume (0.0 to 1.0)
        self.pink_noise_fade_in_ms = 0  # Fade in time in milliseconds
        self.pink_noise_fade_out_ms = 0  # Fade out time in milliseconds
        self.pink_noise_active = False  # Track if pink noise is currently playing
        self.phase_trigger_count = 0  # Count how many times phase trigger occurred

        # controller init
        # ---------- GP-UCB controller state ----------
        self._controller_cfg = {
            "remember_n_stims": 20,
            "res_phase": 61,
            "phase_min": 0.0,
            "phase_max": 2 * np.pi,
            "decay_constant": 0.9,
            "acquisition_k": 0.2,
            "noise_sd": 0.0,
        }
        self._controller_beta = 1.0
        self._controller_retained: TestData | None = None
        self._controller_ready = False
        self._last_next_stim = None  # keeps previous acquisition output

        #osciltrack
        self.tracker = OscillTrack(fc_hz=10.0, fs_hz=250, g=2 ** -4)
        self.amp_est = [0]
        self.phase_est = [0]

        # New buffers for tracking events in the phase plot
        self.pink_noise_events = []  # List of (timestamp, event_type) where event_type is 'start' or 'stop'
        self.target_phase_updates = []  # List of (timestamp, phase_value) for target phase changes
        self.target_phase_buffer = np.full(1000, np.nan)  # Buffer for plotting target phase over time
        
        # Track the current target phase for comparison
        self._last_target_phase = None

    def _setup_filters(self):
        """Initialize digital filters"""
        # High-pass filter (0.5 Hz) - 2nd order
        b_hpf, a_hpf = signal.butter(2, 0.5 / (self.fs / 2), "high")
        self.hpf = ElemindFilter2ndOrder(b_hpf, a_hpf)
        self.hpf_on = True

        # Band-pass filter (alpha band) - 4th order
        low_freq = self.bandpass_centre_freq * (1 - 0.25)
        high_freq = self.bandpass_centre_freq * (1 + 0.25)
        filtfreq = np.array([low_freq, high_freq])
        b_bpf, a_bpf = signal.butter(2, filtfreq / (self.fs / 2), "band")
        self.bpf = ElemindFilter4thOrder(b_bpf, a_bpf)
        self.bpf_on = False

        # Band-stop filter (line noise 45-55 Hz) - 4th order
        filt_line = np.array([45, 55])
        b_bsf, a_bsf = signal.butter(2, filt_line / (self.fs / 2), "stop")
        self.bsf = ElemindFilter4thOrder(b_bsf, a_bsf)
        self.bsf_on = True

        # Store coefficients for post-processing
        self.b_hpf, self.a_hpf = b_hpf, a_hpf
        self.b_bpf, self.a_bpf = b_bpf, a_bpf
        self.b_bsf, self.a_bsf = b_bsf, a_bsf

        if self.debug_mode:
            print("Filters initialized:")
            print(f"HPF: {self.hpf_on}, BPF: {self.bpf_on}, BSF: {self.bsf_on}")

    # ======================================================================
    # GP-UCB CONTROLLER
    # ======================================================================

    def _controller_initialise(self) -> None:
        """Run once: seed the rolling data window and choose the first phase."""
        if self._controller_ready:
            return

        cfg = self._controller_cfg

        # Seed inputs uniformly and set their outputs to the current alpha value
        self._controller_retained = initialise_data(
            cfg["remember_n_stims"],
            cfg["phase_min"],
            cfg["phase_max"],
            cfg["noise_sd"],
            test_variables=np.zeros(4),
            rng=np.random.default_rng(seed=int(time.time())),
        )
        self._controller_retained.output_samples[:] = self.avg_alpha_amp_last_sec

        # Fit model and pick first stimulation phase
        model = build_model(
            self._controller_retained,
            cfg["res_phase"],
            cfg["phase_min"],
            cfg["phase_max"],
        )
        self._last_next_stim = acquisition_func(model, self._controller_beta)
        self.target_phase_rad = [float(self._last_next_stim.stim_variable[0])]
        self._controller_ready = True

        if self.debug_mode:
            print(
                f"[Controller] Started, first phase = {self.target_phase_rad[0]:.2f} rad"
            )

    def _controller_iterate(self) -> None:
        """Call once every second after avg_alpha_amp_last_sec is updated."""
        if not self._controller_ready:
            self._controller_initialise()
            return

        cfg = self._controller_cfg
        measurement = self.avg_alpha_amp_last_sec

        # ----- β update ----------------------------------------------------
        self._controller_beta = evaluate_information(
            self._last_next_stim.expected_sig,
            self._last_next_stim.expected_mew,
            measurement,
            cfg["decay_constant"],
            cfg["acquisition_k"],
        )

        # ----- overwrite one sample in the rolling window -----------------
        idx = replace_next_value(self._controller_retained, self._last_next_stim)

        # Age records by 1 s, insert latest observation
        self._controller_retained = uniform_time_increase(
            self._controller_retained, 1.0
        )
        self._controller_retained.inputs_samples[idx] = (
            self._last_next_stim.stim_variable
        )
        self._controller_retained.output_samples[idx] = measurement
        self._controller_retained.time[idx] = 0.0

        # ----- model + acquisition ----------------------------------------
        model = build_model(
            self._controller_retained,
            cfg["res_phase"],
            cfg["phase_min"],
            cfg["phase_max"],
        )
        self._last_next_stim = acquisition_func(model, self._controller_beta)

        # Update target phase used by the existing phase-trigger routine
        new_target_phase = float(self._last_next_stim.stim_variable[0])
        
        # Check if target phase has changed
        if self._last_target_phase is None or abs(new_target_phase - self._last_target_phase) > 0.01:
            current_time = self.sample_count * self.ts
            self.target_phase_updates.append((current_time, new_target_phase))
            self._last_target_phase = new_target_phase
            
            if self.debug_mode:
                print(f"[Controller] Target phase updated to {new_target_phase:.2f} rad at {current_time:.1f}s")
        
        self.target_phase_rad = [new_target_phase]

        if self.debug_mode:
            print(f"Target phase: {self.target_phase_rad}")
            print(
                f"[Controller] New phase = {self.target_phase_rad[0]:.2f} rad   "
                f"β = {self._controller_beta:4.2f}"
            )

    # ======================================================================
    # End GP-UCB CONTROLLER
    # ======================================================================

    def connect(self) -> bool:
        """Connect to Elemind headband"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
            )
            self.is_connected = True
            print(f"Connected to Elemind headband on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from headband"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        print("Disconnected from Elemind headband")

    def send_command(self, command: str, print_cmd: bool = True) -> bool:
        """Send command to headband
        Parameters:
         * command - string containing the command
         * print_cmd - boolean flag for whether or not to print the same command in the console
        """
        if not self.is_connected:
            print("Not connected to headband")
            return False

        try:
            self.serial_conn.write(f"{command}\n".encode())
            if print_cmd:
                print(f"Sent: {command}")

            # Log command
            if self.log_file:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.log_file.write(f"[{timestamp}] Sent: {command}\n")
                self.log_file.flush()

            return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            return False

    def start_logging(self, log_filename: str = None) -> str:
        """Start logging data to file"""
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = f"elemind_python_log_{timestamp}.txt"

        if os.path.exists(os.path.abspath(log_filename)):
            filesplit = log_filename.split(".")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = filesplit[0] + "_" + timestamp + "." + filesplit[1]

        self.log_file = open(log_filename, "a")
        self.abs_log_path = os.path.abspath(log_filename)
        print(f"Logging started: {self.abs_log_path}")
        return self.abs_log_path

    def stop_logging(self):
        """Stop logging data"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            print("Logging stopped")

    def setup_audio(self):
        """Configure audio output... and setup pink noise for session"""
        print("Configuring audio output...")

        # Set master volume
        self.send_command("audio_set_volume 255")  # 50% master volume

        # Example code to test audio briefly
        print("Testing audio output...")
        self.send_command("audio_play_test 440")
        time.sleep(1)
        self.send_command("audio_stop_test")
        print("Audio test complete.")
        # self.send_command("audio_bgwav_play /audio/RAIN_22M.wav 1")
        # self.send_command("audio_bg_volume 0.2")
        

        # Setup pink noise for session
        self.setup_pink_noise()

    def setup_pink_noise(self):
        """Configure pink noise playback and fade parameters at session start."""
        # self.send_command('audio_set_volume 255')

        # self.send_command(f"audio_pink_fade_out {self.pink_noise_fade_out_ms}", False)
        # self.send_command("audio_pink_unmute", False)
        self.send_command("audio_pink_volume 1.0", False)  # Start muted
        self.send_command("audio_pink_fade_in 0", False)
        self.send_command("audio_pink_play", False)

        self._stop_pink_noise()  # Ensure pink noise is stopped initially

    def start_streaming(self):
        """Start EEG data streaming"""
        print("Starting EEG streaming...")

        # Set filter centre frequency for phase computation
        # self.send_command("therapy_enable_alpha_track 0")
        # self.send_command("echt_config_simple 10")

        # Enable streaming
        self.send_command("stream eeg 1")
        self.send_command("echt_start")
        # self.send_command("audio_pink_volume 1")
        self.send_command("stream inst_amp_phs 1")
        self.send_command("stream accel 0")
        self.send_command("stream audio 0")
        self.send_command("stream leadoff 1")

        # Setup filters
        self.send_command("therapy_enable_line_filters 1")
        self.send_command("therapy_enable_az_filters 1")
        self.send_command("therapy_enable_ac_filters 1")

        # Start session
        self.send_command("eeg_start")
        self.send_command("accel_start")

    def stop_streaming(self):
        """Stop EEG data streaming"""
        print("Stopping EEG streaming...")

        self.send_command("eeg_stop")
        self.send_command("accel_stop")

        # Stop audio
        self.send_command("audio_pink_volume 0")  # Ensure pink noise is muted
        self.send_command("audio_pink_stop")
        self.send_command("audio_pink_unmute")
        self.send_command("audio_bg_fade_out 0")
        self.send_command("audio_bgwav_stop")

    def _serial_reader_thread(self):
        """Thread function for reading serial data"""
        while not self.stop_event.is_set() and self.is_connected:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = (
                        self.serial_conn.readline()
                        .decode("utf-8", errors="ignore")
                        .strip()
                    )
                    if line:
                        # Log received data
                        if self.log_file:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                                :-3
                            ]
                            self.log_file.write(f"[{timestamp}] Recv: {line}\n")
                            self.log_file.flush()

                        # Process data
                        self._process_data_line(line)

                time.sleep(0.0001)

            except Exception as e:
                print(f"Serial reader error: {e}")
                break

    def _process_data_line(self, data: str):
        # Parse data using regex
        match = re.search(r"V \((\d+)\) data_log_(\w+): ([\d\.\-\s]+)", data)
        if not match:
            return

        rtos_timestamp = int(match.group(1))
        data_type = match.group(2)
        values_str = match.group(3)

        try:
            values = [float(x) for x in values_str.split()]
            if not values:
                return

            value_timestamp = values[0]
            value_data = np.array(values[1:])

            if data_type == "eeg":
                self._process_eeg_sample(value_timestamp, value_data)
            elif data_type == "accel":
                value_data = value_data * self.accel_raw2g
                # Store if needed
            elif data_type == "inst_amp_phs":
                self._process_inst_amp_phs(value_timestamp, value_data)

        except Exception as e:
            if self.debug_mode:
                print(f"Error processing data: {e}")

    def _process_eeg_sample(self, timestamp: float, eeg_raw: np.ndarray):
        if len(eeg_raw) != 3:
            return

        # Convert to volts
        eeg_volts = eeg_raw * self.eeg_adc2volts

        # Store raw EEG sample
        self.raw_eeg_data.append([timestamp] + eeg_volts.tolist())

        # Check for valid data
        if not np.all(np.isfinite(eeg_volts)):
            eeg_volts = np.zeros(3)

        # Apply filters per channel
        eeg_filtered = np.zeros(3, dtype=np.float64)
        for ch in range(3):
            sample = float(eeg_volts[ch])

            if self.bsf_on:
                sample = self.bsf.filter_sample(sample)
            if self.hpf_on:
                sample = self.hpf.filter_sample(sample)
            if self.bpf_on:
                sample = self.bpf.filter_sample(sample)

            eeg_filtered[ch] = sample

        # Update plotting buffers
        if self.enable_real_time_plotting:
            self._update_plotting_buffers(eeg_filtered)

        # START YOUR CLOSED LOOP CONTROL HERE TODO: add

        # self.tracker
        self.phase_est, self.amp_est = self.tracker.step(sample)
        self.inst_amp_buffer[:-1]   = self.inst_amp_buffer[1:]
        self.inst_amp_buffer[-1]    = self.amp_est
        self.inst_phase_buffer[:-1] = self.inst_phase_buffer[1:]
        self.inst_phase_buffer[-1]  = self.phase_est % (2*np.pi)
        

        # END YOUR CLOSED LOOP CONTROL HERE

        # Progress reporting
        if self.sample_count % 1000 == 0:
            elapsed_time = self.sample_count / self.fs
            print(f"EEG samples: {self.sample_count} ({elapsed_time:.1f} seconds)")

        self.sample_count += 1

    def _update_target_phase_buffer(self):
        """Update the target phase buffer for real-time plotting"""
        # Shift buffer
        self.target_phase_buffer[:-1] = self.target_phase_buffer[1:]
        
        # Set the new value based on current target phase
        if len(self.target_phase_rad) > 0:
            self.target_phase_buffer[-1] = self.target_phase_rad[0]
        else:
            self.target_phase_buffer[-1] = np.nan

    def _process_inst_amp_phs(self, timestamp: float, inst_data: np.ndarray):
        if len(inst_data) >= 2:
            # amp_volts = inst_data[0] * self.eeg_adc2volts
            # # phase_rad = radians(inst_data[1]) if len(inst_data) > 1 else 0.0
            # phase_rad = inst_data[1] if len(inst_data) > 1 else 0.0
            amp_volts  = self.amp_est
            phase_rad = self.phase_est
            # phase_est, amp_est = self.tracker.step()


            self.inst_amp_phs_data.append([timestamp, amp_volts, phase_rad])

            # CLOSED LOOP CONTROL: Trigger pink noise based on phase
            self._check_phase_trigger(phase_rad)

              # keep 0‒2π    


            # Update live buffers
            # self.inst_amp_buffer[:-1] = self.inst_amp_buffer[1:]
            # self.inst_amp_buffer[-1] = amp_volts
            # self.inst_phase_buffer[:-1] = self.inst_phase_buffer[1:]
            # self.inst_phase_buffer[-1] = phase_rad % (2 * np.pi)  # Ensure 0-2pi

            # Update target phase buffer for plotting
            self._update_target_phase_buffer()

            # Compute rolling 1s average for the last 250 samples at every sample
            if self.sample_count >= self.fs:
                rolling_avg = float(np.mean(np.abs(self.inst_amp_buffer[-self.fs :])))
                self.avg_alpha_amp_buffer[:-1] = self.avg_alpha_amp_buffer[1:]
                self.avg_alpha_amp_buffer[-1] = rolling_avg
            else:
                # Not enough samples yet, keep as zero or partial mean
                self.avg_alpha_amp_buffer[:-1] = self.avg_alpha_amp_buffer[1:]
                self.avg_alpha_amp_buffer[-1] = 0.0

            # ----- maintain one-second average -----
            self._amp_sample_counter += 1

            if self._amp_sample_counter >= self.fs:  # 250 samples ≈ 1 s
                last_sec_amp = self.inst_amp_buffer[-self.fs :]  # most-recent second
                self.avg_alpha_amp_last_sec = float(np.mean(np.abs(last_sec_amp)))

                # save to history for later inspection
                self.alpha_amp_history.append((timestamp, self.avg_alpha_amp_last_sec))

                # update rolling buffer for plotting
                self.avg_alpha_amp_buffer[:-1] = self.avg_alpha_amp_buffer[1:]
                self.avg_alpha_amp_buffer[-1] = self.avg_alpha_amp_last_sec

                # reset for the next second
                self._amp_sample_counter = 0

                if self.debug_mode:
                    print(f"1-s avg α-amp: {self.avg_alpha_amp_last_sec:.3e} V")
                # Only iterate controller after baseline
                if self.sample_count >= self.baseline_time * self.fs:
                    self._controller_iterate()

    # def _check_phase_trigger(self, phase_rad: float):
    #     """Check if phase meets target criteria and trigger pink noise accordingly"""
    #     # Block pink noise during baseline
    #     if self.sample_count < self.baseline_time * self.fs:
    #         return
    #     # Use configurable parameters from __init__
    #     target_phase = self.target_phase_rad
    #     phase_tolerance = self.phase_tolerance

    #     # Check if phase is within target range
    #     phase_diff = np.abs(phase_rad - np.array(target_phase))
    #     phase_diff = np.where(phase_diff > np.pi, 2 * np.pi - phase_diff, phase_diff)
    #     in_target_range = np.any(phase_diff <= phase_tolerance)

    #     # Trigger logic
    #     if in_target_range and not hasattr(self, "pink_noise_active"):
    #         # Initialize pink noise state if not already done
    #         self.pink_noise_active = False

    #     if in_target_range and not self.pink_noise_active:
    #         # Start pink noise
    #         self._start_pink_noise()
    #         self.pink_noise_active = True
    #         self.phase_trigger_count += 1
    #         if self.debug_mode:
    #             target_phase_str = ", ".join([f"{tp:.3f}" for tp in target_phase])
    #             print(
    #                 f"Phase trigger #{self.phase_trigger_count}: {phase_rad:.3f} rad (targets: [{target_phase_str}] ± {phase_tolerance:.3f})"
    #             )

    #     elif not in_target_range and self.pink_noise_active:
    #         # Stop pink noise
    #         self._stop_pink_noise()
    #         self.pink_noise_active = False
    #         if self.debug_mode:
    #             print(f"Phase exit: {phase_rad:.3f} rad")


    def _check_phase_trigger(self, phase_rad: float):
        """
        Trigger pink noise when the phase climbs past any target phase
        (strict upward crossing).  Hold the noise on for 50 ms.
        """

        # Skip the baseline period
        if self.sample_count < self.baseline_time * self.fs:
            self.last_phase = phase_rad
            return

        # One-off initialisation
        if not hasattr(self, "last_phase"):
            self.last_phase = phase_rad
        if not hasattr(self, "hold_counter"):
            self.hold_counter = 0
            self.pink_noise_active = False

        prev_phase = self.last_phase
        curr_phase = phase_rad

        # Unwrap so we can tell genuine upward motion even across the 2π→0 wrap
        # if curr_phase < prev_phase:
        #     curr_phase += 2 * np.pi

        crossing_detected = False
        for tp in np.atleast_1d(self.target_phase_rad):
            tp_unwrapped = tp
            if tp < prev_phase:          # bring target into the same ‘lap’
                tp_unwrapped += 2 * np.pi
            # crossed if target now lies between prev and current values
            if prev_phase < tp_unwrapped <= curr_phase:
                crossing_detected = True
                break

        # Store current phase for next call
        self.last_phase = phase_rad

        # 50 ms hold
        n_sample_hold = int(round(0.05 * self.fs))

        if crossing_detected:
            self._start_pink_noise()
            self.pink_noise_active = True
            self.hold_counter = n_sample_hold
            self.phase_trigger_count += 1
            if self.debug_mode:
                print(f"Phase trigger #{self.phase_trigger_count}: {phase_rad:.3f} rad")

        # Keep noise on for the required number of samples
        if self.pink_noise_active:
            self.hold_counter -= 1
            if self.hold_counter <= 0:
                self._stop_pink_noise()
                self.pink_noise_active = False
                if self.debug_mode:
                    print("Pink noise stopped")

    def _start_pink_noise(self):
        """Set pink noise volume to desired level (start noise)."""
        try:
            self.send_command(f"audio_pink_volume {self.pink_noise_volume}", False)
            self.send_command("audio_pink_play")
            
            # Record the start event with current sample timestamp
            current_time = self.sample_count * self.ts
            self.pink_noise_events.append((current_time, 'start'))
            
            if self.debug_mode:
                print("Pink noise volume set to ON")
        except Exception as e:
            if self.debug_mode:
                print(f"Error starting pink noise: {e}")

    def _stop_pink_noise(self):
        """Set pink noise volume to zero (stop noise)."""
        try:
            if self.sample_count >= self.baseline_time * self.fs:
                self.send_command("audio_pink_volume 0.3", False)
            else:
                self.send_command("audio_pink_volume 0.0", False)

            
            self.send_command("audio_pink_play")
            
            # Record the stop event with current sample timestamp
            current_time = self.sample_count * self.ts
            self.pink_noise_events.append((current_time, 'stop'))
            
            if self.debug_mode:
                print("Pink noise volume set to OFF")
        except Exception as e:
            if self.debug_mode:
                print(f"Error stopping pink noise: {e}")

    def _update_plotting_buffers(self, eeg_filtered: np.ndarray):
        # Always update buffers
        self.simple_eeg_buffer[:-1] = self.simple_eeg_buffer[1:]
        self.simple_eeg_buffer[-1] = eeg_filtered

        self.plot_update_counter += 1

    def run_session(
        self,
        duration_seconds: int = None,
        team_num: int = None,
        subject_num: int = None,
    ):
        """Run a complete EEG recording session"""
        if duration_seconds is None:
            duration_seconds = self.sampling_duration_secs

        try:
            # Initialize real-time plotting if enabled
            if self.enable_real_time_plotting:
                plt.ion()
                fig, axs = plt.subplots(
                    3,
                    1,
                    figsize=(12, 10),
                    sharex=True,
                    gridspec_kw={"height_ratios": [2, 1, 1]},
                )
                fig.suptitle("Real-time EEG Monitor")

                # EEG subplot
                ax_eeg = axs[0]
                ax_eeg.set_title("Real-time Filtered EEG (Last 4 seconds)", fontsize=9)
                ax_eeg.set_ylabel("Voltage (V)")
                ax_eeg.grid(True)
                x_samples = np.arange(0, 4, step=1 / self.fs)
                lines_eeg = []
                colors = ["blue", "red", "green"]
                for i in range(3):
                    (line,) = ax_eeg.plot(
                        x_samples,
                        np.zeros(1000),
                        color=colors[i],
                        linewidth=0.5,
                        label=["Fp1", "Fpz", "Fp2"][i],
                    )
                    lines_eeg.append(line)
                ax_eeg.legend()

                # Amplitude subplot
                ax_amp = axs[1]
                ax_amp.set_title("Instantaneous Amplitude")
                ax_amp.set_ylabel("Amplitude (V)")
                ax_amp.grid(True)
                (line_amp,) = ax_amp.plot(
                    x_samples,
                    np.zeros(1000),
                    color="purple",
                    label="Instantaneous α-amp",
                )
                (line_alpha_avg,) = ax_amp.plot(
                    x_samples,
                    np.zeros(1000),
                    color="blue",
                    linestyle="--",
                    linewidth=1,
                    label="Rolling 1s avg α-amp",
                )
                ax_amp.legend()

                # Enhanced Phase subplot with pink noise events and target phase
                ax_phase = axs[2]
                ax_phase.set_title("Instantaneous Phase with Events")
                ax_phase.set_ylabel("Phase (rad)")
                ax_phase.set_xlabel("Time (s)")
                ax_phase.set_ylim([0, 2 * np.pi])
                ax_phase.grid(True)
                
                # Phase line
                (line_phase,) = ax_phase.plot(x_samples, np.zeros(1000), color="orange", label="Instantaneous phase")
                
                # Target phase line
                (line_target_phase,) = ax_phase.plot(
                    x_samples, 
                    np.full(1000, np.nan), 
                    color="black", 
                    linestyle="-", 
                    linewidth=2, 
                    label="Target phase"
                )
                
                # Initialize empty lists for event lines (will be updated dynamically)
                pink_noise_start_lines = []
                pink_noise_stop_lines = []
                target_phase_points = []
                
                ax_phase.legend()

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)

            # Start logging
            logfilename = "Team_" + str(team_num) + "_sid_" + str(subject_num) + ".txt"
            log_path = self.start_logging(logfilename)

            # Setup audio
            self.setup_audio()

            # Start streaming
            self.start_streaming()

            # Start serial reader thread
            self.session_start_time = time.time()
            reader_thread = threading.Thread(target=self._serial_reader_thread)
            reader_thread.start()

            # Run for specified duration
            print("=== DATA ACQUISITION STARTED ===")
            print(f"Recording for {duration_seconds} seconds...")

            start_time = time.time()
            last_plot_update = 0
            plot_update_interval = 0.1  # Update every 0.5 seconds (2 Hz)
            last_progress = 0

            while time.time() - start_time < duration_seconds:
                elapsed = time.time() - start_time

                if self.enable_real_time_plotting:
                    if elapsed - last_plot_update >= plot_update_interval:
                        time_ax = self.ts * np.linspace(
                            self.sample_count - 1000, self.sample_count, 1000
                        )

                        # EEG
                        for i in range(3):
                            lines_eeg[i].set_ydata(self.simple_eeg_buffer[:, i])
                            lines_eeg[i].set_xdata(time_ax)

                        # Amplitude
                        line_amp.set_ydata(self.inst_amp_buffer)
                        line_amp.set_xdata(time_ax)
                        line_alpha_avg.set_ydata(self.avg_alpha_amp_buffer)
                        line_alpha_avg.set_xdata(time_ax)

                        # Phase
                        line_phase.set_ydata(self.inst_phase_buffer)
                        line_phase.set_xdata(time_ax)
                        
                        # Target phase
                        line_target_phase.set_ydata(self.target_phase_buffer)
                        line_target_phase.set_xdata(time_ax)

                        # Set x-axis limits to scroll with data
                        xmin = time_ax[0]
                        xmax = time_ax[-1]
                        ax_eeg.set_xlim([xmin, xmax])
                        ax_amp.set_xlim([xmin, xmax])
                        ax_phase.set_xlim([xmin, xmax])

                        # Clear old event lines that are outside the current window
                        for line in pink_noise_start_lines[:]:
                            if line.get_xdata()[0] < xmin:
                                line.remove()
                                pink_noise_start_lines.remove(line)
                        
                        for line in pink_noise_stop_lines[:]:
                            if line.get_xdata()[0] < xmin:
                                line.remove()
                                pink_noise_stop_lines.remove(line)
                                
                        for point in target_phase_points[:]:
                            if point.get_offsets()[0][0] < xmin:
                                point.remove()
                                target_phase_points.remove(point)

                        # Add new pink noise event lines within the current window
                        for event_time, event_type in self.pink_noise_events:
                            if xmin <= event_time <= xmax:
                                # Check if we already have a line for this event
                                line_exists = False
                                existing_lines = pink_noise_start_lines if event_type == 'start' else pink_noise_stop_lines
                                for existing_line in existing_lines:
                                    if abs(existing_line.get_xdata()[0] - event_time) < 0.001:  # Small tolerance
                                        line_exists = True
                                        break
                                
                                if not line_exists:
                                    if event_type == 'start':
                                        line = ax_phase.axvline(
                                            x=event_time, 
                                            color='red', 
                                            linestyle='--', 
                                            alpha=0.7, 
                                            linewidth=1,
                                            label='Pink noise ON' if not pink_noise_start_lines else ""
                                        )
                                        pink_noise_start_lines.append(line)
                                    elif event_type == 'stop':
                                        line = ax_phase.axvline(
                                            x=event_time, 
                                            color='blue', 
                                            linestyle='--', 
                                            alpha=0.7, 
                                            linewidth=1,
                                            label='Pink noise OFF' if not pink_noise_stop_lines else ""
                                        )
                                        pink_noise_stop_lines.append(line)

                        # Add new target phase update points within the current window
                        for update_time, phase_value in self.target_phase_updates:
                            if xmin <= update_time <= xmax:
                                # Check if we already have a point for this update
                                point_exists = False
                                for existing_point in target_phase_points:
                                    if abs(existing_point.get_offsets()[0][0] - update_time) < 0.001:
                                        point_exists = True
                                        break
                                
                                if not point_exists:
                                    point = ax_phase.scatter(
                                        update_time, 
                                        phase_value, 
                                        color='black', 
                                        marker='o', 
                                        s=30, 
                                        alpha=0.8,
                                        label='Target phase update' if not target_phase_points else ""
                                    )
                                    target_phase_points.append(point)

                        # Update legend if new elements were added
                        if (pink_noise_start_lines and not any('Pink noise ON' in str(h.get_label()) for h in ax_phase.get_legend().get_texts())) or \
                        (pink_noise_stop_lines and not any('Pink noise OFF' in str(h.get_label()) for h in ax_phase.get_legend().get_texts())) or \
                        (target_phase_points and not any('Target phase update' in str(h.get_label()) for h in ax_phase.get_legend().get_texts())):
                            ax_phase.legend()

                        # Auto-scale y-axes only occasionally
                        ax_eeg.relim()
                        ax_eeg.autoscale_view(scalex=False, scaley=True)
                        ax_amp.relim()
                        ax_amp.autoscale_view(scalex=False, scaley=True)

                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()

                        last_plot_update = elapsed

                # Progress updates every 5 seconds
                if elapsed - last_progress >= 5:
                    remaining = duration_seconds - elapsed
                    progress = 100 * elapsed / duration_seconds
                    print(
                        f"Progress: {elapsed:.1f}/{duration_seconds}s ({progress:.1f}% complete, {remaining:.1f}s remaining)"
                    )
                    last_progress = elapsed

                time.sleep(1)

            print("=== DATA ACQUISITION COMPLETED ===")

        finally:
            # Cleanup
            self.stop_event.set()
            self.stop_streaming()
            self.stop_logging()

            if "reader_thread" in locals():
                reader_thread.join(timeout=2)

            if self.enable_real_time_plotting:
                plt.ioff()

            print("\n=== SESSION SUMMARY ===")
            print(f"Total samples collected: {self.sample_count}")
            if duration_seconds > 0:
                print(
                    f"Duration: {self.sample_count / 250 :.2f} s (expected: {duration_seconds} s)"
                )
            print(f"Log file: {log_path}")

            # Automatically analyze results after session
            print("\nAnalyzing \recorded data...")
            self.analyze_log(log_path)

    def analyze_log(self, log_path: str = None):
        if log_path is None:
            log_path = self.abs_log_path

        if not log_path or not os.path.isfile(log_path):
            print("No valid log file found for analysis")
            return

        print(f"Analyzing: {log_path}")

        # Parse log file
        parsed_data = self._parse_log_file(log_path)

        if "eeg" not in parsed_data or len(parsed_data["eeg"]) == 0:
            print("No EEG data found in log file")
            return

        # Convert to numpy array
        eeg_array = np.array(parsed_data["eeg"])

        # Calculate time axis
        eeg_time = (eeg_array[:, 0] - eeg_array[0, 0]) / 1e6  # us to s
        sample_times = np.diff(eeg_time)

        ts = np.mean(sample_times)
        if ts < 0:
            ts = np.median(sample_times)

        fs = 1.0 / ts

        # Apply filters for post-processing
        raw_parsed_data = eeg_array.copy()
        filtered_eeg = eeg_array.copy()

        if self.bsf_on:
            filtered_eeg[:, 1:4] = filtfilt(
                self.b_bsf, self.a_bsf, filtered_eeg[:, 1:4], axis=0
            )
        if self.hpf_on:
            filtered_eeg[:, 1:4] = filtfilt(
                self.b_hpf, self.a_hpf, filtered_eeg[:, 1:4], axis=0
            )
        if self.bpf_on:
            filtered_eeg[:, 1:4] = filtfilt(
                self.b_bpf, self.a_bpf, filtered_eeg[:, 1:4], axis=0
            )

        # Time slice (trim edges)
        idx_start = int(round(self.time_start * fs))
        idx_end = int(round(self.time_end * fs))

        if idx_end > len(filtered_eeg):
            idx_end = len(filtered_eeg)

        eeg = filtered_eeg[idx_start:idx_end, 1:4]
        N = eeg.shape[0]
        t = ts * np.arange(N)

        # Create all analysis plots
        self._create_analysis_plots(
            eeg, t, fs, N, idx_start, idx_end, parsed_data, raw_parsed_data, ts
        )

        # Perform alpha power analysis
        # eeg_array: shape (N, 4) [timestamp, ch1, ch2, ch3]
        perc90, mean, std, stim_z, mean_z = analyze_alpha_power(
            eeg_array,
            fs=250,
            baseline_time=self.baseline_time,
            stimulation_time=self.stimulation_time,
            baseline_exclude=self.baseline_time / 2,
        )
        print("90th percentile z-scored alpha power (channels Fp1, Fpz, Fp2):", perc90)
        print("Highest value:", np.max(perc90))
        print("Mean z-scored alpha power (channels Fp1, Fpz, Fp2):", mean_z)

        print("Analysis complete!")

    def _parse_log_file(self, log_path: str) -> dict[str, list]:
        parsed_data = defaultdict(list)

        with open(log_path, "r") as fid:
            for raw_line in fid:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                match = re.search(
                    r"V \((\d+)\) data_log_(\w+): ([\d\.\-\s]+)", raw_line
                )
                if match:
                    rtos_timestamp = int(match.group(1))
                    data_type = match.group(2)
                    values = list(map(float, match.group(3).split()))

                    if values:
                        value_timestamp = values[0]
                        value_data = values[1:]

                        if data_type == "eeg":
                            value_data = np.array(value_data) * self.eeg_adc2volts
                            parsed_row = [value_timestamp] + value_data.tolist()

                        elif data_type == "accel":
                            value_data = np.array(value_data) * self.accel_raw2g
                            parsed_row = [value_timestamp] + value_data.tolist()

                        elif data_type == "inst_amp_phs":
                            try:
                                value_data[0] *= self.eeg_adc2volts
                                value_data[1] = degrees(value_data[1])
                                parsed_row = [value_timestamp] + value_data
                            except Exception:
                                parsed_row = [value_timestamp, 0, 0]

                        else:
                            continue

                        parsed_data[data_type].append(parsed_row)

        return parsed_data

    def _create_analysis_plots(
        self, eeg, t, fs, N, idx_start, idx_end, parsed_data, raw_parsed_data, ts
    ):
        # Plot 1: EEG Time Data - PostProcessed
        plt.figure("EEG Time Data - PostProcessed", figsize=(10, 6))
        plt.plot(t, eeg)
        plt.title("Time Data")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Voltage (V)")
        plt.legend(["Fp1", "Fpz", "Fp2"])
        plt.grid(True)

        # Plot 2: EEG Frequency Data
        plt.figure("EEG Freq Data", figsize=(10, 6))
        EEG_fft = fft(eeg, axis=0) / (N / 2)
        EEG_mag = np.abs(EEG_fft)
        EEG_pow = 20 * np.log10(EEG_mag + 1e-12)  # Avoid log(0)
        f = (fs / N) * np.arange(N)

        plt.plot(f, EEG_pow)
        plt.xlim([0, 65])
        plt.title("Frequency Data")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dBV)")
        plt.legend(["Fp1", "Fpz", "Fp2"])
        plt.grid(True)

        # Keep plots open
        plt.show()


def main():
    team_num = 2  # CHANGE TO YOUR TEAM NUMBER
    subject_num = 0  # CHANGE TO YOUR SUBJECT NUMBER

    # Recording parameters
    eeg_baseline = 60  # Baseline before stimulation starts

    stimulation_time = 2 * 60  # Time for which stimulation is active
    sampling_duration_secs = stimulation_time + eeg_baseline  # total recording time

    # List available ports
    ports = serial.tools.list_ports.comports()
    print("Available COM ports:")
    for port in ports:
        print(f"  {port.device} - {port.description}")

    # Setup port (modify as needed)
    # Check in /dev/tty* for Mac/Linux, or Device Manager for Windows
    # Replace with your actual COM port.

    # port = "/dev/ttyUSB0"  # Linux example
    # port = "/dev/tty.usbmodem14401"  # Mac example
    port = "/dev/tty.usbmodem101"  # Windows example

    # Create headband interface
    headband = ElemindHeadband(port, debug=True)

    # Configuration
    headband.enable_real_time_plotting = True  # Set to False for better performance

    # Set timing parameters
    headband.stimulation_time = stimulation_time
    headband.sampling_duration_secs = sampling_duration_secs
    headband.baseline_time = eeg_baseline
    headband.time_start = eeg_baseline
    headband.time_end = sampling_duration_secs

    # Configure closed-loop control parameters
    headband.target_phase_rad = [
        np.pi / 3,
        5 * np.pi / 6,
        4 * np.pi / 3,
        11 * np.pi / 6,
    ]  # Target phase: π radians (180 degrees)
    headband.phase_tolerance = 0.1  # Tolerance: ±0.1 radians (±5.7 degrees)
    headband.pink_noise_volume = 0.99  # Pink noise volume: 40%
    headband.pink_noise_fade_in_ms = 0  # Fade in: 200ms
    headband.pink_noise_fade_out_ms = 0  # Fade out: 200ms

    print("Closed-loop control configured:")
    # print(f"  Target phase: {headband.target_phase_rad:.2f} rad ({np.degrees(headband.target_phase_rad):.1f}°)")
    # print(f"  Tolerance: ±{headband.phase_tolerance:.2f} rad (±{np.degrees(headband.phase_tolerance):.1f}°)")
    # print(f"  Pink noise volume: {headband.pink_noise_volume*100:.0f}%")
    # print(f"  Fade in/out: {headband.pink_noise_fade_in_ms}ms")

    # Connect to headband
    if not headband.connect():
        print("Failed to connect to headband")
        return

    try:
        # Run session
        headband.run_session(sampling_duration_secs, team_num, subject_num)

    except KeyboardInterrupt:
        print("\nSession interrupted by user")
    except Exception as e:
        print(f"Error during session: {e}")
        import traceback

        traceback.print_exc()
    finally:
        headband.disconnect()


if __name__ == "__main__":
    main()
