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
        self.enable_real_time_plotting = True

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

        # Closed-loop control parameters
        self.target_phase_rad = [np.pi/3, 5*np.pi/6, 4*np.pi/3, 11*np.pi/6]  # Target phase value (modify as needed)
        self.phase_tolerance = 0.1  # Tolerance around target phase (radians)
        self.pink_noise_volume = 0.5  # Pink noise volume (0.0 to 1.0)
        self.pink_noise_fade_in_ms = 100  # Fade in time in milliseconds
        self.pink_noise_fade_out_ms = 100  # Fade out time in milliseconds
        self.pink_noise_active = False  # Track if pink noise is currently playing
        self.phase_trigger_count = 0  # Count how many times phase trigger occurred

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
        """Configure audio output"""
        print("Configuring audio output...")

        # Set master volume
        self.send_command("audio_set_volume 128")  # 50% master volume

        # Example code to test audio briefly
        print("Testing audio output...")
        self.send_command("audio_play_test 440")
        time.sleep(1)
        self.send_command("audio_stop_test")
        print("Audio test complete.")
        # End of example code to test audio briefly

    def start_streaming(self):
        """Start EEG data streaming"""
        print("Starting EEG streaming...")

        # Set filter centre frequency for phase computation
        # self.send_command("therapy_enable_alpha_track 0")
        # self.send_command("echt_config_simple 10")

        # Enable streaming
        self.send_command("stream eeg 1")
        self.send_command("echt_start")
        self.send_command("audio_pink_volume 0")
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
        self.send_command("audio_pink_volume 1")
        self.send_command("audio_pink_fade_out 0")
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

        print(data)

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
        # phase_est, amp_est = self.tracker.step(sample)
        # self.inst_amp_buffer[:-1]   = self.inst_amp_buffer[1:]
        # self.inst_amp_buffer[-1]    = amp_est
        # self.inst_phase_buffer[:-1] = self.inst_phase_buffer[1:]
        # self.inst_phase_buffer[-1]  = phase_est % (2*np.pi)  # keep 0‒2π

        # END YOUR CLOSED LOOP CONTROL HERE

        # Progress reporting
        if self.sample_count % 1000 == 0:
            elapsed_time = self.sample_count / self.fs
            print(f"EEG samples: {self.sample_count} ({elapsed_time:.1f} seconds)")

        self.sample_count += 1

    def _process_inst_amp_phs(self, timestamp: float, inst_data: np.ndarray):
        if len(inst_data) >= 2:
            amp_volts = inst_data[0] * self.eeg_adc2volts
            # phase_rad = radians(inst_data[1]) if len(inst_data) > 1 else 0.0
            phase_rad = inst_data[1] if len(inst_data) > 1 else 0.0

            self.inst_amp_phs_data.append([timestamp, amp_volts, phase_rad])

            # CLOSED LOOP CONTROL: Trigger pink noise based on phase
            self._check_phase_trigger(phase_rad)

            # Update live buffers
            self.inst_amp_buffer[:-1] = self.inst_amp_buffer[1:]
            self.inst_amp_buffer[-1] = amp_volts
            self.inst_phase_buffer[:-1] = self.inst_phase_buffer[1:]
            self.inst_phase_buffer[-1] = phase_rad % (2 * np.pi)  # Ensure 0-2pi

    def _check_phase_trigger(self, phase_rad: float):
        """Check if phase meets target criteria and trigger pink noise accordingly"""
        # Use configurable parameters from __init__
        target_phase = self.target_phase_rad
        phase_tolerance = self.phase_tolerance

        # Check if phase is within target range
        phase_diff = np.abs(phase_rad - np.array(target_phase))
        phase_diff = np.where(phase_diff > np.pi, 2 * np.pi - phase_diff, phase_diff)
        in_target_range = np.any(phase_diff <= phase_tolerance)

        # Trigger logic
        if in_target_range and not hasattr(self, "pink_noise_active"):
            # Initialize pink noise state if not already done
            self.pink_noise_active = False

        if in_target_range and not self.pink_noise_active:
            # Start pink noise
            self._start_pink_noise()
            self.pink_noise_active = True
            self.phase_trigger_count += 1
            if self.debug_mode:
                target_phase_str = ", ".join([f"{tp:.3f}" for tp in target_phase])
                print(
                    f"Phase trigger #{self.phase_trigger_count}: {phase_rad:.3f} rad (targets: [{target_phase_str}] ± {phase_tolerance:.3f})"
                )

        elif not in_target_range and self.pink_noise_active:
            # Stop pink noise
            self._stop_pink_noise()
            self.pink_noise_active = False
            if self.debug_mode:
                print(f"Phase exit: {phase_rad:.3f} rad")

    def _start_pink_noise(self):
        """Start pink noise with specified parameters"""
        try:
            # Set volume and fade parameters using configurable values
            self.send_command(f"audio_pink_volume {self.pink_noise_volume}", False)
            self.send_command(f"audio_pink_fade_in {self.pink_noise_fade_in_ms}", False)
            self.send_command(
                f"audio_pink_fade_out {self.pink_noise_fade_out_ms}", False
            )

            # Start playing
            self.send_command("audio_pink_play", False)
            self.send_command("audio_pink_unmute", False)

            if self.debug_mode:
                print("Pink noise started")

        except Exception as e:
            if self.debug_mode:
                print(f"Error starting pink noise: {e}")

    def _stop_pink_noise(self):
        """Stop pink noise"""
        try:
            self.send_command("audio_pink_stop", False)
            if self.debug_mode:
                print("Pink noise stopped")

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
                (line_amp,) = ax_amp.plot(x_samples, np.zeros(1000), color="purple")

                # Phase subplot
                ax_phase = axs[2]
                ax_phase.set_title("Instantaneous Phase")
                ax_phase.set_ylabel("Phase (rad)")
                ax_phase.set_xlabel("Time (s)")
                ax_phase.set_ylim([0, 2 * np.pi])
                ax_phase.grid(True)
                (line_phase,) = ax_phase.plot(x_samples, np.zeros(1000), color="orange")

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

                        # Phase
                        line_phase.set_ydata(self.inst_phase_buffer)
                        line_phase.set_xdata(time_ax)

                        # Set x-axis limits to scroll with data
                        xmin = time_ax[0]
                        xmax = time_ax[-1]
                        ax_eeg.set_xlim([xmin, xmax])
                        ax_amp.set_xlim([xmin, xmax])
                        ax_phase.set_xlim([xmin, xmax])

                        # Auto-scale y-axes only occasionally
                        ax_eeg.relim()
                        ax_eeg.autoscale_view(scalex=False, scaley=True)
                        ax_amp.relim()
                        ax_amp.autoscale_view(scalex=False, scaley=True)
                        # ax_phase: y-limits fixed to [0, 2pi]

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
    eeg_basline = 60  # Baseline before stimulation starts
    stimulation_time = 2 * 60  # Time for which stimulation is active
    sampling_duration_secs = stimulation_time + eeg_basline  # total recording time

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
    port = "COM6"  # Windows example

    # Create headband interface
    headband = ElemindHeadband(port, debug=True)

    # Configuration
    headband.enable_real_time_plotting = True  # Set to False for better performance

    # Set timing parameters
    headband.stimulation_time = stimulation_time
    headband.sampling_duration_secs = sampling_duration_secs
    headband.baseline_time = eeg_basline
    headband.time_start = eeg_basline
    headband.time_end = sampling_duration_secs

    # Configure closed-loop control parameters
    headband.target_phase_rad = [
        np.pi / 3,
        5 * np.pi / 6,
        4 * np.pi / 3,
        11 * np.pi / 6,
    ]  # Target phase: π radians (180 degrees)
    headband.phase_tolerance = 0.1  # Tolerance: ±0.1 radians (±5.7 degrees)
    headband.pink_noise_volume = 1  # Pink noise volume: 40%
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
