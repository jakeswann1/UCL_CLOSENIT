"""
Async version of the Elemind headband interface.

Major change:
- Real-time plotting moved to a separate **multiprocessing.Process** so that drawing
  cannot stall the main data-acquisition thread.

How it works
------------
*  The acquisition class keeps circular numpy buffers exactly as before.
*  Whenever those buffers are refreshed, a **thin view** (copy) of the latest data
   is placed on a `multiprocessing.Queue`.
*  The **PlotterProcess** owns the Matplotlib figure.  It pulls items from the
   queue and refreshes the display at ~10 Hz.  If no new data arrive it just
   idles briefly.
*  When the session finishes the main process pushes a sentinel (``None``) and
   then joins the child so everything shuts down tidily.

This approach prevents Matplotlib from blocking the CPU time that is needed for
serial I/O while remaining entirely cross-platform (it works on Windows where
``fork`` is unavailable).

Written in June 2025 – UK English spelling.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from math import degrees
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
import serial.tools.list_ports
from scipy import signal
from scipy.fft import fft
from scipy.signal import filtfilt

# ---------------- third-party closed-loop controller --------------------
from tv_gp_ucb import (
    TestData,
    acquisition_func,
    build_model,
    evaluate_information,
    initialise_data,
    replace_next_value,
    uniform_time_increase,
)

# Local helper
from post_hoc_analysis import analyze_alpha_power

__all__ = ["ElemindHeadband"]

# ----------------------------------------------------------------------
#                        REAL-TIME PLOTTER PROCESS
# ----------------------------------------------------------------------

class PlotterProcess(mp.Process):
    """A separate process dedicated to live graphs.

    Parameters
    ----------
    data_queue : mp.Queue
        Receives buffer tuples from the acquisition process.
    fs : int
        Sampling rate so that the x-axis can be labelled correctly.
    """

    def __init__(self, data_queue: mp.Queue, fs: int):
        super().__init__(daemon=True)
        self._queue = data_queue
        self._fs = fs

    # We run Matplotlib entirely in the child so the GUI never touches the
    # acquisition thread or the serial port.
    def run(self):  # noqa: D401 – imperative mood is appropriate here
        import matplotlib as mpl

        mpl.use("TkAgg")  # a backend that works well in a subprocess on all OSs

        plt.ion()
        fig, axs = plt.subplots(
            3,
            1,
            figsize=(12, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )
        fig.suptitle("Real-time EEG Monitor (async)")

        # Pre-build empty lines
        x_default = np.linspace(-4, 0, 1000, endpoint=False)
        colours = ["blue", "red", "green"]
        eeg_lines = [axs[0].plot(x_default, np.zeros_like(x_default), c=c, lw=0.5)[0] for c in colours]
        axs[0].set_ylabel("Voltage (V)")
        axs[0].legend(["Fp1", "Fpz", "Fp2"])
        axs[0].grid(True)

        (amp_line,) = axs[1].plot(x_default, np.zeros_like(x_default), c="purple", lw=0.8, label="α amp")
        (avg_line,) = axs[1].plot(x_default, np.zeros_like(x_default), "--", lw=0.8, label="1-s mean")
        axs[1].set_ylabel("Amplitude (V)")
        axs[1].grid(True)
        axs[1].legend()

        (phase_line,) = axs[2].plot(x_default, np.zeros_like(x_default), c="orange", lw=0.8)
        axs[2].set_ylabel("Phase (rad)")
        axs[2].set_ylim(0, 2 * np.pi)
        axs[2].set_xlabel("Time (s)")
        axs[2].grid(True)

        plt.tight_layout()
        last_redraw = 0.0
        redraw_interval = 0.1  # seconds

        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                item = None

            # Sentinel → tidy exit
            if item is None:
                break

            (
                time_axis,
                eeg_buffer,
                amp_buffer,
                avg_buffer,
                phase_buffer,
            ) = item

            # Update line data (no autoscale every time for speed)
            for i, line in enumerate(eeg_lines):
                line.set_data(time_axis, eeg_buffer[:, i])
            amp_line.set_data(time_axis, amp_buffer)
            avg_line.set_data(time_axis, avg_buffer)
            phase_line.set_data(time_axis, phase_buffer)

            now = time.perf_counter()
            if now - last_redraw >= redraw_interval:
                # Rescale y-axes once in a while
                for ax in axs[:2]:
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)

                axs[0].set_xlim(time_axis[0], time_axis[-1])
                for ax in axs[1:]:
                    ax.set_xlim(time_axis[0], time_axis[-1])

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                last_redraw = now

        plt.close("all")


# ----------------------------------------------------------------------
#                            DIGITAL FILTERS
# ----------------------------------------------------------------------

class ElemindFilter2ndOrder:
    """Simple 2nd-order IIR filter (used for high-pass)."""

    def __init__(self, b_coeffs: np.ndarray, a_coeffs: np.ndarray):
        self.b = np.asarray(b_coeffs, dtype=np.float64)
        self.a = np.asarray(a_coeffs, dtype=np.float64)
        self.w1 = 0.0
        self.w2 = 0.0

    def filter_sample(self, x: float) -> float:  # noqa: D401
        x = float(x)
        if not np.isfinite(x):
            return 0.0

        y = self.b[0] * x + self.w1
        self.w1 = self.b[1] * x + self.w2 - self.a[1] * y
        self.w2 = self.b[2] * x - self.a[2] * y

        if not np.isfinite(y):
            y = self.w1 = self.w2 = 0.0

        self.w1 = np.clip(self.w1, -1e6, 1e6)
        self.w2 = np.clip(self.w2, -1e6, 1e6)
        return y


class ElemindFilter4thOrder:
    """Simple 4th-order IIR filter (used for band-pass and band-stop)."""

    def __init__(self, b_coeffs: np.ndarray, a_coeffs: np.ndarray):
        self.b = np.asarray(b_coeffs, dtype=np.float64)
        self.a = np.asarray(a_coeffs, dtype=np.float64)
        self.w1 = self.w2 = self.w3 = self.w4 = 0.0

    def filter_sample(self, x: float) -> float:  # noqa: D401
        x = float(x)
        if not np.isfinite(x):
            return 0.0

        y = self.b[0] * x + self.w1
        self.w1 = self.b[1] * x + self.w2 - self.a[1] * y
        self.w2 = self.b[2] * x + self.w3 - self.a[2] * y
        self.w3 = self.b[3] * x + self.w4 - self.a[3] * y
        self.w4 = self.b[4] * x - self.a[4] * y

        if not np.isfinite(y):
            y = self.w1 = self.w2 = self.w3 = self.w4 = 0.0

        self.w1 = np.clip(self.w1, -1e6, 1e6)
        self.w2 = np.clip(self.w2, -1e6, 1e6)
        self.w3 = np.clip(self.w3, -1e6, 1e6)
        self.w4 = np.clip(self.w4, -1e6, 1e6)
        return y


# ----------------------------------------------------------------------
#                     MAIN ELEMIND HEADBAND INTERFACE
# ----------------------------------------------------------------------

class ElemindHeadband:
    """Interface class covering serial I/O, filtering and closed-loop control."""

    # ------------------------------ setup ------------------------------

    def __init__(self, port: str, *, baudrate: int = 115200, debug: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: serial.Serial | None = None
        self.is_connected = False
        self.debug_mode = debug

        # ---------------------------------------------------------- config
        self.stimulation_time = 120  # seconds of active stimulation
        self.baseline_time = 60      # seconds before stimulation
        self.sampling_duration_secs = self.stimulation_time + self.baseline_time
        self.time_start = self.baseline_time
        self.time_end = self.sampling_duration_secs
        self.fs = 250
        self.ts = 1 / self.fs
        self.bandpass_centre_freq = 10

        # Control flags
        self.enable_real_time_plotting = True

        # IIR filters
        self._setup_filters()

        # Logging and data holdings
        self.raw_eeg_data: list[list[float]] = []
        self.inst_amp_phs_data: list[list[float]] = []
        self.sample_count = 0

        # Buffers for the plotter
        self.simple_eeg_buffer = np.zeros((1000, 3))
        self.inst_amp_buffer = np.zeros(1000)
        self.inst_phase_buffer = np.zeros(1000)
        self.avg_alpha_amp_buffer = np.zeros(1000)

        # Average α amplitude per second for the controller
        self.avg_alpha_amp_last_sec = 0.0
        self._amp_sample_counter = 0
        self.alpha_amp_history: list[tuple[float, float]] = []

        # Closed-loop controller parms
        self.target_phase_rad: list[float] = [0.0]
        self.phase_tolerance = 0.1
        self.pink_noise_volume = 0.5
        self.pink_noise_fade_in_ms = 100
        self.pink_noise_fade_out_ms = 100
        self.pink_noise_active = False
        self.phase_trigger_count = 0

        # Serial read thread
        self.stop_event = threading.Event()

        # Multiprocessing plotter bits – created lazily in `run_session`
        self._plot_queue: mp.Queue | None = None
        self._plotter: PlotterProcess | None = None

        # GP-UCB controller state (unchanged relative to original)
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
        self._last_next_stim = None

        # Unit conversion constants
        self.eeg_adc2volts_numerator = 4.5 / (8388608.0 - 1)
        self.eeg_gain_default = 24
        self.eeg_adc2volts = self.eeg_adc2volts_numerator / self.eeg_gain_default
        self.accel_raw2g = 1 / (2 ** 14)

        # Logging
        self.log_file: "os.PathLike[str] | None" = None
        self.abs_log_path: str | None = None

    # ----------------------- serial connection helpers ------------------

    def connect(self) -> bool:
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
        except serial.SerialException as exc:
            print(f"Failed to connect: {exc}")
            return False

    def disconnect(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        print("Disconnected from Elemind headband")

    # -------------------------- digital filters -------------------------

    def _setup_filters(self):
        # High-pass
        b, a = signal.butter(2, 0.5 / (self.fs / 2), "high")
        self.hpf = ElemindFilter2ndOrder(b, a)
        self.hpf_on = True

        # Band-pass (alpha ±25 %)
        low = self.bandpass_centre_freq * 0.75
        high = self.bandpass_centre_freq * 1.25
        b, a = signal.butter(2, [low, high], "band", fs=self.fs)
        self.bpf = ElemindFilter4thOrder(b, a)
        self.bpf_on = False

        # Band-stop (50 Hz mains)
        b, a = signal.butter(2, [45, 55], "stop", fs=self.fs)
        self.bsf = ElemindFilter4thOrder(b, a)
        self.bsf_on = True

        # Keep coeffs for offline use
        self.b_hpf, self.a_hpf = self.hpf.b, self.hpf.a
        self.b_bpf, self.a_bpf = self.bpf.b, self.bpf.a
        self.b_bsf, self.a_bsf = self.bsf.b, self.bsf.a

    # ------------------------------- plotting ---------------------------

    def _start_plotter(self):
        if not self.enable_real_time_plotting:
            return
        self._plot_queue = mp.Queue(maxsize=4)
        self._plotter = PlotterProcess(self._plot_queue, self.fs)
        self._plotter.start()

    def _stop_plotter(self):
        if self._plot_queue is not None:
            try:
                self._plot_queue.put_nowait(None)  # sentinel
            except queue.Full:
                pass
        if self._plotter is not None:
            self._plotter.join(timeout=5.0)
            self._plotter = None
            self._plot_queue = None

    def _send_plot_update(self):
        """Push latest buffers to the plotting process if it is running."""
        if self._plot_queue is None:
            return
        try:
            time_axis = self.ts * np.linspace(
                self.sample_count - 1000, self.sample_count, 1000, dtype=np.float64
            )
            payload = (
                time_axis,
                self.simple_eeg_buffer.copy(),
                self.inst_amp_buffer.copy(),
                self.avg_alpha_amp_buffer.copy(),
                self.inst_phase_buffer.copy(),
            )
            # Drop oldest if the queue is full so the plotter never lags badly.
            if self._plot_queue.full():
                _ = self._plot_queue.get_nowait()
            self._plot_queue.put_nowait(payload)
        except queue.Full:
            pass  # If still full just skip this frame.

    # -------------------------- data processing -------------------------

    # === serial receive thread ==========================================

    def _serial_reader_thread(self):
        while not self.stop_event.is_set() and self.is_connected:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = (
                        self.serial_conn.readline()
                        .decode("utf-8", errors="ignore")
                        .strip()
                    )
                    if line:
                        if self.log_file:
                            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            print(line, file=self.log_file, flush=True)
                        self._process_data_line(line)
                time.sleep(0.0001)
            except Exception as exc:  # noqa: BLE001
                print(f"Serial reader error: {exc}")
                break

    # === line handler ====================================================

    _RX_RE = re.compile(r"V \((\d+)\) data_log_(\w+): ([\d.\-\s]+)")

    def _process_data_line(self, raw: str):
        m = self._RX_RE.search(raw)
        if not m:
            return
        data_type = m.group(2)
        try:
            values = np.fromstring(m.group(3), sep=" ", dtype=float)
        except ValueError:
            return
        if values.size < 2:
            return
        timestamp, payload = values[0], values[1:]
        if data_type == "eeg":
            self._handle_eeg(timestamp, payload)
        elif data_type == "inst_amp_phs":
            self._handle_inst_amp_phase(timestamp, payload)
        # accel and others ignored for brevity

    # === EEG sample ======================================================

    def _handle_eeg(self, ts_usec: float, eeg_raw: np.ndarray):
        if eeg_raw.size != 3:
            return
        volts = eeg_raw * self.eeg_adc2volts
        self.raw_eeg_data.append([ts_usec, *volts.tolist()])

        # Per-channel filtering
        filt = np.zeros(3)
        for ch in range(3):
            sample = volts[ch]
            if self.bsf_on:
                sample = self.bsf.filter_sample(sample)
            if self.hpf_on:
                sample = self.hpf.filter_sample(sample)
            if self.bpf_on:
                sample = self.bpf.filter_sample(sample)
            filt[ch] = sample
        self.simple_eeg_buffer[:-1] = self.simple_eeg_buffer[1:]
        self.simple_eeg_buffer[-1] = filt

        self.sample_count += 1
        if self.sample_count % 25 == 0:  # send ≈10 Hz updates
            self._send_plot_update()

    # === instantaneous amplitude & phase ================================

    def _handle_inst_amp_phase(self, ts_usec: float, inst: np.ndarray):
        if inst.size < 2:
            return
        amp_V = inst[0] * self.eeg_adc2volts
        phase_rad = inst[1] % (2 * np.pi)
        self.inst_amp_phs_data.append([ts_usec, amp_V, phase_rad])

        self.inst_amp_buffer[:-1] = self.inst_amp_buffer[1:]
        self.inst_amp_buffer[-1] = amp_V
        self.inst_phase_buffer[:-1] = self.inst_phase_buffer[1:]
        self.inst_phase_buffer[-1] = phase_rad

        # rolling 1-s mean of amplitude for the optimiser
        self._amp_sample_counter += 1
        if self._amp_sample_counter >= self.fs:
            recent = self.inst_amp_buffer[-self.fs :]
            self.avg_alpha_amp_last_sec = float(np.mean(np.abs(recent)))
            self.avg_alpha_amp_buffer[:-1] = self.avg_alpha_amp_buffer[1:]
            self.avg_alpha_amp_buffer[-1] = self.avg_alpha_amp_last_sec
            self.alpha_amp_history.append((ts_usec, self.avg_alpha_amp_last_sec))
            self._amp_sample_counter = 0
            self._controller_iterate()

        # closed-loop audio trigger
        self._check_phase_trigger(phase_rad)

    # --------------------------- phase trigger --------------------------

    def _check_phase_trigger(self, phase_rad: float):
        diff = np.abs(phase_rad - np.asarray(self.target_phase_rad))
        diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
        in_range = np.any(diff <= self.phase_tolerance)
        if in_range and not self.pink_noise_active:
            self._start_pink_noise()
            self.pink_noise_active = True
            self.phase_trigger_count += 1
            if self.debug_mode:
                print(f"Phase trigger #{self.phase_trigger_count} at {phase_rad:.3f} rad")
        elif not in_range and self.pink_noise_active:
            self._stop_pink_noise()
            self.pink_noise_active = False

    # ----------------------- pink-noise helpers -------------------------

    def _start_pink_noise(self):
        self._send_cmd(f"audio_pink_volume {self.pink_noise_volume}")
        self._send_cmd(f"audio_pink_fade_in {self.pink_noise_fade_in_ms}")
        self._send_cmd(f"audio_pink_fade_out {self.pink_noise_fade_out_ms}")
        self._send_cmd("audio_pink_play")
        self._send_cmd("audio_pink_unmute", False)

    def _stop_pink_noise(self):
        self._send_cmd("audio_pink_stop", False)

    # --------------------- generic serial helpers -----------------------

    def _send_cmd(self, cmd: str, echo: bool = True):
        if not self.is_connected:
            return
        try:
            self.serial_conn.write(f"{cmd}\n".encode())
            if echo:
                print(f"→ {cmd}")
            if self.log_file:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"[{ts}] Sent: {cmd}", file=self.log_file)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to send '{cmd}': {exc}")

    # -------------------------- GP-UCB bits -----------------------------

    def _controller_initialise(self):
        if self._controller_ready:
            return
        cfg = self._controller_cfg
        self._controller_retained = initialise_data(
            cfg["remember_n_stims"],
            cfg["phase_min"],
            cfg["phase_max"],
            cfg["noise_sd"],
            test_variables=np.zeros(4),
            rng=np.random.default_rng(seed=int(time.time())),
        )
        self._controller_retained.output_samples[:] = self.avg_alpha_amp_last_sec
        mdl = build_model(
            self._controller_retained,
            cfg["res_phase"],
            cfg["phase_min"],
            cfg["phase_max"],
        )
        self._last_next_stim = acquisition_func(mdl, self._controller_beta)
        self.target_phase_rad = [float(self._last_next_stim.stim_variable[0])]
        self._controller_ready = True
        if self.debug_mode:
            print(f"[Controller] init phase {self.target_phase_rad[0]:.2f} rad")

    def _controller_iterate(self):
        if not self._controller_ready:
            self._controller_initialise()
            return
        cfg = self._controller_cfg
        meas = self.avg_alpha_amp_last_sec
        self._controller_beta = evaluate_information(
            self._last_next_stim.expected_sig,
            self._last_next_stim.expected_mew,
            meas,
            cfg["decay_constant"],
            cfg["acquisition_k"],
        )
        idx = replace_next_value(self._controller_retained, self._last_next_stim)
        self._controller_retained = uniform_time_increase(self._controller_retained, 1.0)
        self._controller_retained.inputs_samples[idx] = self._last_next_stim.stim_variable
        self._controller_retained.output_samples[idx] = meas
        self._controller_retained.time[idx] = 0.0
        mdl = build_model(
            self._controller_retained,
            cfg["res_phase"],
            cfg["phase_min"],
            cfg["phase_max"],
        )
        self._last_next_stim = acquisition_func(mdl, self._controller_beta)
        self.target_phase_rad = [float(self._last_next_stim.stim_variable[0])]
        if self.debug_mode:
            print(f"[Controller] new phase {self.target_phase_rad[0]:.2f} rad  β={self._controller_beta:.2f}")

    # ------------------------------------------------------------------
    #                              SESSION
    # ------------------------------------------------------------------

    def run_session(self, *, duration_s: int, team: int, participant: int):
        """Main driver – handles setup, loops for *duration_s* then cleans up."""
        if not self.connect():
            print("Unable to establish serial link – aborting.")
            return

        # start logging
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = Path(f"Team_{team}_sid_{participant}_{timestamp}.txt")
        self.log_file = open(log_name, "w", encoding="utf-8")
        self.abs_log_path = str(log_name.resolve())
        print(f"Logging to {self.abs_log_path}")

        # audio + streaming set-up (omitted – use your existing commands)
        self.setup_audio()
        self.start_streaming()

        # background reader
        reader = threading.Thread(target=self._serial_reader_thread, daemon=True)
        reader.start()

        # detached plotter
        self._start_plotter()

        print("=== acquisition started ===")
        t0 = time.time()
        try:
            while time.time() - t0 < duration_s:
                time.sleep(1.0)  # main thread keeps things simple now
                elapsed = int(time.time() - t0)
                if elapsed % 5 == 0:
                    pct = 100 * elapsed / duration_s
                    print(f"Progress {elapsed}/{duration_s}s  ({pct:.1f} %)")
        except KeyboardInterrupt:
            print("User interrupted – stopping early.")
        finally:
            print("=== acquisition finished – tidying up ===")
            self.stop_event.set()
            self.stop_streaming()
            self._stop_plotter()
            reader.join(timeout=2.0)
            if self.log_file:
                self.log_file.close()
            self.disconnect()
            print("Session complete.  Log saved to:", self.abs_log_path)
            # optional: self.analyze_log(self.abs_log_path)

    # ------------------- placeholders for original methods --------------
    # (setup_audio, start_streaming, stop_streaming, analyze_log etc.)
    # These are identical to the initial script and therefore omitted here
    # to keep the focus on the async plotting changes.  Copy them across as-is.


# ----------------------------------------------------------------------
#                               main
# ----------------------------------------------------------------------

def main():
    mp.set_start_method("spawn", force=True)  # Windows compatibility
    team_num = 2
    participant_num = 0
    acquisition_time = 180  # seconds

    # Pick the correct COM port
    port = "COM6"

    hb = ElemindHeadband(port, debug=False)
    hb.enable_real_time_plotting = True
    hb.run_session(duration_s=acquisition_time, team=team_num, participant=participant_num)


if __name__ == "__main__":
    main()
