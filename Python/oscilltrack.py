import numpy as np
class OscillTrack:
    """
    Real-time phase tracker inspired by McNamara et al. (2022).
    All calculations use float64 for clarity; if porting to embedded
    hardware replace the trig calls with a lookup table as described
    in the paper.
    """

    def __init__(self, fc_hz: float, fs_hz: float, g: float = 2 ** -4):
        """
        Parameters
        ----------
        fc_hz : centre frequency to track (Hz)
        fs_hz : sample rate (Hz)
        g     : error update coefficient (pass-band width controller)
        """
        self.fc = fc_hz
        self.fs = fs_hz
        self.g = g

        # Iterative state
        self.a = 0.0
        self.b = 0.0
        self.theta = 0.0                    # reference phase
        self.dtheta = 2 * np.pi * self.fc / self.fs  # phase step per sample

    def step(self, s: float):
        """
        Update the tracker with one new sample and return
        (phase_estimate_rad, amplitude_estimate).
        """
        sin_t = np.sin(self.theta)
        cos_t = np.cos(self.theta)

        # Complex estimate r_n  (eq. 2)  -------------------------------
        r_real = self.a * sin_t + self.b * cos_t
        r_imag = self.b * sin_t - self.a * cos_t

        # Error term D_n  (eq. 3) --------------------------------------
        D = s - r_real

        # Update coefficients a_{n+1}, b_{n+1}  (eq. 4) ----------------
        self.a += self.g * D * sin_t
        self.b += self.g * D * cos_t

        # Phase (CORDIC in hardware; arctan2 here)  --------------------
        phase = np.arctan2(r_imag, r_real)  # φ_n in paper
        amplitude = np.hypot(self.a, self.b)

        # Advance reference phase
        self.theta += self.dtheta
        if self.theta >= 2 * np.pi:         # cheap modulo 2π
            self.theta -= 2 * np.pi

        return phase, amplitude