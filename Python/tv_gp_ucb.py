# tv_gp_ucb.py – 1-D (phase-only) adaptation of tv_gp_ucb.py
# Author: (your name here)
# Licence: MIT
# -----------------------------------------------------------
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Tuple

# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------

@dataclass
class TestData:
    """Container for retained samples used to build the GP model."""
    time: NDArray[np.float64]            # (n,)
    inputs_samples: NDArray[np.float64]  # (n, 1) ────────────── phase only
    output_samples: NDArray[np.float64]  # (n,)


@dataclass
class GaussianProcess:
    """Mean and (co)variance evaluated on a 1-D grid of phases."""
    domain: NDArray[np.float64]          # (m, 1)
    mew: NDArray[np.float64]             # (m,)
    sd: NDArray[np.float64]              # (m,)


@dataclass
class TargetStim:
    """Next stimulus suggested by the acquisition function."""
    stim_variable: NDArray[np.float64]   # (1,) – phase only
    expected_sig: float
    expected_mew: float


# A global used by evaluate_information, mirroring the original C++
diff_exp: float = 0.0

# -------------------------------------------------------------------------
# Core functions
# -------------------------------------------------------------------------

def calc_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    sigfsquared: float,
    l: float,
) -> float:
    """Periodic kernel operating only on phase (x ∈ ℝ, measured in rad)."""
    a = x1[0] - x2[0]
    # Mason 2023: k(ϕ₁,ϕ₂) = σ² exp{-2 [sin(|ϕ₁−ϕ₂|/2) / l]² }
    return sigfsquared * np.exp(-2.0 * (np.sin(0.5 * abs(a)) / l) ** 2)


# -------------------------------------------------------------------------
# Synthetic response (test-bench only)
# -------------------------------------------------------------------------

def get_result(
    x: NDArray[np.float64],
    time_iteration: int,                # kept for API parity
    rng: np.random.Generator,
    noise_sd: float,
    test_variables: NDArray[np.float64],
) -> float:
    """Toy function: mixture of sinusoids plus Gaussian noise (1-D)."""
    mean = 0.0
    noise = rng.normal(mean, noise_sd)

    # test_variables:
    #   0: amplitude scale   1: mix
    #   2: phase offset      3: reserved for future use
    y = test_variables[0] * (
        test_variables[1] * np.sin(x[0] + test_variables[2])
        + (1.0 - test_variables[1]) * np.sin(2.0 * x[0] + test_variables[2])
    )
    return y + noise


# -------------------------------------------------------------------------
# Dataset initialisation & book-keeping helpers
# -------------------------------------------------------------------------

def initialise_data(
    length_dataset: int,
    x_min: float,
    x_max: float,
    noise_sd: float,
    test_variables: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> TestData:
    """Generate a uniform random set of phases within [x_min, x_max]."""
    if rng is None:
        rng = np.random.default_rng()

    inputs = rng.uniform(x_min, x_max, length_dataset).reshape(-1, 1)
    outputs = np.array(
        [get_result(inputs[i], 0, rng, noise_sd, test_variables)
         for i in range(length_dataset)]
    )
    time = np.zeros(length_dataset)
    return TestData(time=time, inputs_samples=inputs, output_samples=outputs)


def uniform_time_increase(dataset: TestData, time_increase: float) -> TestData:
    dataset.time += time_increase
    return dataset


# -------------------------------------------------------------------------
# Gaussian-process model construction
# -------------------------------------------------------------------------

def build_model(
    dataset: TestData,
    model_resolution: int,
    x_min: float,
    x_max: float,
    noise: float = 1.0,
) -> GaussianProcess:

    # Hyper-parameters – tweak as needed
    kernel_hyp_l = 0.6
    kernel_hyp_s = 3.0

    # Grid of candidate phases
    xs = np.linspace(x_min, x_max, model_resolution)
    domain = xs.reshape(-1, 1)

    n_data = dataset.inputs_samples.shape[0]
    bigk = np.empty((n_data, n_data))

    # Covariance of retained samples
    for i in range(n_data):
        for j in range(n_data):
            bigk[i, j] = calc_kernel(
                dataset.inputs_samples[i],
                dataset.inputs_samples[j],
                kernel_hyp_s, kernel_hyp_l
            )
            if i == j:
                bigk[i, j] += noise

    # Pre-compute inverse
    bigk_inv = np.linalg.inv(bigk)
    mu = np.empty(domain.shape[0])
    sd = np.empty(domain.shape[0])

    # Predictive mean/var for every x*
    for idx, x_star in enumerate(domain):
        k_star = np.array([
            calc_kernel(x_star, dataset.inputs_samples[j],
                        kernel_hyp_s, kernel_hyp_l)
            for j in range(n_data)
        ])
        k_star_star = calc_kernel(x_star, x_star, kernel_hyp_s, kernel_hyp_l)

        mu[idx] = k_star @ bigk_inv @ dataset.output_samples
        sd[idx] = k_star_star - k_star @ bigk_inv @ k_star

    return GaussianProcess(domain=domain, mew=mu, sd=sd)


# -------------------------------------------------------------------------
# Acquisition & replacement heuristics
# -------------------------------------------------------------------------

def acquisition_func(
    model: GaussianProcess,
    beta: float,
) -> TargetStim:
    """TV-GP-UCB: maximise μ + β·σ (note sign flip vs. minimise)."""
    objective = model.mew + beta * model.sd
    idx = int(objective.argmax())

    stim_variable = model.domain[idx].copy()
    expected_sig = float(model.sd[idx])
    expected_mew = float(model.mew[idx])
    return TargetStim(stim_variable, expected_sig, expected_mew)


def evaluate_information(
    expected_sig: float,
    expected_mew: float,
    true_change_trem_intensity: float,
    decay_k: float,
    K: float,
) -> float:
    """Update β (exploration weight) according to Mason 2023."""
    global diff_exp

    diff = true_change_trem_intensity - expected_mew

    limited_sig = np.clip(expected_sig, 0.01, 10.0)
    diff = np.clip(abs(diff), 1e-4, 100.0)

    diff_exp = K * (diff / limited_sig) ** 4 + decay_k * diff_exp
    return diff_exp


def replace_next_value(
    dataset: TestData,
    aquisition_target: TargetStim,
) -> int:
    """Choose which existing sample to overwrite (old + similar preferred)."""
    # Score = –|Δphase| – age
    diffs = np.abs(dataset.inputs_samples[:, 0] - aquisition_target.stim_variable[0])
    score = -diffs - dataset.time
    return int(score.argmax())  # index of sample to replace


# -------------------------------------------------------------------------
# Top-level helper replicating the original main loop (phase-only)
# -------------------------------------------------------------------------

def run_tv_gp_ucb(
    n_iterations: int,
    remember_n_stims: int,
    phase_rad_min: float,
    phase_rad_max: float,
    model_res_phase: int,
    decay_constant: float,
    acquisition_k: float,
    noise_sd: float,
    test_variables: NDArray[np.float64],
    rng: np.random.Generator | None = None,
):
    """Execute *n_iterations* of TV-GP-UCB using a 1-D phase domain."""
    if rng is None:
        rng = np.random.default_rng()

    retained = initialise_data(
        remember_n_stims,
        phase_rad_min,
        phase_rad_max,
        noise_sd,
        test_variables,
        rng,
    )

    beta = 1.0  # initial exploration weight
    for run_n in range(n_iterations):
        model = build_model(
            retained,
            model_res_phase,
            phase_rad_min,
            phase_rad_max,
        )

        next_stim = acquisition_func(model, beta)

        change = get_result(next_stim.stim_variable, run_n, rng, noise_sd, test_variables)

        beta = evaluate_information(
            next_stim.expected_sig,
            next_stim.expected_mew,
            change,
            decay_constant,
            acquisition_k,
        )

        idx = replace_next_value(retained, next_stim)

        # Overwrite the chosen slot and age the rest
        retained = uniform_time_increase(retained, 1.0)
        retained.inputs_samples[idx] = next_stim.stim_variable
        retained.output_samples[idx] = change
        retained.time[idx] = 0.0

    return retained, model
