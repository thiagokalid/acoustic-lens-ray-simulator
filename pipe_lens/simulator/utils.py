from numpy import sqrt

import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"
from numba import njit, prange

import numpy as np

from numpy import ndarray

__all__ = ['dist', 'fmc_sim_kernel_gaussian', 'numba_gausspulse', 'fmc2sscan']

FLOAT = np.float32

def dist(x1, y1, x2, y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

@njit(parallel=True)
def fmc_sim_kernel_gaussian(tspan: ndarray, tofs: ndarray, amplitudes, n_elem: int, fc_Hz: float, bw: float) -> ndarray:
    ascan_data = np.zeros(shape=(len(tspan), n_elem, n_elem), dtype=FLOAT)

    for combined_idx in prange(n_elem * n_elem):
        idx_e = combined_idx // n_elem
        idx_r = combined_idx % n_elem
        tof = tofs[idx_e, idx_r]

        if tof == -1.0:
            continue

        # Ideal gausspulse shifted (spatial impulse simulator):
        ascan_data[:, idx_e, idx_r] = numba_gausspulse(tspan - tof, fc_Hz, bw)

        # Amplitude after propagation (transmission + reception):
        ascan_data[:, idx_e, idx_r] *= amplitudes[idx_e, idx_r]

    return ascan_data

@njit(fastmath=True)
def numba_gausspulse(t, fc_Hz, bw, bwr=-6):
    ref = pow(10.0, bwr / 20.0)
    a = -(np.pi * fc_Hz * bw) ** 2 / (4.0 * np.log(ref))
    return np.real(np.exp(-a * np.power(t, 2)) * np.exp(1j * 2 * np.pi * fc_Hz * t))


def fmc2sscan(fmc_sims: ndarray, shifts_e, shifts_r, n_elem: int):
    # From a given FMC apply the delays and compute the Summed-Scan (S-Scan):
    if len(fmc_sims.shape) == 3:
        num_sims = 1
        fmc_sims = fmc_sims[..., np.newaxis]
    elif len(fmc_sims.shape) == 4:
        num_sims = fmc_sims.shape[2]
    else:
        raise ValueError('Invalid fmc_sims dimension')

    num_elems = fmc_sims.shape[1]
    num_samples = fmc_sims.shape[0]
    num_laws = shifts_r.shape[1]

    sscan = np.zeros(shape=(num_samples, num_laws, num_sims), dtype=FLOAT)
    # signal_recepted_by_focus = np.zeros(shape=(num_samples, num_laws, num_sims), dtype=FLOAT)
    for scan_idx in range(num_laws): #
        # Delay And Sum in emission:
        shift_e = shifts_e[:, scan_idx]
        rolled_fmc = np.zeros_like(fmc_sims)
        for i in range(n_elem):
            rolled_fmc[:, i, :, :] = np.roll(fmc_sims[:, i, :, :], int(shift_e[i]), axis=0)
        das_emission = np.sum(rolled_fmc, axis=1)
        # signal_recepted_by_focus[:, scan_idx, :] = np.sum(das_emission, axis=1)

        # Delay And Sum in reception:
        shift_r = shifts_r[:, scan_idx]
        das = np.zeros_like(das_emission)
        for i in range(num_elems):
            das[:, i, :] = np.roll(das_emission[:, i, :], int(shift_r[i]), axis=0)
        ascan = np.sum(das, axis=1)
        sscan[:, scan_idx, :] = ascan

    # return sscan, signal_recepted_by_focus
    return sscan


@njit(parallel=True, fastmath=True)
def fmc2sscan_numba(
    fmc,
    shifts_e,
    shifts_r,
):
    """
    fmc:
        (samples, tx, rx)

    shifts_e:
        (tx, laws)

    shifts_r:
        (rx, laws)

    returns:
        (samples, laws)
    """

    n_samples = fmc.shape[0]
    n_tx = fmc.shape[1]
    n_rx = fmc.shape[2]
    n_laws = shifts_e.shape[1]

    sscan = np.zeros((n_samples, n_laws), dtype=fmc.dtype)

    for law in prange(n_laws):

        for t in range(n_samples):

            acc = 0.0

            for tx in range(n_tx):

                te = t - shifts_e[tx, law]

                if te < 0:
                    te += n_samples
                elif te >= n_samples:
                    te -= n_samples

                for rx in range(n_rx):

                    tr = te - shifts_r[rx, law]

                    if tr < 0:
                        tr += n_samples
                    elif tr >= n_samples:
                        tr -= n_samples

                    acc += fmc[tr, tx, rx]

            sscan[t, law] = acc

    return sscan

def angle_between(a, b):
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)

    cos_theta = dot / (norm_a * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)

def snell(incident, normal, c1, c2):
    incident = np.asarray(incident, dtype=float)
    normal = np.asarray(normal, dtype=float)

    single_input = incident.ndim == 1

    if single_input:
        incident = incident[None, :]
        normal = normal[None, :]

    # Normalize vectors
    i = incident / np.linalg.norm(incident, axis=1, keepdims=True)
    n = normal / np.linalg.norm(normal, axis=1, keepdims=True)

    mu = c2/c1

    #
    cos_theta1 = -np.sum(n * i, axis=1)

    # Replace n by its negative whenever yields cos_theta1 < 0:
    n[cos_theta1 < 0] = -n[cos_theta1 < 0]

    # Recompute cos(theta1):
    cos_theta1 = -np.sum(n * i, axis=1)

    radicand = 1 - mu**2 * (1 - cos_theta1)**2
    cos_theta2 = np.sqrt(radicand)

    v_refract = mu * i + np.tile(mu * cos_theta1 - cos_theta2, reps=(2, 1)).T * n
    v_reflect = i + 2 * np.tile(cos_theta1, reps=(2, 1)).T * n

    T = v_refract
    R = v_reflect

    # h = v1 / v2
    #
    # c = -np.sum(normal * incident, axis=1)
    #
    # #
    # ni = np.sqrt(1 + h ** 2 * (c ** 2 - 1))
    # omega = h * c
    #
    # beta_refrac = -omega + np.sign(c) * ni
    # beta_reflect = -omega - np.sign(c) * ni




    # T = h * incident - beta_refrac[:, None] * normal
    # R = h * incident - beta_reflect[:, None] * normal




    theta_i = angle_between(incident, normal)
    theta_r = angle_between(R, normal)
    theta_t = angle_between(T, normal)

    return T, R, theta_i, theta_t, theta_r

def line2circle_projection(xline, zline, slope_line, r):
    # Analytical formula for intesection between a line and a circle
    dx = np.cos(slope_line)
    dz = np.sin(slope_line)

    a = dx ** 2 + dz ** 2
    b = 2 * (xline * dx + zline * dz)
    c = xline ** 2 + zline ** 2 - r ** 2

    delta = b ** 2 - 4 * a * c
    x1 = (-b + np.sqrt(delta)) / (2 * a)
    x2 = (-b - np.sqrt(delta)) / (2 * a)

    # First and second intersection points:
    p1 = np.asarray([xline + x1 * dx, zline + x1 * dz], dtype=np.float32)
    p2 = np.asarray([xline + x2 * dx, zline + x2 * dz], dtype=np.float32)

    # If there is no intersection:
    p1 = p1 if np.linalg.norm(p1) > 0 else None
    p2 = p2 if np.linalg.norm(p2) > 0 else None

    return p1, p2