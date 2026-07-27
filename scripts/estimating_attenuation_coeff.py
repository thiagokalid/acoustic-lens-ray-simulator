from framework import file_m2k
import numpy as np
from pathlib import Path
from typing import Tuple
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"

def apply_window(ascan, time_grid, start: float, window_samples: int):
    ibeg = np.searchsorted(time_grid, start)
    iend = ibeg + window_samples
    return ascan[ibeg:iend]

material = "steel" # or aluminium

# This estimation method assumes standard properties of water at a given temperature.
# We will assume that the water is at 25 celsisu, and the properties are:
c0 = 1483 # water propagation speed in [m/s]
alpha0_norm = 25.3e-15 # Normalized attenuation coefficient (i.e., α₀/f²) in [s² Np/ m];
rho0 = 997 # water density in [kg/m^3]

# Gates for the reference experiement (only with water):
window_duration = 5e-6
fs = 125e6 # it is assumed to be a 125 MHz sampling frequency acquisitions
window_samples = int(window_duration * fs)
f = np.fft.fftfreq(window_samples, d=1 / fs) # frequencies values of each beam in Hz
ref_echo_start = [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6]

if material == "steel":
    window_start = {
            "WaterReflection_Ra": [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6],
            "MaterialReflection_Rb": [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6],
            "DirectOnewayEcho_T": [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6],
            "DirectTwowayEcho_Tprime": [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6],
    }
elif material == "aluminium":
    window_start = {
            "WaterReflection_Ra": [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6],
            "MaterialReflection_Rb": [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6],
            "DirectOnewayEcho_T": [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6],
            "DirectTwowayEcho_Tprime": [0.e-6, 0.e-6, 0.e-6, 0.e-6, 0.e-6],
    }
else:
    raise ValueError(f"Unknown material {material}")

acquisition_configs = {
    "tp_transd": "gaussian",
    "central_freq": 5,
}


for rep in range(0):
    # Reference inspection (only with water):
    data_ref = file_m2k.read(str(data_dir / "pulse_echo_water.m2k"), sel_shots=rep, **acquisition_configs)

    # Extract meaningful signals that will be used on the estimator:
    ascan_ref = data_ref.ascan_data[:, 0]
    time_grid_ref = data_ref.time_grid[:, 0]
    T0 = apply_window(ascan_ref, time_grid_ref, ref_echo_start[rep], window_samples)

    # Measurement inspection:
    data_insp = file_m2k.read(str(data_dir / f"pulse_echo_{material}.m2k"), sel_shots=rep, **acquisition_configs)

    # Extract meaningful signals that will be used on the estimator:
    ascan_tx = data_insp.ascan_data[:, 0]
    ascan_rx = data_insp.ascan_data[:, 1]
    time_grid = data_insp.time_grid[:, 0]

    # Window each signal according to pre-defined gates:
    Ra = apply_window(ascan_tx, time_grid, window_start["WaterReflection_Ra"][rep], window_samples)
    Rb = apply_window(ascan_tx, time_grid, window_start["MaterialReflection_Rb"][rep], window_samples)
    T = apply_window(ascan_tx, time_grid, window_start["DirectOnewayEcho_T"][rep], window_samples)
    Tprime = apply_window(ascan_tx, time_grid, window_start["DirectTwowayEcho_Tprime"][rep], window_samples)

    # Evaluate the reference (water) complex wavenumber:
    k0 = (2 * np.pi * f) / c0 + 1j * (alpha0_norm * f**2) / 2

    # Evaluate the reflection coefficient based on the spectra of each signal:
    r10_squared = (Tprime / T) / ((Tprime / T) - (Rb / Ra))
    r10 = np.sqrt(r10_squared)
    r01 = -r10

    # Estimate the specimen propagation speed:
    num = 2 * np.unwrap(np.angle(T) - np.angle(T0) - np.angle(1 - r10_squared))
    den = np.unwrap(np.angle(Tprime) - np.angle(T) - np.angle(r10_squared))
    c = c0 * (1 - num / den)

    # Estimate the specimen width:
    l = c/(2 * np.pi * f) * np.unwrap(np.angle(Tprime) - np.angle(T) - np.angle(r10_squared))

    # Estimate the specimen medium attenuation coefficient:
    alpha = (alpha0_norm * f**2) - 2/l * (np.log((np.abs(T) / np.abs(T0))) - np.log(np.abs(1 - np.square(r01))))

    # Evaluate the complex wavenumber concept:
    k1 = (2 * np.pi * f) / c + 1j * alpha/2

    # Estimate the specimen medium density:
    rho = rho0 * k1 / k0 * (1 + r01) / (1 - r01)