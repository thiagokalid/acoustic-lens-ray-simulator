from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")

# Constants:
LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth
cm = 1 / 2.54
FIGSIZE = (5.23 * cm, 5 * cm)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_PATH = PROJECT_ROOT / "figures"
COMPUTE_SIM_RESULTS = True
COMPUTE_EXP_RESULTS = True
SAVE_FIGS = True

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

import numpy as np
import matplotlib.pyplot as plt

def plot_experimental_data(ax, thetas, mean_values, std_values, color='k', style='o-', label="_"):
    ax.plot(thetas, mean_values, style, color=color, lw=1, ms=2.25, label=label)
    ax.fill_between(
        x=thetas,
        y1=mean_values + std_values,
        y2=mean_values - std_values,
        alpha=.25, color=color
    )

    ax.set_xticks(np.arange(0, 40 + 5, 5))

sim_fwhm_w_focused_wave = np.load(project_root / "results" / "sim_fwhm_w_focused_wave.npy")
sim_fwhm_h_focused_wave = np.load(project_root / "results" / "sim_fwhm_h_focused_wave.npy")
sim_amplitude_focused_wave = np.load(project_root / "results" / "sim_amplitude_focused_wave.npy")

sim_fwhm_w_plane_wave = np.load(project_root / "results" / "sim_fwhm_w_plane_wave.npy")
sim_fwhm_h_plane_wave = np.load(project_root / "results" / "sim_fwhm_h_plane_wave.npy")
sim_amplitude_plane_wave = np.load(project_root / "results" / "sim_amplitude_plane_wave.npy")

mean_exp_fwhm_w_focused_wave = np.load(project_root / "results" / "mean_exp_fwhm_w_focused_wave.npy")
mean_exp_fwhm_h_focused_wave = np.load(project_root / "results" / "mean_exp_fwhm_h_focused_wave.npy")
mean_exp_amplitude_focused_wave = np.load(project_root / "results" / "mean_exp_amplitude_focused_wave.npy")

mean_exp_fwhm_w_plane_wave = np.load(project_root / "results" / "mean_exp_fwhm_w_plane_wave.npy")
mean_exp_fwhm_h_plane_wave = np.load(project_root / "results" / "mean_exp_fwhm_h_plane_wave.npy")
mean_exp_amplitude_plane_wave = np.load(project_root / "results" / "mean_exp_amplitude_plane_wave.npy")

std_exp_fwhm_w_focused_wave = np.load(project_root / "results" / "std_exp_fwhm_w_focused_wave.npy")
std_exp_fwhm_h_focused_wave = np.load(project_root / "results" / "std_exp_fwhm_h_focused_wave.npy")
std_exp_amplitude_focused_wave = np.load(project_root / "results" / "std_exp_amplitude_focused_wave.npy")

std_exp_fwhm_w_plane_wave = np.load(project_root / "results" / "std_exp_fwhm_w_plane_wave.npy")
std_exp_fwhm_h_plane_wave = np.load(project_root / "results" / "std_exp_fwhm_h_plane_wave.npy")
std_exp_amplitude_plane_wave = np.load(project_root / "results" / "std_exp_amplitude_plane_wave.npy")

#%%

exp_amplitude_maximum = np.max([
    mean_exp_amplitude_focused_wave,
    mean_exp_amplitude_plane_wave
])

sim_amplitude_maximum = np.max([
    sim_amplitude_plane_wave, sim_amplitude_focused_wave
])

# Transform horizontal fwhm to degrees:
sim_fwhm_w_plane_wave = np.degrees(sim_fwhm_w_plane_wave)
sim_fwhm_w_focused_wave = np.degrees(sim_fwhm_w_focused_wave)
mean_exp_fwhm_w_plane_wave = np.degrees(mean_exp_fwhm_w_plane_wave)
std_exp_fwhm_w_plane_wave = np.degrees(std_exp_fwhm_w_plane_wave)
mean_exp_fwhm_w_focused_wave = np.degrees(mean_exp_fwhm_w_focused_wave)
std_exp_fwhm_w_focused_wave = np.degrees(std_exp_fwhm_w_focused_wave)

# Transform vertical fwhm to mm:
sim_fwhm_h_plane_wave *= 1e3
sim_fwhm_h_focused_wave *= 1e3
std_exp_fwhm_h_plane_wave *= 1e3
mean_exp_fwhm_h_plane_wave *= 1e3
std_exp_fwhm_h_focused_wave *= 1e3
mean_exp_fwhm_h_focused_wave *= 1e3

# Normalize amplitude based on common references:
sim_amplitude_focused_wave = sim_amplitude_focused_wave / sim_amplitude_maximum * 100
sim_amplitude_plane_wave = sim_amplitude_plane_wave / sim_amplitude_maximum * 100
mean_exp_amplitude_focused_wave = mean_exp_amplitude_focused_wave / exp_amplitude_maximum * 100
std_exp_amplitude_focused_wave = std_exp_amplitude_focused_wave / exp_amplitude_maximum * 100
mean_exp_amplitude_plane_wave = mean_exp_amplitude_plane_wave / exp_amplitude_maximum * 100
std_exp_amplitude_plane_wave = std_exp_amplitude_plane_wave / exp_amplitude_maximum * 100

theta_scatterer = np.radians(np.arange(0, 40 + 1, 1))

#%%
thetas_deg = np.degrees(theta_scatterer)

fig, ax = plt.subplots(figsize=FIGSIZE)

# Horizontal Onda Plana:
ax.plot(np.degrees(theta_scatterer), sim_fwhm_w_plane_wave, '-', color='#ff1f23', lw=2)
plot_experimental_data(ax, thetas_deg, mean_exp_fwhm_w_plane_wave, std_exp_fwhm_w_plane_wave, style='x-', color='#FF1F5B')

# Horizontal Onda Focada:
ax.plot(np.degrees(theta_scatterer), sim_fwhm_w_focused_wave, '-', color='k', lw=2)
plot_experimental_data(ax, thetas_deg, mean_exp_fwhm_w_focused_wave, std_exp_fwhm_w_focused_wave, style='o-', color='k')

ax.set_xticks(np.arange(0, 40 + 5, 5))

ax.set_xlabel("Reflector angular position / (degrees)")
ax.set_ylabel(r"Width / (degrees)")

ax.grid()
plt.tight_layout()
plt.savefig(project_root/ "figures"/ "psf_fwhm_horizontal.pdf")
plt.show()

#%%%

fig, ax = plt.subplots(figsize=FIGSIZE)
# Vertical Onda Plana:
ax.plot(thetas_deg, sim_fwhm_h_plane_wave, '-', color='#FF1F5B', lw=2)
plot_experimental_data(ax, thetas_deg, mean_exp_fwhm_h_plane_wave, std_exp_fwhm_h_plane_wave, style='x-', color='#FF1F5B')

# Vertical Onda Focada:
ax.plot(thetas_deg, sim_fwhm_h_focused_wave, '-', color='k', lw=2)
plot_experimental_data(ax, thetas_deg, mean_exp_fwhm_h_focused_wave, std_exp_fwhm_h_focused_wave, style='o-', color='k')

ax.set_xticks(np.arange(0, 40 + 5, 5))
ax.set_xlabel("Reflector angular position / (degrees)")
ax.set_ylabel(r"Height / (mm)")

ax.grid()
plt.tight_layout()

plt.savefig(project_root/ "figures"/ "psf_fwhm_vertical.pdf")
plt.show()


#%%%

fig, ax = plt.subplots(figsize=FIGSIZE)

# Vertical Onda Plana:
ax.plot(np.degrees(theta_scatterer), sim_amplitude_plane_wave, '-', color='#FF1F5B', lw=2)
plot_experimental_data(ax, thetas_deg, mean_exp_amplitude_plane_wave, std_exp_amplitude_plane_wave, style='x-', color='#FF1F5B')


# Vertical Onda Focada:
ax.plot(np.degrees(theta_scatterer), sim_amplitude_focused_wave, '-', color='k', lw=2)
plot_experimental_data(ax, thetas_deg, mean_exp_amplitude_focused_wave, std_exp_amplitude_focused_wave, style='o-', color='k')

ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xticks(np.arange(0, 40 + 5, 5))

ax.set_xlabel("Reflector angular position / (degrees)")
ax.set_ylabel(r"Normalized Amplitude / (%)")

ax.grid()
plt.tight_layout()
plt.savefig(project_root/ "figures"/ "psf_normalized_amplitude.pdf")
plt.show()

#%%

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(np.degrees(theta_scatterer), mean_exp_amplitude_plane_wave, lw=2.5, color='k', label='y')
axs[0].plot(np.degrees(theta_scatterer), sim_amplitude_plane_wave, '-o', lw=.5, ms=3, label='y_ref', color='k')
axs[0].legend()
axs[0].grid()

axs[1].plot(np.degrees(theta_scatterer), (mean_exp_amplitude_plane_wave - sim_amplitude_plane_wave) / sim_amplitude_plane_wave * 100, '-ok', label='y - y_ref')
axs[1].grid()
plt.show()

# %% Error computation
import numpy as np


def relative_error(y, y_ref, std_ref=1, eps=1e-12):
    """Compute error normalized by the experimental standard deviation."""
    std_ref_safe = np.maximum(np.abs(std_ref), eps)
    return (y - y_ref) / std_ref_safe


def compute_metrics(y, y_ref, eps=1e-12):
    """Compute RMSE and MAPE (%) between simulated and experimental values."""
    y = np.asarray(y, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)

    valid = np.isfinite(y) & np.isfinite(y_ref)
    y = y[valid]
    y_ref = y_ref[valid]

    if y.size == 0:
        return np.nan, np.nan

    rmse = np.sqrt(np.mean((y - y_ref) ** 2))

    nonzero = np.abs(y_ref) > eps
    mape = (
        100 * np.mean(np.abs((y[nonzero] - y_ref[nonzero]) / y_ref[nonzero]))
        if np.any(nonzero)
        else np.nan
    )

    return rmse, mape


LIMIT_FOCUSED_WAVE = 30
LIMIT_PLANE_WAVE = 10

# Amplitude
pw_amplitude = compute_metrics(
    sim_amplitude_plane_wave[:LIMIT_PLANE_WAVE],
    mean_exp_amplitude_plane_wave[:LIMIT_PLANE_WAVE],
)

fw_amplitude = compute_metrics(
    sim_amplitude_focused_wave[:LIMIT_FOCUSED_WAVE],
    mean_exp_amplitude_focused_wave[:LIMIT_FOCUSED_WAVE],
)

# Width
pw_width = compute_metrics(
    sim_fwhm_w_plane_wave[:LIMIT_PLANE_WAVE],
    mean_exp_fwhm_w_plane_wave[:LIMIT_PLANE_WAVE],
)

fw_width = compute_metrics(
    sim_fwhm_w_focused_wave[:LIMIT_FOCUSED_WAVE],
    mean_exp_fwhm_w_focused_wave[:LIMIT_FOCUSED_WAVE],
)

# Height
pw_height = compute_metrics(
    sim_fwhm_h_plane_wave[:LIMIT_PLANE_WAVE],
    mean_exp_fwhm_h_plane_wave[:LIMIT_PLANE_WAVE],
)

fw_height = compute_metrics(
    sim_fwhm_h_focused_wave[:LIMIT_FOCUSED_WAVE],
    mean_exp_fwhm_h_focused_wave[:LIMIT_FOCUSED_WAVE],
)


# Print results
for name, pw, fw in [
    ("Amplitude", pw_amplitude, fw_amplitude),
    ("Width", pw_width, fw_width),
    ("Height", pw_height, fw_height),
]:
    print(f"{name}:")
    print(f"  Plane wave:   RMSE = {pw[0]:.4g}, MAPE = {pw[1]:.2f}%")
    print(f"  Focused wave: RMSE = {fw[0]:.4g}, MAPE = {fw[1]:.2f}%")