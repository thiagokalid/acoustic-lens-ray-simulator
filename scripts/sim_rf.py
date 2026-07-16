from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

import matplotlib
matplotlib.use("TkAgg")

LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pipe_lens.utils import convert_parameters
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.pipeline import Pipeline
from pipe_lens.transducer import Transducer
from pipe_lens.simulator.ray_physics.ray_physics import LossModel
from pipe_lens.simulator.ray_physics.ray_focus_lens import RayFocusThroughLens
from pipe_lens.simulator.optimizers.newton_optimizer import NewtonOptimizer
from pipe_lens.simulator.raytracer_scenario import RaytracerScenario
from pipe_lens.simulator.simulator import Simulator
from pipe_lens.simulator.utils import fmc2sscan_numba
from pipe_lens.types import Solid, Fluid
from framework.post_proc import envelope
from pipe_lens.utils import api_polar, convert_time2radius


#%% USER INPUT:

LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth
PLOT_SSCANS = True

# Materials related:
steel = Solid(cp=5900, cs=3200, density=7850)
aluminium = Solid(cp=6400.88, cs=3130, density=2700)
water = Fluid(cp=1483, density=1000)

# Transducer related:
pitch = .5e-3
num_elem = 64
element_gap = .1e-3
central_freq = 5e6
bw = .4
bwr = -6
pulse_type = "gaussian"

# Pipe related:
outer_radius = 70e-3
pipe_material = steel
wall_thickness = 20e-3
center = (0., 0.)

# Acoustic-lens related:
wedge_material = aluminium
coupling_medium = water
d = 2 * 105e-3
# h0 = 90e-3
h0, _, _ = convert_parameters(c=d/2, wc=4.1e-3, R=outer_radius, v1=wedge_material.cp, v2=coupling_medium.cp)
alpha_max = np.radians(50)
alpha_0 = 0

# Delay-law related:
focus_alpha_step = np.radians(.5)
focus_alpha_max= np.radians(50)
focus_alpha_min = np.radians(-50)
focus_radial_pos = 50e-3

# Configure simulation:
loss_model = LossModel(transmission=False, transmission_diffraction=False, reception_diffraction=False, attenuation=False)

#%%
# Create pipeline:
pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center)
pipeline_shifted = Pipeline(pipe_material, outer_radius, wall_thickness, (0, 0))

# Create acoustic lens:
acoustic_lens = AcousticLens(wedge_material, coupling_medium, d, alpha_max, alpha_0, h0, outer_radius)

# Create transducer:
transducer = Transducer(pitch, num_elem, element_gap, central_freq, bw, bwr, pulse_type, center=(0., acoustic_lens.d))

# Define location for the focuses:
focus_alpha_pos = np.arange(focus_alpha_min, focus_alpha_max + focus_alpha_step, focus_alpha_step)

#%% Simulation:

xfocus, zfocus = focus_radial_pos * np.sin(focus_alpha_pos), focus_radial_pos * np.cos(focus_alpha_pos)

# Compute delay-law:
physics = RayFocusThroughLens(acoustic_lens, pipeline, transducer, loss_model)
optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=15, eps=1e-4)
raytracer = RaytracerScenario(physics, optimizer)

opt_rays = raytracer.solve(xfocus, zfocus)
roundtrip_tofs = physics.compute_tofs(opt_rays)
tofs_roi = roundtrip_tofs.diagonal(axis1=1, axis2=2).T / 2
delay_law = np.mean(tofs_roi, axis=0) - tofs_roi

physics = RayFocusThroughLens(acoustic_lens, pipeline_shifted, transducer, loss_model)
optimizer = NewtonOptimizer(alpha_max=alpha_max,  alpha_min=acoustic_lens.alpha_min, max_iter=15, eps=1e-4)
raytracer = RaytracerScenario(physics, optimizer)
simulation = Simulator(gate_start=60e-6, gate_end=80e-6, fs=62.5e6, raytracer=raytracer)

# Conver the time-axix to radius:
radius_span = convert_time2radius(
    tspan=simulation.tspan,
    t_outer=acoustic_lens.t_outer,
    t_inner=acoustic_lens.t_outer + 2 * pipeline.wall_width/pipeline.material.cp,
    cl_surf=pipeline.material.cp,
    cl_upper=acoustic_lens.coupling_medium.cp,
    cl_bottom=acoustic_lens.coupling_medium.cp,
    outer_r=pipeline.outer_radius
)

# Define API scatterers grid:
r_scatterer = 60e-3
theta_scatterer = np.radians(np.arange(0, 50 + 1, 1))

x_scatterer, z_scatterer = r_scatterer * np.sin(theta_scatterer), r_scatterer * np.cos(theta_scatterer)

amplitudes_v = [None] * len(theta_scatterer)
fwhm_horizontal = [None] * len(theta_scatterer)
fwhm_vertical = [None] * len(theta_scatterer)

for i in tqdm(range(len(theta_scatterer))):
    fmc = simulation.simulate_scatterer(np.asarray([x_scatterer[i]]), np.asarray([z_scatterer[i]]))
    delay_law_in_shifts = np.round(delay_law * simulation.fs).astype(np.int32)

    # assuming linearity, we create an s-scan from the fmc based on the emission and reception delay-law
    sscan = fmc2sscan_numba(
        fmc=fmc,
        shifts_e=delay_law_in_shifts,
        shifts_r=delay_law_in_shifts
    )[..., np.newaxis]

    sscan_env = envelope(sscan[..., 0], axis=0)

    if i == 0:
        ref_amplitude = sscan_env.max()

    corners_in = [
        (65e-3, theta_scatterer[i] - np.radians(8)), # north-west
        (55e-3, theta_scatterer[i] + np.radians(8))  # south-east
    ]

    _, _, pixels_above_threshold, amplitudes_v[i], fwhm_vertical[i], fwhm_horizontal[i] =\
        api_polar(sscan_env, radius_span, focus_alpha_pos, corners_in, drawSquare=True)

    if PLOT_SSCANS:
        fig, axs = plt.subplots(figsize=(LATEX_LINEWIDTH * .4, 2.4))
        sscan_db = np.log10(sscan_env / ref_amplitude + 1E-6)
        plt.imshow(sscan_db, vmin=-3, vmax=0, extent=[np.degrees(focus_alpha_min), np.degrees(focus_alpha_max), simulation.gate_end * 1e6, simulation.gate_start * 1e6],cmap="inferno", aspect='auto')
        plt.axhline(y=acoustic_lens.t_outer * 1e6, color='lime', ls='--', lw=1.5)
        plt.axhline(y=(acoustic_lens.t_outer + 2 * pipeline.wall_width/pipeline.material.cp) * 1e6, color='lime', ls='--', lw=1.5)

        delta_thickness = outer_radius - r_scatterer

        plt.plot(np.degrees(theta_scatterer[i]), (acoustic_lens.t_outer + 2 * delta_thickness/pipeline.material.cp) * 1e6, 'o', markerfacecolor="None", color='lime', ms=15)
        plt.xlabel("Angular position / (degrees)")
        plt.ylabel(r"Time / $\mathrm{(\mu s)}$")
        plt.ylim([70, 60])
        plt.xticks(np.arange(-50, 50 + 25, 25))
        plt.tight_layout()
        plt.savefig(project_root / "figures" / f"sample_sscan_{np.degrees(theta_scatterer[i]):.0f}deg.pdf")
        plt.show(block=True)


#%%
PLOT_FOR_PAPER = True

# if PLOT_FOR_PAPER:
# else:
fig, ax = plt.subplots(figsize=(LATEX_LINEWIDTH * .55, 2.4))
if not PLOT_FOR_PAPER:
    plt.suptitle("Resolution measured by considering punctual scatterer.")
    ax.set_title("Full Width at Half Maximum (FWHM)")
ax.plot(np.degrees(theta_scatterer), np.asarray(fwhm_horizontal) / np.asarray(fwhm_horizontal).max(), ':ok', label="Horizontal FWHM", ms=5)
# ax.plot(-np.degrees(theta_scatterer), np.asarray(fwhm_horizontal) * 1e3, '-or')
# ax.plot(np.degrees(theta_scatterer), np.asarray(fwhm_vertical)/fwhm_vertical[0] * 100, '-ob', label="Vertical FWHM")
ax.set_xlabel("Scatterer angular location / (degrees)")
ax.set_ylabel("Length / (mm)")
# ax = plt.twinx()
# ax.plot(np.degrees(theta_scatterer), np.asarray(fwhm_vertical) * 1e3, '-ob')


if not PLOT_FOR_PAPER:
    ax.legend()

ax.grid()
plt.tight_layout()
plt.savefig(project_root / "figures" / "norm_fwhm_sim.pdf")
plt.show()

fig, ax = plt.subplots(figsize=(LATEX_LINEWIDTH * .55, 2.4))
ax.plot(np.degrees(theta_scatterer), np.asarray(amplitudes_v) / np.asarray(amplitudes_v).max(), ':ok', ms=5)
# ax.plot(-np.degrees(theta_scatterer), np.asarray(amplitudes_v), '-or')
if not PLOT_FOR_PAPER:
    ax.set_title("Amplitudes")
ax.set_xlabel("Scatterer angular location / (degrees)")
ax.set_ylabel("Amplitude")
ax.grid()
plt.tight_layout()
plt.savefig(project_root / "figures" / "norm_amplitude_sim.pdf")
plt.show()