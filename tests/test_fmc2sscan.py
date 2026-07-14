import matplotlib

from pipe_lens.utils import convert_parameters

matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.pipeline import Pipeline
from pipe_lens.transducer import Transducer
from pipe_lens.simulator.ray_physics.ray_physics import LossModel
from pipe_lens.simulator.ray_physics.ray_focus_lens import RayFocusThroughLens
from pipe_lens.simulator.optimizers.newton_optimizer import NewtonOptimizer
from pipe_lens.simulator.raytracer_scenario import RaytracerScenario
from pipe_lens.simulator.simulator import Simulator
from pipe_lens.simulator.utils import fmc2sscan, fmc2sscan_fft, fmc2sscan_numba
from pipe_lens.types import Solid, Fluid
from framework.post_proc import envelope
from pipe_lens.utils import api_polar, convert_time2radius

import time

#%% USER INPUT:

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
loss_model = LossModel(transmission=True, transmission_diffraction=True, reception_diffraction=True, attenuation=False)

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
physics = RayFocusThroughLens(acoustic_lens, pipeline, transducer)
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
theta_scatterer = np.radians(np.arange(30, 50 + 5, 5))

x_scatterer, z_scatterer = r_scatterer * np.sin(theta_scatterer), r_scatterer * np.cos(theta_scatterer)

amplitudes_v = [None] * len(theta_scatterer)
fwhm_horizontal = [None] * len(theta_scatterer)
fwhm_vertical = [None] * len(theta_scatterer)

for i in tqdm(range(len(theta_scatterer))):

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------

    t0 = time.time()

    fmc = simulation.simulate_scatterer(
        np.asarray([x_scatterer[i]]),
        np.asarray([z_scatterer[i]])
    )

    delay_law_in_shifts = np.round(
        delay_law * simulation.fs
    ).astype(np.int32)

    print(f"simulator      : {(time.time() - t0) * 1e3:.2f} ms")

    # --------------------------------------------------
    # Original implementation
    # --------------------------------------------------

    t0 = time.time()

    sscan_roll = fmc2sscan(
        fmc_sims=fmc,
        shifts_e=delay_law_in_shifts,
        shifts_r=delay_law_in_shifts,
        n_elem=transducer.num_elem
    )

    t_roll = (time.time() - t0) * 1e3

    print(f"fmc2sscan roll : {t_roll:.2f} ms")

    # --------------------------------------------------
    # FFT implementation
    # --------------------------------------------------

    t0 = time.time()

    sscan_fft = fmc2sscan_numba(
        fmc=fmc,
        shifts_e=delay_law_in_shifts,
        shifts_r=delay_law_in_shifts
    )[..., np.newaxis]

    t_fft = (time.time() - t0) * 1e3

    print(f"fmc2sscan fft  : {t_fft:.2f} ms")

    # --------------------------------------------------
    # Compare
    # --------------------------------------------------

    err_abs = np.max(np.abs(sscan_roll - sscan_fft))

    denom = np.max(np.abs(sscan_roll))
    err_rel = err_abs / denom if denom > 0 else 0.0

    print(f"max abs error  : {err_abs:.3e}")
    print(f"max rel error  : {err_rel:.3e}")

    # --------------------------------------------------
    # Plot first scatterer
    # --------------------------------------------------

    if i == 0:

        env_roll = envelope(sscan_roll[..., 0])
        env_fft = envelope(sscan_fft[..., 0])

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        ax[0].imshow(
            env_roll,
            aspect="auto",
            origin="lower"
        )
        ax[0].set_title("Original")

        ax[1].imshow(
            env_fft,
            aspect="auto",
            origin="lower"
        )
        ax[1].set_title("FFT")

        ax[2].imshow(
            np.abs(env_roll - env_fft),
            aspect="auto",
            origin="lower"
        )
        ax[2].set_title("Abs Difference")

        plt.tight_layout()
        plt.show()