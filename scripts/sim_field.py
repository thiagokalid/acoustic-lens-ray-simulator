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
from pipe_lens.simulator.ray_physics.ray_oneway import RayOnewayThroughLens
from pipe_lens.simulator.ray_physics.ray_focus_lens import RayFocusThroughLens
from pipe_lens.simulator.optimizers.newton_optimizer import NewtonOptimizer
from pipe_lens.simulator.raytracer_scenario import RaytracerScenario
from pipe_lens.simulator.simulator import Simulator
from pipe_lens.types import Solid, Fluid


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
LENS_NAME = "v2-0-0_LL-Longa"
d = 2 * 105e-3
# h0 = 90e-3
h0, _, _ = convert_parameters(c=d/2, wc=4.1e-3, R=outer_radius, v1=wedge_material.cp, v2=coupling_medium.cp)
alpha_max = np.radians(50)
alpha_0 = 0

# Delay-law related:
focus_angle = np.radians(30)
focus_radius = 60e-3

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


#%% Simulation:

xfocus, zfocus = focus_radius * np.sin(focus_angle), focus_radius * np.cos(focus_angle)

# Compute delay-law:
physics = RayFocusThroughLens(acoustic_lens, pipeline, transducer)
optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=30, eps=1e-4)
raytracer = RaytracerScenario(physics, optimizer)

opt_rays = raytracer.solve(xfocus, zfocus)
roundtrip_tofs = physics.compute_tofs(opt_rays)
tofs_roi = roundtrip_tofs.diagonal(axis1=1, axis2=2).T / 2
delay_law = np.mean(tofs_roi, axis=0) - tofs_roi

# Compute tensile-based field:
physics = RayOnewayThroughLens(acoustic_lens, pipeline_shifted, transducer, loss_model)
optimizer = NewtonOptimizer(alpha_max=alpha_max,  alpha_min=acoustic_lens.alpha_min, max_iter=15, eps=1e-4)
raytracer = RaytracerScenario(physics, optimizer)
simulation = Simulator(gate_start=60e-6, gate_end=80e-6, fs=62.5e6, raytracer=raytracer)

# Define API scatterers grid:
ang_step = 4
radius_step = 1e-3

radius_span = np.arange(50e-3, 70e-3, 1e-3)
ang_span = np.radians(np.arange(-45, 45 + ang_step, ang_step))

RR, TT = np.meshgrid(radius_span, ang_span)
XX, ZZ = RR * np.sin(TT), RR * np.cos(TT)

field = np.zeros_like(XX)

for (i, j), x in tqdm(
    np.ndenumerate(XX),
    total=XX.size,
    desc="Simulating scatterers"
):
    z = ZZ[i, j]

    fmc_t = simulation.simulate_scatterer(
            np.asarray([x]),
            np.asarray([z])
    )
    fmc_w = np.fft.fft(fmc_t, axis=0)
    shift = np.exp(-1j * np.outer(simulation.omega, delay_law))
    fmc_w = fmc_w[..., 0] * shift
    ith_field = np.real(np.fft.ifft(fmc_w, axis=0))

    field[i, j] = np.sqrt(np.sum(ith_field**2))

# normalize the tension field just for plotting purposes:
field = (field - field.min()) / (field.max() - field.min())

#%%

fig, axs = plt.subplots(figsize=(12, 8))
axs.pcolormesh(XX * 1e3, ZZ * 1e3, field, cmap='RdBu')

# plot pipe and lens overlay:
axs.plot(pipeline.xout * 1e3, pipeline.zout * 1e3, '--', color='black')
axs.plot(pipeline.xint * 1e3, pipeline.zint * 1e3, '--', color='black')
axs.plot(acoustic_lens.xlens * 1e3, acoustic_lens.zlens * 1e3, '--', color='black')

axs.set_aspect('equal')
axs.set_xlabel("x-axis / (mm)")
axs.set_ylabel("z-axis / (mm)")
axs.set_ylim([25, 90])