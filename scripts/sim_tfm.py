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
from matplotlib.ticker import FuncFormatter
matplotlib.use("TkAgg")

from scipy.signal import hilbert

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

import numpy as np
import matplotlib.pyplot as plt

from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.pipeline import Pipeline
from pipe_lens.transducer import Transducer
from pipe_lens.simulator.ray_physics.ray_physics import LossModel
from pipe_lens.simulator.ray_physics.ray_focus_lens import RayFocusThroughLens
from pipe_lens.simulator.optimizers.newton_optimizer import NewtonOptimizer
from pipe_lens.simulator.raytracer_scenario import RaytracerScenario
from pipe_lens.simulator.simulator import Simulator
from pipe_lens.types import Solid, Fluid
from pipe_lens.utils import convert_parameters

LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth
LOG_CTE = 1E-6 # constant for log-scale transformation

if __name__ == "__main__":
    #%% USER INPUT:

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
    h0, _, _ = convert_parameters(c=d / 2, wc=4.1e-3, R=outer_radius, v1=wedge_material.cp, v2=coupling_medium.cp)
    alpha_max = np.radians(49.94)
    alpha_0 = 0


    #%%
    # Create pipeline:
    pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center=(0, 0))
    pipeline_shifted = Pipeline(pipe_material, outer_radius, wall_thickness, center=(0, 0))

    # Create acoustic lens:
    acoustic_lens = AcousticLens(wedge_material, coupling_medium, d, alpha_max, alpha_0, h0, outer_radius)

    # Create transducer:
    transducer = Transducer(pitch, num_elem, element_gap, central_freq, bw, bwr, pulse_type, center=(0., acoustic_lens.d))

    # Define the scatterer position:
    radius_scatterer = 60e-3
    theta_scatterer = np.radians(40)

    #%% Simulation:

    xfocus, zfocus = radius_scatterer * np.sin(theta_scatterer) + pipeline_shifted.xcenter, radius_scatterer * np.cos(theta_scatterer) + pipeline_shifted.zcenter

    #%%
    # Define imaging ROI:
    xspan = np.arange(-50, 50, .25) * 1e-3
    zspan = np.arange(0, 100, .25) * 1e-3

    # is_tof_already_computed = project_root / "data" / "tof" / ...

    XX, ZZ = np.meshgrid(xspan, zspan)

    # check which pixels are within the pipe wall region
    is_within_pipe = np.logical_and(np.sqrt(XX**2 + ZZ**2) < pipeline.outer_radius, np.sqrt(XX**2 + ZZ**2) > pipeline.inner_radius)
    print(f"Number of imaging pixels: {np.count_nonzero(is_within_pipe)}")

    # Configure simulation:
    loss_model = LossModel(transmission=False, transmission_diffraction=False, reception_diffraction=False, attenuation=False)

    # Finding the round-trip TOF for each point from the region of interest
    physics = RayFocusThroughLens(acoustic_lens, pipeline, transducer, loss_model=loss_model)
    optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=30, eps=1e-6, verbose=True)
    raytracer = RaytracerScenario(physics, optimizer)
    rays = raytracer.solve(XX[is_within_pipe].ravel(), ZZ[is_within_pipe].ravel())
    tofs = physics.compute_tofs(rays)

    # np.save(project_root / "data" / "tof" / ..., tofs)

    #%%
    # Compute the full-matrix capture response for a inspection with a single punctual scatterer
    simulation = Simulator(gate_start=60e-6, gate_end=80e-6, fs=62.5e6, raytracer=raytracer)

    # find the full matrix capture response:
    fmc = hilbert(simulation.simulate_scatterer(
        xf=radius_scatterer * np.sin(theta_scatterer),
        zf=radius_scatterer * np.cos(theta_scatterer)
    ), axis=0)

    # Number of elements
    n_elem = transducer.num_elem

    tfm_pipe = np.zeros(tofs.shape[0], dtype=np.complex64)


    for tx in range(n_elem):
        for rx in range(n_elem):
            tof_total = np.round(tofs[:, tx, rx] * simulation.fs).astype(np.int64)

            tof_corrected = (tof_total - simulation.gate_start * simulation.fs).astype(np.int64)

            valid = tof_total - simulation.gate_start * simulation.fs < simulation.gate_samples

            tfm_pipe[valid] += fmc[tof_corrected[valid], tx, rx]


    tfm = np.zeros(XX.shape, dtype=np.complex64)
    tfm[is_within_pipe] = tfm_pipe


    img = np.zeros(XX.shape, dtype=np.float64)
    img[is_within_pipe] = np.abs(tfm)[is_within_pipe]
    img[~is_within_pipe] = np.nan

    #%%

    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(
        img,
        extent=(xspan[0], xspan[-1], zspan[-1], zspan[0]),
        cmap="inferno"
    )

    ax.set_ylim([40e-3, 80e-3])

    ax.plot(
        pipeline.xout,
        pipeline.zout,
        color="r",
        lw=3,
        ls="-"
    )

    ax.plot(
        pipeline.xint,
        pipeline.zint,
        color="r",
        lw=3,
        ls="-"
    )

    ax.set_aspect("equal")

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x * 1e3:.0f}")
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y * 1e3:.0f}")
    )

    ax.set_xlabel("x / (mm)")
    ax.set_ylabel("z / (mm)")
    ax.set_xlim((-10e-3, 50e-3))


    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

