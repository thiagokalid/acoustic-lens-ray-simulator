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
from pipe_lens.utils import convert_parameters
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def compute_field_point(x, z):
    fmc_w = simulation.simulate_scatterer(
        np.asarray([x]),
        np.asarray([z]),
        time_domain=False
    )
    shift = np.exp(-1j * np.multiply.outer(simulation.omega, delay_law))
    fmc_w_delayed = fmc_w * shift

    beam_w = np.sum(fmc_w_delayed, axis=1)

    return np.sqrt(np.sum(np.abs(beam_w)**2))

def worker(ij):
    i, j = ij
    return i, j, compute_field_point(XX[i, j], ZZ[i, j])

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
    h0, _, _ = convert_parameters(c=d/2, wc=4.1e-3, R=outer_radius, v1=wedge_material.cp, v2=coupling_medium.cp)
    alpha_max = np.radians(50)
    alpha_0 = 0

    # Delay-law related:
    focus_angle = np.radians(20)
    focus_radius = 60e-3

    # Configure simulation:
    loss_model = LossModel(transmission=False, transmission_diffraction=False, attenuation=False)

    #%%
    # Create pipeline:
    pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center=(0, 0))
    pipeline_shifted = Pipeline(pipe_material, outer_radius, wall_thickness, center=(0, 0))

    # Create acoustic lens:
    acoustic_lens = AcousticLens(wedge_material, coupling_medium, d, alpha_max, alpha_0, h0, outer_radius)

    # Create transducer:
    transducer = Transducer(pitch, num_elem, element_gap, central_freq, bw, bwr, pulse_type, center=(0., acoustic_lens.d))


    #%% Simulation:

    xfocus, zfocus = focus_radius * np.sin(focus_angle) + pipeline_shifted.xcenter, focus_radius * np.cos(focus_angle) + pipeline_shifted.zcenter


    # Convergent-wave delay law:
    physics = RayFocusThroughLens(acoustic_lens, pipeline, transducer)
    optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=30, eps=1e-6)
    raytracer = RaytracerScenario(physics, optimizer)

    opt_rays = raytracer.solve(xfocus, zfocus)
    roundtrip_tofs = physics.compute_tofs(opt_rays)
    tofs_roi = roundtrip_tofs.diagonal(axis1=1, axis2=2).T / 2
    delay_law = np.mean(tofs_roi, axis=0) - tofs_roi
    delay_law = delay_law[..., 0]

    # Plane-wave delay law:
    # i = np.arange(0, transducer.num_elem)
    # steering_angle = acoustic_lens.pipe2steering_angle(focus_angle)
    # delay_law = pitch/acoustic_lens.c1 * np.outer(np.sin(steering_angle), (i - 1)[:, np.newaxis])
    # delay_law[steering_angle < 0] = pitch/acoustic_lens.c1 * np.outer(abs(np.sin(steering_angle)), (transducer.num_elem - i))[steering_angle < 0]
    # delay_law = delay_law[0, :]


    # Compute tensile-based field:
    physics = RayOnewayThroughLens(acoustic_lens, pipeline_shifted, transducer, loss_model)
    optimizer = NewtonOptimizer(alpha_max=alpha_max,  alpha_min=acoustic_lens.alpha_min, max_iter=25, eps=1e-4)
    raytracer = RaytracerScenario(physics, optimizer)
    simulation = Simulator(gate_start=60e-6, gate_end=80e-6, fs=62.5e6, raytracer=raytracer)

    # Define API scatterers grid:
    ang_step = .5
    radius_step = .25e-3

    radius_span = np.arange(50e-3, 70e-3, radius_step)
    ang_span = np.radians(np.arange(np.degrees(focus_angle) - 20, np.degrees(focus_angle) + 20 + ang_step, ang_step))

    RR, TT = np.meshgrid(radius_span, ang_span)
    XX, ZZ = RR * np.sin(TT) + pipeline_shifted.xcenter, RR * np.cos(TT) + pipeline_shifted.zcenter

    field = np.empty_like(XX, dtype=float)

    print("Total available CPU cores: ", cpu_count())

    with Pool(cpu_count()) as pool:
        results = pool.imap_unordered(
            worker,
            product(range(XX.shape[0]), range(XX.shape[1])),
            chunksize=100
        )

        for i, j, value in tqdm(
                results,
                total=XX.size,
                desc="Simulating scatterers"
        ):
            field[i, j] = value


    #%%
    PLOT_FOR_PAPER = True
    if PLOT_FOR_PAPER:
        fig, axs = plt.subplots(figsize=(LATEX_LINEWIDTH * .5, 2.4))
    else:
        fig, axs = plt.subplots(figsize=(8, 8))
    im = axs.pcolormesh(XX * 1e3, ZZ * 1e3, field/field.max(), cmap='inferno', rasterized=True,)
    plt.colorbar(im)

    # find maximum
    imax, jmax = np.unravel_index(np.argmax(field), field.shape)
    # axs.plot(XX[imax, jmax] * 1e3, ZZ[imax, jmax] * 1e3, "o", color='lime', markeredgewidth=1, markersize=12, markerfacecolor='none', label="Local max")
    axs.plot(xfocus * 1e3, zfocus * 1e3, "o", color='lime',markersize=3, label="Focus")


    # plot pipe and lens overlay:

    axs.plot(pipeline_shifted.xout * 1e3, pipeline_shifted.zout * 1e3, '--', color='black', label="Real tube position")
    axs.plot(pipeline_shifted.xint * 1e3, pipeline_shifted.zint * 1e3, '--', color='black', label="_")
    axs.plot(acoustic_lens.xlens * 1e3, acoustic_lens.zlens * 1e3, '--', color='black', label="_")
    axs.plot(transducer.xt * 1e3, transducer.zt * 1e3, 's', color='black', label="Transducer")

    axs.set_aspect('equal')
    axs.set_xlabel("x-axis / (mm)")
    axs.set_ylabel("z-axis / (mm)")


    if PLOT_FOR_PAPER:
        axs.set_ylim([25, 100])
        axs.set_xlim([-50, 50])

    else:
        axs.set_ylim([25, 220])
        axs.plot(pipeline.xout * 1e3, pipeline.zout * 1e3, '-', color='r', alpha=.5, lw=3, label="Assumed tube position for delay-law focusing")
        axs.plot(pipeline.xint * 1e3, pipeline.zint * 1e3, '-', color='r', alpha=.5, lw=3, label="_")
        axs.set_title("RMS pressure field")
        plt.legend()

    plt.tight_layout()
    plt.savefig(project_root / "figures" / "rms_pfield_convergent_wave.pdf")
    plt.show()


    #%%
    if PLOT_FOR_PAPER:
        fig, axs = plt.subplots(figsize=(LATEX_LINEWIDTH * .5, 2.4))
    else:
        fig, axs = plt.subplots(figsize=(8, 8))

    bars = axs.bar(
        x=np.arange(transducer.num_elem),
        height=(delay_law - delay_law.min()) * 1e6,
        color='b'
    )
    axs.set_xticks(np.arange(0, 64 + 8, 8))
    axs.set_xlabel("Transducer element")
    axs.set_ylabel(r"Transmission delay / ($\mathrm{\mu s}$)")

    plt.tight_layout()
    plt.savefig(project_root / "figures" / "rms_field_delaylaw.pdf")
    plt.show(block=True)
