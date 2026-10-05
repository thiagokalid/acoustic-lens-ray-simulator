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

import matplotlib
matplotlib.use("TkAgg")

cm = 1 / 2.54

LATEX_LINEWIDTH = 6.8438 # LaTeX linewidth


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
from pipe_lens.delay_laws import planewave_delaylaw, focusedwave_delaylaw
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

REF_AMPLITUDE = 11e3

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
    wall_thickness = 60e-3
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
    focus_angle = np.radians(30)
    focus_radius = 60e-3
    delay_law_type = "null" # "plane_wave" or "focused_wave" or "null"

    # Configure simulation:
    loss_model = LossModel(transmission=False,transmission_diffraction=False, attenuation=False)

    #%%
    # Create pipeline:
    pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center=(0e-3, 0))
    pipeline_shifted = Pipeline(pipe_material, outer_radius, wall_thickness, center=(0e-3, 0))
    pipeline_original = Pipeline(pipe_material, outer_radius, 20e-3, center=(0, 0))

    # Create acoustic lens:
    acoustic_lens = AcousticLens(wedge_material, coupling_medium, d, alpha_max, alpha_0, h0, outer_radius)

    # Create transducer:
    transducer = Transducer(pitch, num_elem, element_gap, central_freq, bw, bwr, pulse_type, center=(0., acoustic_lens.d))


    #%% Simulation:

    xfocus, zfocus = focus_radius * np.sin(focus_angle) + pipeline_shifted.xcenter, focus_radius * np.cos(focus_angle) + pipeline_shifted.zcenter

    match delay_law_type:
        case "plane_wave":
            # Plane-wave delay law:
            steering_angle = acoustic_lens.pipe2steering_angle(focus_angle)
            delay_law = planewave_delaylaw(acoustic_lens.c1, transducer.xt, steering_angle)[:, 0]
        case "focused_wave":
            # Convergent-wave delay law:
            physics = RayFocusThroughLens(acoustic_lens, pipeline, transducer)
            optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=30, eps=1e-6)
            raytracer = RaytracerScenario(physics, optimizer)

            opt_rays = raytracer.solve(xfocus, zfocus)
            roundtrip_tofs = physics.compute_tofs(opt_rays)
            tofs_roi = roundtrip_tofs.diagonal(axis1=1, axis2=2).T / 2
            delay_law = focusedwave_delaylaw(tofs_roi)[:, 0]
        case "null":
            delay_law = np.zeros(transducer.num_elem)
            focus_angle = 0
        case _:
            raise ValueError("Unknown delay_law type")


    # Compute tensile-based field:
    physics = RayOnewayThroughLens(acoustic_lens, pipeline_shifted, transducer, loss_model)
    optimizer = NewtonOptimizer(alpha_max=alpha_max,  alpha_min=acoustic_lens.alpha_min, max_iter=25, eps=1e-4)
    raytracer = RaytracerScenario(physics, optimizer)
    simulation = Simulator(gate_start=60e-6, gate_end=80e-6, fs=62.5e6, raytracer=raytracer)

    # Define API scatterers grid:
    ang_step = .5
    radius_step = .25e-3

    radius_span = np.arange(10e-3, 70e-3, radius_step)
    ang_span = np.radians(np.arange(-45, 45 + ang_step, ang_step))

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
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Convert amplitude to dB
    img = 20 * np.log10(
        np.maximum(field / REF_AMPLITUDE, 1e-30)
    )

    # dB levels: -30, -25, ..., 0
    levels = np.arange(-30, 1, 5)

    # White -> black
    colors = np.linspace(1.0, 0.0, len(levels))

    cmap = ListedColormap(
        [(c, c, c) for c in colors],
        name="white_black_5dB"
    )

    # 5 dB-wide intervals centered on each level
    bounds = np.arange(-32.5, 2.5, 5)
    norm = BoundaryNorm(bounds, cmap.N)

    #
    if delay_law_type == "focused_wave":
        FIGSIZE = ((5 + 1) * cm, 6 * cm)
    else:
        FIGSIZE = (5 * cm, 6 * cm)
    #
    fig, axs = plt.subplots(figsize=FIGSIZE)

    im = axs.pcolormesh(
        XX * 1e3,
        ZZ * 1e3,
        img,
        cmap="Grays",
        vmax=0,
        vmin=-20,
        rasterized=True,
    )

    axs.plot(pipeline_shifted.xout * 1e3, pipeline_shifted.zout * 1e3, '--', color='r', label="_")

    xpipe = np.arange(-pipeline.outer_radius, pipeline.outer_radius, .1e-3)
    axs.fill_between(
        xpipe * 1e3,
        np.sqrt(pipeline.outer_radius**2 - xpipe**2) * 1e3,
        np.sqrt(pipeline_original.inner_radius**2 - xpipe**2) * 1e3,
        color='r',
        ls='-',
        alpha=0.15,
        label="_"
    )


    axs.plot(acoustic_lens.xlens * 1e3, acoustic_lens.zlens * 1e3, '-', color='black', label="_")
    axs.plot(transducer.xt * 1e3, transducer.zt * 1e3, 's', color='black', label="_")



    axs.set_aspect('equal')
    axs.set_xlabel("x-axis / (mm)")
    axs.set_ylabel("z-axis / (mm)")

    if delay_law_type == "null":
        axs.set_xlim([-35, 35])
        axs.set_xticks(np.arange(-30, 30 + 15, 15))

    else:
        axs.set_xlim([-20, 50])
        axs.set_xticks(np.arange(-20, 40 + 20, 20))

    if delay_law_type == "focused_wave":
        axs.plot(xfocus * 1e3, zfocus * 1e3, "o", color='lime', markersize=4, label="Focus")

        cbar = fig.colorbar(
            im,
            ax=axs,
        )

        cbar.set_label("Amplitude / (dB)")
        cbar.set_ticks([0, -5, -10, -15, -20])
    elif delay_law_type == "plane_wave":
        axs.axline((0, 0), slope=np.tan(np.pi / 2 - focus_angle), color='b', lw=1, ls=':', alpha=.5)



    # find maximum
    imax, jmax = np.unravel_index(np.argmax(field), field.shape)
    axs.plot(XX[imax, jmax] * 1e3, ZZ[imax, jmax] * 1e3, "*", color='r', markersize=3, label="Maximum")

    if delay_law_type == "focused_wave":
        plt.legend(loc='upper right')

    axs.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(project_root / "figures" / f"rms_pfield_{delay_law_type}.pdf")
    plt.show()
