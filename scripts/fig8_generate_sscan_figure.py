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

from pathlib import Path

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
from pipe_lens.delay_laws import planewave_delaylaw, focusedwave_delaylaw
from framework.post_proc import envelope

# Constants:
LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth
FIGSIZE = (4, 3)
CROPPED_SSCAN_FIGSIZE = (2, 2)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_PATH = PROJECT_ROOT / "figures" / "fig8"
if not FIGS_PATH.exists():
    FIGS_PATH.mkdir(parents=True)
DATA_PATH = PROJECT_ROOT / "data" / "experiment_1"
COMPUTE_SIM_RESULTS = True
COMPUTE_EXP_RESULTS = True
SAVE_FIGS = True

def plot_sscan(sscan_db, tbeg, tend):
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(6. * cm, 5 * cm))


    extent = (np.degrees(focus_alpha_min), np.degrees(focus_alpha_max), tend * 1e6, tbeg * 1e6)
    im = ax.imshow(sscan_db, vmin=0, vmax=1, extent=extent, cmap="YlGnBu", aspect='auto')

    ax.set_xlabel("Angular position / (degrees)")
    ax.set_ylabel(r"Time / $\mathrm{(\mu s)}$")
    ax.set_xticks(np.arange(-45, 45 + 15, 15))

    # Subfig pequena
    ax.set_ylim([60.0, 52.19])

    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig(FIGS_PATH / f"sscan_{delay_law_type}_full.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show(block=True)


def plot_cropped(sscan, tbeg: float, tend: float, tbeg_crop: float, tend_crop: float, abeg, aend, angle_rad=0, inspection_type="experimental"):
    fig, ax = plt.subplots(1, 1, figsize=CROPPED_SSCAN_FIGSIZE)

    extent = (np.degrees(focus_alpha_min), np.degrees(focus_alpha_max), tend, tbeg)
    ax.imshow(sscan, vmin=0, vmax=1, extent=extent, cmap="YlGnBu", aspect='auto')

    ax.set_ylim([tend_crop, tbeg_crop])
    ax.set_xlim([abeg, aend])
    plt.axis('off')
    plt.tight_layout()

    if SAVE_FIGS:
        fig_name = f"windowed_sscan_angle_{np.degrees(angle_rad):.1f}.png"

        if delay_law_type == "focused_wave" and inspection_type == "simulated":
            subfolder = '1st_column'
        elif delay_law_type == "plane_wave" and inspection_type == "simulated":
            subfolder = '2nd_column'
        elif delay_law_type == "focused_wave" and inspection_type == "experimental":
            subfolder = '3rd_column'
        else:
            subfolder = '4th_column'

        fig_path = FIGS_PATH / subfolder
        if not fig_path.exists():
            fig_path.mkdir(parents=True)

        plt.savefig(fig_path / fig_name, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show(block=True)


def simulate_single_scatterer(xreflector, zreflector, simulation, delay_law):
    fmc = simulation.simulate_scatterer(
        xf=xreflector,
        zf=zreflector
    )
    delay_law_in_shifts = np.round(delay_law * simulation.fs).astype(np.int32)

    # assuming linearity, we create an s-scan from the fmc based on the emission and reception delay-law
    sscan = fmc2sscan_numba(
        fmc=fmc,
        shifts_e=delay_law_in_shifts,
        shifts_r=delay_law_in_shifts
    )[..., np.newaxis]

    return sscan

def generate_focal_law(delay_law_type, focus_radius, focus_angle, acoustic_lens, pipeline, transducer):
    if delay_law_type == "focused_wave":
        # Focii positions:
        xfocus, zfocus = focus_radius * np.sin(focus_angle), focus_radius * np.cos(focus_angle)

        # Convergent-wave delay law:
        physics = RayFocusThroughLens(acoustic_lens, pipeline, transducer)
        optimizer = NewtonOptimizer(alpha_max=acoustic_lens.alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=30, eps=1e-6)
        raytracer = RaytracerScenario(physics, optimizer)

        opt_rays = raytracer.solve(xfocus, zfocus)
        roundtrip_tofs = physics.compute_tofs(opt_rays)
        tofs_roi = roundtrip_tofs.diagonal(axis1=1, axis2=2).T / 2

        delay_law = focusedwave_delaylaw(tofs_roi)

    elif delay_law_type == "plane_wave":
        # Plane-wave delay law:
        steering_angle = acoustic_lens.pipe2steering_angle(focus_angle)
        delay_law = planewave_delaylaw(acoustic_lens.c1, transducer.xt, steering_angle)

    else:
        raise ValueError("Invalid delay_law_type")

    return delay_law


if __name__ == "__main__":
    #%% USER INPUT:

    # Materials related:
    steel = Solid(cp=5900, cs=3200, density=7850)#, attenuation_coeff=[(23.4 * 1.1513e-7, 0)])#, (5.74e-3 * 1.1513e-25, 4)])
    aluminium = Solid(cp=6332.94, cs=3128.18, density=2730)#, attenuation_coeff=[(14.4 * 1.1513e-7, 0)]) # attenuation not measured.
    water = Fluid(cp=1430, density=997)#, attenuation_coeff=[(25e-15, 2.)])

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
    wall_thickness = 18.5e-3
    center = (0., 0.)

    # Acoustic-lens related:
    wedge_material = aluminium
    coupling_medium = water
    d = 2 * 85e-3
    # h0 = 90e-3
    h0, _, _ = convert_parameters(c=d/2, wc=8.6e-3, R=outer_radius, v1=wedge_material.cp, v2=coupling_medium.cp)
    alpha_max = np.radians(50)
    alpha_0 = 0


    # Delay-law related:
    focus_alpha_step = np.radians(.5)
    focus_alpha_max= np.radians(45)
    focus_alpha_min = np.radians(-45)
    focus_radius = 50e-3 # not necessary if plane_wave is selected.

    # Define reflector positions for the PSF evaluation:
    reflector_radius = 60e-3
    reflector_angular_position = np.radians(np.array([0, 10, 20, 30, 40]))
    num_reflectors = len(reflector_angular_position)

    # Configure simulation:
    loss_model = LossModel(transmission=True, transmission_diffraction=True, reception_diffraction=True, attenuation=True)

    # Create pipeline:
    pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center=(0, 0))
    pipeline_shifted = Pipeline(pipe_material, outer_radius, wall_thickness, (0, 0))

    # Create acoustic lens:
    acoustic_lens = AcousticLens(wedge_material, coupling_medium, d, alpha_max, alpha_0, h0, outer_radius)

    # Create transducer:
    transducer = Transducer(pitch, num_elem, element_gap, central_freq, bw, bwr, pulse_type, center=(0., acoustic_lens.d))

    # Define location for the focuses:
    focus_angle = np.arange(focus_alpha_min, focus_alpha_max + focus_alpha_step, focus_alpha_step)

    #%% Compute the **Simulated** results:

    # Objects for the simulation:
    physics = RayFocusThroughLens(acoustic_lens, pipeline_shifted, transducer, loss_model)
    optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=15, eps=1e-4)
    raytracer = RaytracerScenario(physics, optimizer)
    simulation = Simulator(gate_start=40e-6, gate_end=80e-6, fs=62.5e6, raytracer=raytracer)

    if COMPUTE_SIM_RESULTS:
        for delay_law_type in ["focused_wave", "plane_wave"]:
            delay_law = generate_focal_law(delay_law_type, focus_radius, focus_angle, acoustic_lens, pipeline, transducer)

            x_reflector, z_reflector = reflector_radius * np.sin(reflector_angular_position), reflector_radius * np.cos(reflector_angular_position)

            for i in tqdm(range(num_reflectors), desc="Running simulations"):
                sscan = simulate_single_scatterer(
                    x_reflector[i], z_reflector[i], simulation, delay_law)
                sscan_env = envelope(sscan, axis=0)[..., 0]

                if i == 0:
                    ref_amplitude = sscan_env.max()
                    print(ref_amplitude)

                sscan_db = np.log10(sscan_env / ref_amplitude + 1E-6)

                theta_scatterer = np.arctan2(x_reflector[i], z_reflector[i])

                plot_cropped(sscan_env / ref_amplitude,
                             tbeg=simulation.gate_start, tend=simulation.gate_end,
                             abeg=np.degrees(theta_scatterer)-14, aend=np.degrees(theta_scatterer)+14,
                             tbeg_crop=57e-6, tend_crop=58.8e-6,
                             inspection_type="simulated", angle_rad=theta_scatterer
                             )


    #%% Compute the **Experimental** results:
    if COMPUTE_EXP_RESULTS:
        from framework import file_m2k

        for delay_law_type in ["focused_wave", "plane_wave"]:
            inspection_index = 1 if delay_law_type == "focused_wave" else 2

            # Read the reference signal (without pipe):
            data_ref = file_m2k.read(str(DATA_PATH / f"ref.m2k"), freq_transd=central_freq, bw_transd=bw,
                                     tp_transd=pulse_type)[inspection_index]
            channels_ref = np.mean(data_ref.ascan_data, axis=-1)

            for i in tqdm(range(num_reflectors), desc="Processing experimental acquisitions"):
                theta_scatterer = reflector_angular_position[i]
                shot = int(np.degrees(theta_scatterer))

                data_insp = file_m2k.read(str(DATA_PATH / f"psf_v1.m2k"),
                                          freq_transd=central_freq, bw_transd=bw, tp_transd=pulse_type, sel_shots=shot)[inspection_index]
                channels = data_insp.ascan_data[..., 0]

                sscan = np.sum(channels - channels_ref, axis=2)


                sscan_env = envelope(sscan, axis=0)
                time_grid = data_insp.time_grid[..., 0] * 1e-6
                time_shift = -1.5e-6



                if delay_law_type == "plane_wave":
                    tbeg_crop = 55.54e-6
                    tend_crop = tbeg_crop + 1.8e-6
                else:
                    tbeg_crop = 55.54e-6
                    tend_crop = tbeg_crop + 1.8e-6

                if i == 0:
                    delta_ang = np.radians(14)
                    row1, row2 = np.searchsorted(time_grid, tbeg_crop), np.searchsorted(time_grid, tend_crop)
                    col1, col2 = np.searchsorted(focus_angle, theta_scatterer - delta_ang), np.searchsorted(focus_angle, theta_scatterer + delta_ang)

                    ref_amplitude = np.max(sscan_env[row1:row2, col1:col2])

                sscan_db = np.log10(sscan_env / ref_amplitude + 1E-6)

                if i == 0:
                    plot_sscan(sscan_env / ref_amplitude, tbeg=time_grid[0], tend=time_grid[-1])


                plot_cropped(sscan_env / ref_amplitude,
                             tbeg=time_grid[0], tend=time_grid[-1],
                             abeg=np.degrees(theta_scatterer) - 14, aend=np.degrees(theta_scatterer) + 14,
                             tbeg_crop=tbeg_crop, tend_crop=tend_crop,
                             inspection_type="experimental", angle_rad=theta_scatterer
                             )

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(0.125, 4.7 * cm))

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cbar = matplotlib.colorbar.ColorbarBase(
        ax,
        cmap="YlGnBu",
        norm=norm,
    )

    cbar.set_label("Amplitude")

    fig.savefig(FIGS_PATH / "colorbar.pdf", bbox_inches="tight")
    plt.close(fig)