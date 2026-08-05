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
from pipe_lens.utils import fwhm, convert_time2radius


#%% USER INPUT:

LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth
PLOT_SSCANS = False
COMPUTE_SIM_RESULTS = True
COMPUTE_EXP_RESULTS = True
PLOT_FOR_PAPER = True

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
delay_law_type = "focused_wave"  # "focused_wave" or "plane_wave"
focus_alpha_step = np.radians(.5)
focus_alpha_max= np.radians(45)
focus_alpha_min = np.radians(-45)
focus_radius = 50e-3 # not necessary if plane_wave is selected.

# Define reflector positions for the PSF evaluation:
r_scatterer = 60e-3
theta_scatterer = np.radians(np.arange(0, 40 + 1, 1))

# Configure simulation:
loss_model = LossModel(transmission=True, transmission_diffraction=True, reception_diffraction=True, attenuation=True)

# Create pipeline:
pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center)
pipeline_shifted = Pipeline(pipe_material, outer_radius, wall_thickness, (0, 0))

# Create acoustic lens:
acoustic_lens = AcousticLens(wedge_material, coupling_medium, d, alpha_max, alpha_0, h0, outer_radius)

# Create transducer:
transducer = Transducer(pitch, num_elem, element_gap, central_freq, bw, bwr, pulse_type, center=(0., acoustic_lens.d))

# Define location for the focuses:
focus_angle = np.arange(focus_alpha_min, focus_alpha_max + focus_alpha_step, focus_alpha_step)

#%% Compute the **Simulated** results:

scaling_factor = 1. if PLOT_FOR_PAPER else 3

# Objects for the simulation:
physics = RayFocusThroughLens(acoustic_lens, pipeline_shifted, transducer, loss_model)
optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=15, eps=1e-4)
raytracer = RaytracerScenario(physics, optimizer)
simulation = Simulator(gate_start=50e-6, gate_end=70e-6, fs=62.5e6, raytracer=raytracer)

if COMPUTE_SIM_RESULTS:
    if delay_law_type == "focused_wave":
        # Focii positions:
        xfocus, zfocus = focus_radius * np.sin(focus_angle), focus_radius * np.cos(focus_angle)

        # Convergent-wave delay law:
        physics = RayFocusThroughLens(acoustic_lens, pipeline, transducer)
        optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=30, eps=1e-6)
        raytracer = RaytracerScenario(physics, optimizer)

        opt_rays = raytracer.solve(xfocus, zfocus)
        roundtrip_tofs = physics.compute_tofs(opt_rays)
        tofs_roi = roundtrip_tofs.diagonal(axis1=1, axis2=2).T / 2
        delay_law = np.mean(tofs_roi, axis=0) - tofs_roi

    elif delay_law_type == "plane_wave":
        # Plane-wave delay law:
        i = np.arange(0, transducer.num_elem)
        steering_angle = acoustic_lens.pipe2steering_angle(focus_angle)
        delay_law = pitch/acoustic_lens.c1 * np.outer(np.sin(steering_angle), (i - 1)[:, np.newaxis])
        delay_law[steering_angle < 0] = pitch/acoustic_lens.c1 * np.outer(abs(np.sin(steering_angle)), (transducer.num_elem - i))[steering_angle < 0]

    else:
        raise ValueError("Invalid delay_law_type")

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

    x_scatterer, z_scatterer = r_scatterer * np.sin(theta_scatterer), r_scatterer * np.cos(theta_scatterer)

    sim_amplitude = np.zeros(shape=(len(theta_scatterer)))
    sim_fwhm_w = np.zeros(shape=(len(theta_scatterer)))
    sim_fwhm_h = np.zeros(shape=(len(theta_scatterer)))


    for i in tqdm(range(len(theta_scatterer)), desc="Running simulations"):
        fmc = simulation.simulate_scatterer(np.asarray([x_scatterer[i]]), np.asarray([z_scatterer[i]]))
        delay_law_in_shifts = np.round(delay_law * simulation.fs).astype(np.int32)

        optimal_ray = raytracer.solve(np.asarray([x_scatterer[i]]), np.asarray([z_scatterer[i]]))
        distances = np.asarray([ray.compute_dist() for ray in optimal_ray])


        # assuming linearity, we create an s-scan from the fmc based on the emission and reception delay-law
        sscan = fmc2sscan_numba(
            fmc=fmc,
            shifts_e=delay_law_in_shifts,
            shifts_r=delay_law_in_shifts
        )[..., np.newaxis]

        sscan_env = envelope(sscan[..., 0], axis=0)

        if i == 0:
            ref_amplitude = sscan_env.max()

        # print(f"S-scan relative max = {sscan_env.max() / ref_amplitude}")


        window_corners = [
            (65e-3, theta_scatterer[i] - np.radians(8)),  # north-east
            (55e-3, theta_scatterer[i] + np.radians(8))  # south-west
        ]
        sim_fwhm_w[i], sim_fwhm_h[i], sim_amplitude[i], mask = fwhm(sscan_env, radius_span, focus_angle, window_corners, drawSquare=False)

        if PLOT_SSCANS:
            sscan_db = np.log10(sscan_env / ref_amplitude + 1E-6)

            fig, axs = plt.subplots(1, 2, figsize=(LATEX_LINEWIDTH * .8 * scaling_factor, 2 * scaling_factor))
            axs[0].imshow(sscan_db, vmin=-6, vmax=0, extent=[np.degrees(focus_alpha_min), np.degrees(focus_alpha_max), simulation.gate_end * 1e6, simulation.gate_start * 1e6],cmap="inferno", aspect='auto')
            axs[0].axhline(y=acoustic_lens.t_outer * 1e6, color='lime', ls='--', lw=1)
            axs[0].axhline(y=(acoustic_lens.t_outer + 2 * pipeline.wall_width/pipeline.material.cp) * 1e6, color='lime', ls='--', lw=1)

            delta_thickness = outer_radius - r_scatterer

            # plt.plot(np.degrees(theta_scatterer[i]), (acoustic_lens.t_outer + 2 * delta_thickness/pipeline.material.cp) * 1e6, 'o', markerfacecolor="None", color='lime', ms=15)
            axs[0].set_xlabel("Angular position / (degrees)")
            axs[0].set_ylabel(r"Time / $\mathrm{(\mu s)}$")
            axs[0].set_ylim([57.5 + 4, 57.5 - 4])
            axs[0].set_xticks(np.arange(-45, 45 + 15, 15))


            axs[1].imshow(mask, extent=[np.degrees(focus_alpha_min), np.degrees(focus_alpha_max), simulation.gate_end * 1e6, simulation.gate_start * 1e6],cmap="binary", aspect='auto')
            axs[1].set_xlabel("Angular position / (degrees)")
            axs[1].set_ylabel(r"Time / $\mathrm{(\mu s)}$")
            axs[1].set_ylim([57.5 + 4, 57.5 - 4])
            axs[1].set_xticks(np.arange(-45, 45 + 15, 15))


            # plt.savefig(project_root / "figures" / f"sample_sscan_{np.degrees(theta_scatterer[i]):.0f}deg_sim.pdf")
            plt.tight_layout()
            plt.show(block=True)

#%% Compute the **Experimental** results:
if COMPUTE_EXP_RESULTS:
    from framework import file_m2k

    n_reps = 2
    n_pos = len(theta_scatterer)

    exp_amplitude = np.zeros(shape=(n_reps, n_pos))
    exp_fwhm_w = np.zeros(shape=(n_reps, n_pos))
    exp_fwhm_h = np.zeros(shape=(n_reps, n_pos))

    for i in tqdm(range(n_pos), desc="Processing experimental acquisitions"):
        data_path = project_root / "data"
        for r in range(2):
            shot = int(np.degrees(theta_scatterer[i]))
            data_insp = file_m2k.read(str(data_path / f"lens_{delay_law_type}_1degree_v{r + 1}.m2k"), freq_transd=central_freq,
                                      bw_transd=bw, tp_transd=pulse_type, sel_shots=shot)
            sscan = np.sum(data_insp.ascan_data[..., 0], axis=2)
            sscan_env = envelope(sscan, axis=0)
            time_grid = data_insp.time_grid[..., 0] * 1e-6
            time_shift = 1.1e-6

            radius_span = convert_time2radius(
                tspan=time_grid,
                t_outer=55.5e-6,
                t_inner=61.4e-6,
                cl_surf=pipeline.material.cp,
                cl_upper=acoustic_lens.coupling_medium.cp,
                cl_bottom=acoustic_lens.coupling_medium.cp,
                outer_r=pipeline.outer_radius
            )

            if i == 0:
                ref_amplitude = sscan_env.max()

            window_corners = [
                (65e-3, theta_scatterer[i] - np.radians(8)),  # north-east
                (55e-3, theta_scatterer[i] + np.radians(8))  # south-west
            ]

            exp_fwhm_w[r, i], exp_fwhm_h[r, i], exp_amplitude[r, i], mask = fwhm(sscan_env, radius_span, focus_angle, window_corners)


            if PLOT_SSCANS:

                fig, ax = plt.subplots(figsize=(LATEX_LINEWIDTH * .4 * scaling_factor, 2 * scaling_factor))
                sscan_db = np.log10(sscan_env / sscan_env.max() + 1E-3)
                plt.imshow(sscan_db, vmin=-3, vmax=0,
                           extent=[-45, 45, time_grid[-1] * 1e6, time_grid[0] * 1e6], cmap="inferno", aspect='auto')

                delta_thickness = outer_radius - r_scatterer

                plt.xlabel("Angular position / (degrees)")
                plt.ylabel(r"Time / $\mathrm{(\mu s)}$")
                plt.ylim([58.97 + 4, 58.97 - 4])
                plt.xticks(np.arange(-45, 45 + 15, 15))
                plt.tight_layout()
                plt.savefig(project_root / "figures" / f"sample_sscan_{np.degrees(theta_scatterer[i]):.0f}deg_exp.pdf")
                plt.show(block=True)
    std_exp_amplitude = np.std(exp_amplitude, axis=0)
    std_exp_fwhm_w = np.std(exp_fwhm_w, axis=0)
    std_exp_fwhm_h = np.std(exp_fwhm_h, axis=0)

    mean_exp_amplitude = np.mean(exp_amplitude, axis=0)
    mean_exp_fwhm_w = np.mean(exp_fwhm_w, axis=0)
    mean_exp_fwhm_h = np.mean(exp_fwhm_h, axis=0)


#%%
if COMPUTE_EXP_RESULTS and COMPUTE_SIM_RESULTS:
    fig, ax = plt.subplots(figsize=(LATEX_LINEWIDTH * .4 * scaling_factor, 2 * scaling_factor))
    if not PLOT_FOR_PAPER:
        plt.suptitle("Resolution measured by considering punctual reflector.")
        ax.set_title("Full Width at Half Maximum (FWHM)")
    if COMPUTE_SIM_RESULTS:
        # Horizontal:
        ax.plot(np.degrees(theta_scatterer), np.degrees(sim_fwhm_w), '-k', label="Horizontal Simulated FWHM", lw=2)

        # Vertical:
        # ax.plot(np.degrees(theta_scatterer), sim_fwhm_h * 1e3, '-k', label="Horizontal Simulated FWHM", lw=3)
    if COMPUTE_EXP_RESULTS:
        # Horizontal:
        ax.plot(np.degrees(theta_scatterer), np.degrees(mean_exp_fwhm_w), 'x-', color='#FF1F5B', label="Horizontal Experimental FWHM", lw=1, ms=4)
        ax.fill_between(
        x=np.degrees(theta_scatterer),
        y1=np.degrees(mean_exp_fwhm_w - std_exp_fwhm_w),
        y2=np.degrees(mean_exp_fwhm_w + std_exp_fwhm_w),
        color='r', alpha=.3)

    ax.set_xticks(np.arange(0, 40 + 8, 8))
    ax.set_xlabel("Reflector angular position / (degrees)")
    ax.set_ylabel(r"Width / (degrees)")


    if not PLOT_FOR_PAPER:
        ax.legend()

    ax.grid()
    plt.tight_layout()
    plt.savefig(project_root / "figures" / "fwhm_horizontal_vs_angle.pdf")
    plt.show()

    # FWHM Vertical:
    fig, ax = plt.subplots(figsize=(LATEX_LINEWIDTH * .4 * scaling_factor, 2 * scaling_factor))
    if not PLOT_FOR_PAPER:
        plt.suptitle("Resolution measured by considering punctual reflector.")
        ax.set_title("Full Width at Half Maximum (FWHM)")
    if COMPUTE_SIM_RESULTS:
        # Vertical:
        ax.plot(np.degrees(theta_scatterer), sim_fwhm_h * 1e3, '-k', label="Simulated", lw=2, ms=4)
    if COMPUTE_EXP_RESULTS:
        # Vertical:
        ax.plot(np.degrees(theta_scatterer), mean_exp_fwhm_h * 1e3, '-x', color='#FF1F5B', label="Experimental", lw=1, ms=4)
        ax.fill_between(
        x=np.degrees(theta_scatterer),
        y1=(mean_exp_fwhm_h - std_exp_fwhm_h) * 1e3,
        y2=(mean_exp_fwhm_h + std_exp_fwhm_h) * 1e3,
        color='r', alpha=.3)

    ax.set_xticks(np.arange(0, 40 + 8, 8))
    ax.set_xlabel("Reflector angular position / (degrees)")
    ax.set_ylabel(r"Height / (mm)")

    # if not PLOT_FOR_PAPER:
    ax.legend()

    ax.grid()
    plt.tight_layout()
    plt.savefig(project_root / "figures" / "fwhm_vertical_vs_angle.pdf")
    plt.show()

    fig, ax = plt.subplots(figsize=(LATEX_LINEWIDTH * .4 * scaling_factor, 2 * scaling_factor))

    if COMPUTE_SIM_RESULTS:
        ax.plot(np.degrees(theta_scatterer), np.asarray(sim_amplitude) / np.max(sim_amplitude), '-k', label="Simulated Amplitude", lw=2)
    if COMPUTE_EXP_RESULTS:
        ax.plot(np.degrees(theta_scatterer), np.asarray(mean_exp_amplitude) / np.max(mean_exp_amplitude), '-x', color='#FF1F5B', label="Horizontal Experimental FWHM", lw=1, ms=4)
        ax.fill_between(
        x=np.degrees(theta_scatterer),
        y1=(mean_exp_amplitude - std_exp_amplitude) / np.max(mean_exp_amplitude),
        y2=(mean_exp_amplitude + std_exp_amplitude) / np.max(mean_exp_amplitude),
        color='r', alpha=.3)

    if not PLOT_FOR_PAPER:
        ax.set_title("Amplitudes")
    ax.set_xticks(np.arange(0, 40 + 8, 8))
    ax.set_xlabel("Reflector angular position / (degrees)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid()
    plt.tight_layout()
    plt.savefig(project_root / "figures" / "amplitude_vs_angle.pdf")
    plt.show()