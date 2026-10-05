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

LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth
FIGSIZE = (LATEX_LINEWIDTH * .4 * 1.25, 2 * 1.25)

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
from pipe_lens.delay_laws import planewave_delaylaw, focusedwave_delaylaw
from framework.post_proc import envelope
from pipe_lens.utils import fwhm, convert_time2radius

# Constants:
LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth
cm = 1 / 2.54
FIGSIZE = (7 * cm, 6 * cm)
CROPPED_SSCAN_FIGSIZE = (2, 2)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_PATH = PROJECT_ROOT / "figures"
COMPUTE_SIM_RESULTS = False
COMPUTE_EXP_RESULTS = True
SAVE_FIGS = True

def plot_sscan(sscan_db, tbeg, tend, touter = 53.5e-6, tinner = 61.5e-6, insp_type="exp"):
    cm = 1 / 2.54
    if insp_type == "exp":
        fig, ax = plt.subplots(figsize=((7 + 1.) * cm, 5 * cm))
    else:
        fig, ax = plt.subplots(figsize=(7 * cm, 5 * cm))


    extent = (np.degrees(focus_alpha_min), np.degrees(focus_alpha_max), tend * 1e6,tbeg * 1e6)
    im = ax.imshow(20 * sscan_db, vmin=-60, vmax=0, extent=extent, cmap="YlGnBu", aspect='auto')

    if insp_type == "exp":
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Amplitude / (dB)")

    ax.set_xlabel("Angular position / (degrees)")
    ax.set_ylabel(r"Time / $\mathrm{(\mu s)}$")
    ax.set_ylim([tinner * 1e6, touter * 1e6])
    ax.set_xticks(np.arange(-45, 45 + 15, 15))

    plt.tight_layout()
    if SAVE_FIGS:
        filepath = project_root / "figures" / f"{insp_type}_sscan_{np.degrees(theta_scatterer[i]):.0f}deg_shifted.pdf"
        print(filepath)
        plt.savefig(filepath, bbox_inches='tight')
        plt.show(block=True)
    else:
        plt.show(block=True)

def simulate_single_scatterer(xreflector, zreflector, simulation, raytracer, delay_law, radius_span, focus_angle):
    theta_scatterer = np.arctan2(xreflector, zreflector)
    fmc = simulation.simulate_scatterer(
        xf=xreflector,
        zf=zreflector
    )
    delay_law_in_shifts = np.round(delay_law * simulation.fs).astype(np.int32)

    optimal_ray = raytracer.solve(
        xf=xreflector,
        zf=zreflector
    )
    distances = np.asarray([ray.compute_dist() for ray in optimal_ray])

    # assuming linearity, we create an s-scan from the fmc based on the emission and reception delay-law
    sscan = fmc2sscan_numba(
        fmc=fmc,
        shifts_e=delay_law_in_shifts,
        shifts_r=delay_law_in_shifts
    )[..., np.newaxis]

    sscan_env = envelope(sscan, axis=0)[..., 0]

    # print(f"S-scan relative max = {sscan_env.max() / ref_amplitude}")

    window_corners = [
        (70e-3, theta_scatterer - np.radians(14)),  # north-east
        (45e-3, theta_scatterer + np.radians(14))  # south-west
    ]
    fwhm_w, fwhm_h, amplitude, mask, api = fwhm(sscan_env, radius_span, focus_angle, window_corners,
                                                                drawSquare=False)

    return fwhm_w, fwhm_h, amplitude, sscan, mask, api

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
    h0, _, _ = convert_parameters(c=d/2, wc=8.6e-3, R=outer_radius, v1=wedge_material.cp, v2=coupling_medium.cp)
    alpha_max = np.radians(50)
    alpha_0 = 0


    # Delay-law related:
    focus_alpha_step = np.radians(.5)
    focus_alpha_max= np.radians(45)
    focus_alpha_min = np.radians(-45)
    focus_radius = 50e-3 # not necessary if plane_wave is selected.

    # Define reflector positions for the PSF evaluation:
    r_scatterer = 60e-3
    # theta_scatterer = np.radians(np.arange(-40, 40 + 1, 1))
    theta_scatterer = np.radians(np.arange(0, 40 + 1, 1))

    # Configure simulation:
    loss_model = LossModel(transmission=True, transmission_diffraction=True, reception_diffraction=True, attenuation=True)

    # Create pipeline:
    pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center=(-5e-3, 0))
    pipeline_shifted = Pipeline(pipe_material, outer_radius, wall_thickness, (-5e-3, 0))

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
        delay_law = generate_focal_law("focused_wave", focus_radius, focus_angle, acoustic_lens, pipeline, transducer)

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

        time_shift = 0

        x_scatterer, z_scatterer = r_scatterer * np.sin(theta_scatterer) + pipeline.xcenter, r_scatterer * np.cos(theta_scatterer) + pipeline.zcenter

        sim_amplitude = np.zeros(shape=(len(theta_scatterer)))
        sim_fwhm_w = np.zeros(shape=(len(theta_scatterer)))
        sim_fwhm_h = np.zeros(shape=(len(theta_scatterer)))
        sim_api = np.zeros(shape=(len(theta_scatterer)))

        for i in tqdm(range(len(theta_scatterer)), desc="Running simulations"):
            sim_fwhm_w[i], sim_fwhm_h[i], sim_amplitude[i], sscan, mask, sim_api[i] = simulate_single_scatterer(
                x_scatterer[i], z_scatterer[i], simulation, raytracer, delay_law, radius_span, focus_angle)
            sscan_env = envelope(sscan, axis=0)[..., 0]

            if i == 0:
                ref_amplitude = sscan_env.max()

            sscan_db = np.log10(sscan_env / sscan_env.max() + 1E-6)

            if theta_scatterer[i] == np.degrees(0):
                plot_sscan(sscan_db, tbeg=simulation.gate_start, tend=simulation.gate_end, insp_type="sim")

    #%% Compute the **Experimental** results:
    if COMPUTE_EXP_RESULTS:
        from framework import file_m2k

        n_reps = 2
        n_pos = len(theta_scatterer)

        exp_amplitude = np.zeros(shape=(n_reps, n_pos))
        exp_fwhm_w = np.zeros(shape=(n_reps, n_pos))
        exp_fwhm_h = np.zeros(shape=(n_reps, n_pos))
        exp_api = np.zeros(shape=(n_reps, n_pos))

        data_path = project_root / "data"
        data_ref = file_m2k.read(str(data_path / f"multisalvo_xl_deslocado_-5mm_ref.m2k"), freq_transd=central_freq,
                      bw_transd=bw, tp_transd=pulse_type)
        ref_channels = np.mean(data_ref[1].ascan_data, axis=3)
        for i in tqdm(range(len(theta_scatterer)), desc="Processing experimental acquisitions"):

            for rep in range(n_reps):
                lado = "esquerda" if theta_scatterer[i] < 0 else "direita"
                shot = abs(int(np.degrees(theta_scatterer[i])))
                data_insp = file_m2k.read(str(data_path / f"multisalvo_xl_deslocado_-5mm_{lado}_v{rep + 1}.m2k"), freq_transd=central_freq,
                                          bw_transd=bw, tp_transd=pulse_type, sel_shots=shot)[1]
                sscan = np.sum(data_insp.ascan_data[..., 0] - ref_channels, axis=2)
                sscan_env = envelope(sscan, axis=0)
                time_grid = data_insp.time_grid[..., 0] * 1e-6
                time_shift = 1.1e-6

                # plt.figure()
                # plt.imshow(sscan_env, extent=(focus_angle[0], focus_angle[-1], time_grid[-1] * 1e6, time_grid[0] * 1e6), aspect='auto')
                # plt.show(block=True)

                radius_span = convert_time2radius(
                    tspan=time_grid,
                    t_outer=53.2e-6,
                    t_inner=60.05e-6,
                    cl_surf=pipeline.material.cp,
                    cl_upper=acoustic_lens.coupling_medium.cp,
                    cl_bottom=acoustic_lens.coupling_medium.cp,
                    outer_r=pipeline.outer_radius
                )

                window_corners = [
                    (68e-3, theta_scatterer[i] - np.radians(5)),  # north-east
                    (55e-3, theta_scatterer[i] + np.radians(12))  # south-west
                ]

                exp_fwhm_w[rep, i], exp_fwhm_h[rep, i], exp_amplitude[rep, i], mask, exp_api[rep, i] = fwhm(sscan_env, radius_span, focus_angle, window_corners, drawSquare=False)

                if i == 0:
                    ref_amplitude = sscan_env.max()

                sscan_db = np.log10(sscan_env / sscan_env.max() + 1E-6)

                if theta_scatterer[i] == np.degrees(0) and rep == 0:

                    row_beg, row_end = np.searchsorted(radius_span[::-1], 56e-3), np.searchsorted(radius_span[::-1], 64e-3)
                    col_beg, col_end = np.searchsorted(focus_angle, np.radians(-35)), np.searchsorted(focus_angle, np.radians(-15))
                    row_beg = len(radius_span) - row_beg
                    row_end = len(radius_span) - row_end

                    sscan_signal = np.copy(sscan_env)
                    sscan_signal[~mask] = np.nan

                    noise_mask = np.zeros_like(sscan_env, dtype=bool)
                    noise_mask[row_end:row_beg, col_beg:col_end] = True

                    signal_mean = np.nanmean(sscan_signal)
                    noise_mean = np.nanmean(sscan_env[noise_mask])
                    noise_std = np.nanstd(sscan_env[noise_mask])

                    noise_signal = np.copy(sscan_env)
                    noise_signal[~noise_mask] = np.nan

                    fig, axs = plt.subplots(nrows=1, ncols=3)
                    axs[0].imshow(sscan_env, aspect='auto', interpolation='none')
                    axs[1].imshow(sscan_signal, aspect='auto', interpolation='none')
                    axs[2].imshow(noise_signal, aspect='auto', interpolation='none')
                    plt.show(block=True)

                    print(f"CNR = {20 * np.log10(np.abs(signal_mean - noise_mean) / noise_std):.2f} dB")


                if theta_scatterer[i] == np.degrees(0):
                    touter = 51.98e-6
                    tinner = 60.20e-6
                    plot_sscan(sscan_db, tbeg=time_grid[0], tend=time_grid[-1], touter = touter, tinner = tinner, insp_type="exp")



        std_exp_amplitude = np.std(exp_amplitude, axis=0)
        std_exp_fwhm_w = np.std(exp_fwhm_w, axis=0)
        std_exp_fwhm_h = np.std(exp_fwhm_h, axis=0)
        std_exp_api = np.std(exp_api, axis=0)

        mean_exp_amplitude = np.mean(exp_amplitude, axis=0)
        mean_exp_fwhm_w = np.mean(exp_fwhm_w, axis=0)
        mean_exp_fwhm_h = np.mean(exp_fwhm_h, axis=0)
        mean_exp_api = np.mean(exp_api, axis=0)



    #%%
    # FWHM Vertical:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if COMPUTE_SIM_RESULTS:
        # Vertical:
        ax.plot(np.degrees(theta_scatterer), sim_api * 1e6, '-k', label="Simulated", lw=3)
    if COMPUTE_EXP_RESULTS:
        # Vertical:
        ax.plot(np.degrees(theta_scatterer), mean_exp_api * 1e6, '-o', color='k', label="Experimental", lw=1, ms=2)
        ax.fill_between(
        x=np.degrees(theta_scatterer),
        y1=(mean_exp_api - std_exp_api) * 1e6,
        y2=(mean_exp_api + std_exp_api) * 1e6,
        color='k', alpha=.25)

    ax.set_xticks(np.arange(-40, 40 + 10, 10))
    ax.set_xlabel("Reflector angular position / (degrees)")
    ax.set_ylabel(r"Reflector Area / ($\mathrm{mm}^2$)")

    # if not PLOT_FOR_PAPER:
    ax.legend()

    ax.grid()
    plt.tight_layout()
    plt.savefig(project_root / "figures" / "shifted_api.pdf")
    plt.show()
    #%%

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if COMPUTE_SIM_RESULTS:
        ax.plot(np.degrees(theta_scatterer), np.asarray(sim_amplitude) / np.max(sim_amplitude) * 100, '-k', label="Simulated Amplitude", lw=3)
    if COMPUTE_EXP_RESULTS:
        ax.plot(np.degrees(theta_scatterer), np.asarray(mean_exp_amplitude) / np.max(mean_exp_amplitude) * 100, '-o', color='k', lw=1, ms=2)
        ax.fill_between(
        x=np.degrees(theta_scatterer),
        y1=(mean_exp_amplitude - std_exp_amplitude) / np.max(mean_exp_amplitude) * 100,
        y2=(mean_exp_amplitude + std_exp_amplitude) / np.max(mean_exp_amplitude) * 100,
        color='k', alpha=.25)

    ax.set_xticks(np.arange(-40, 40 + 10, 10))
    ax.set_xlabel("Reflector angular position / (degrees)")
    ax.set_ylabel("Normalized Amplitude / (%)")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid()
    plt.tight_layout()
    plt.savefig(project_root / "figures" / "shifted_amplitude_vs_angle.pdf")
    plt.show()