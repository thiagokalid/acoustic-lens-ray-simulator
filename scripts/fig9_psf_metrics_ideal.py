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
import psutil
import os
import time

# Constants:
LATEX_LINEWIDTH = 6.3091141732 # LaTeX linewidth
cm = 1 / 2.54
FIGSIZE = (7 * cm, 6 * cm)
CROPPED_SSCAN_FIGSIZE = (2, 2)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_PATH = PROJECT_ROOT / "figures"
DATA_PATH = PROJECT_ROOT / "data" / "experiment_1"
COMPUTE_SIM_RESULTS = False
COMPUTE_EXP_RESULTS = True
SAVE_FIGS = True

def plot_experimental_data(ax, thetas, mean_values, std_values, color='k', style='o-', label="_"):
    ax.plot(thetas, mean_values, style, color=color, lw=1, ms=2.25, label=label)
    ax.fill_between(
        x=thetas,
        y1=mean_values + std_values,
        y2=mean_values - std_values,
        alpha=.25, color=color
    )

    ax.set_xticks(np.arange(0, 40 + 5, 5))

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
    steel = Solid(cp=5900, cs=3200, density=7850)
    aluminium = Solid(cp=6332.94, cs=3128.18, density=2730)
    water = Fluid(cp=1430, density=997)

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
    pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center=(0, 0))
    pipeline_shifted = Pipeline(pipe_material, outer_radius, wall_thickness, (0, 0))

    # Create acoustic lens:
    acoustic_lens = AcousticLens(wedge_material, coupling_medium, d, alpha_max, alpha_0, h0, outer_radius)

    # Create transducer:
    transducer = Transducer(pitch, num_elem, element_gap, central_freq, bw, bwr, pulse_type, center=(0., acoustic_lens.d))

    # Define location for the focuses:
    focus_angle = np.arange(focus_alpha_min, focus_alpha_max + focus_alpha_step, focus_alpha_step)

    print(f"Maximum firing angle: {np.rad2deg(acoustic_lens.pipe2steering_angle(focus_alpha_max)):.2f}."
          f"Minimum firing angle: {np.rad2deg(acoustic_lens.pipe2steering_angle(focus_alpha_min)):.2f}")
    #%% Compute the **Simulated** results:

    # Objects for the simulation:
    physics = RayFocusThroughLens(acoustic_lens, pipeline_shifted, transducer, loss_model)
    optimizer = NewtonOptimizer(alpha_max=alpha_max, alpha_min=acoustic_lens.alpha_min, max_iter=15, eps=1e-4)
    raytracer = RaytracerScenario(physics, optimizer)
    simulation = Simulator(gate_start=40e-6, gate_end=80e-6, fs=62.5e6, raytracer=raytracer)

    if COMPUTE_SIM_RESULTS:
        delay_law = generate_focal_law(delay_law_type, focus_radius, focus_angle, acoustic_lens, pipeline, transducer)

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



        process = psutil.Process(os.getpid())

        runtime = []
        memory_usage = []

        for i in tqdm(range(len(theta_scatterer)), desc="Running simulations"):
            start_time = time.perf_counter()

            sim_fwhm_w[i], sim_fwhm_h[i], sim_amplitude[i], sscan, mask, _ = simulate_single_scatterer(
                x_scatterer[i], z_scatterer[i], simulation, raytracer, delay_law, radius_span, focus_angle)
            sscan_env = envelope(sscan, axis=0)[..., 0]

            # Runtime of this iteration
            elapsed_time = time.perf_counter() - start_time
            runtime.append(elapsed_time)

            # RAM currently used by this Python process
            memory_usage.append(process.memory_info().rss / 1024 ** 2)  # MB

            time_shift = 0


        print("Average runtime per iteration: ", np.mean(runtime))
        print("Average memory usage per iteration: ", np.mean(memory_usage))
        np.save(project_root / "results" / f"sim_amplitude_{delay_law_type}.npy", sim_amplitude)
        np.save(project_root / "results" / f"sim_fwhm_w_{delay_law_type}.npy", sim_fwhm_w)
        np.save(project_root / "results" / f"sim_fwhm_h_{delay_law_type}.npy", sim_fwhm_h)

    #%% Compute the **Experimental** results:
    if COMPUTE_EXP_RESULTS:
        from framework import file_m2k

        for delay_law_type in ["focused_wave", "plane_wave"]:
            inspection_index = 1 if delay_law_type == "focused_wave" else 2

            n_reps = 5
            n_pos = len(theta_scatterer)

            exp_amplitude = np.zeros(shape=(n_reps, n_pos))
            exp_fwhm_w = np.zeros(shape=(n_reps, n_pos))
            exp_fwhm_h = np.zeros(shape=(n_reps, n_pos))


            data_ref = file_m2k.read(str(DATA_PATH / f"ref.m2k"), freq_transd=central_freq, bw_transd=bw,
                                     tp_transd=pulse_type)[inspection_index]
            channels_ref = np.mean(data_ref.ascan_data, axis=-1)


            for i in tqdm(range(n_pos), desc="Processing experimental acquisitions"):
                for rep in range(n_reps):
                    shot = int(np.degrees(theta_scatterer[i]))


                    data_insp = file_m2k.read(str(DATA_PATH / f"psf_v{rep + 1}.m2k"), freq_transd=central_freq,
                                              bw_transd=bw, tp_transd=pulse_type, sel_shots=shot)[inspection_index]
                    channels = data_insp.ascan_data[..., 0]

                    sscan = np.sum(channels - channels_ref, axis=2)

                    sscan_env = envelope(sscan, axis=0)
                    time_grid = data_insp.time_grid[..., 0] * 1e-6

                    if delay_law_type == "focused_wave":
                        radius_span = convert_time2radius(
                            tspan=time_grid,
                            t_outer=53.4e-6,
                            t_inner=58.9e-6,
                            cl_surf=pipeline.material.cp,
                            cl_upper=acoustic_lens.coupling_medium.cp,
                            cl_bottom=acoustic_lens.coupling_medium.cp,
                            outer_r=pipeline.outer_radius
                        )
                    else:
                        radius_span = convert_time2radius(
                            tspan=time_grid,
                            t_outer=53.12e-6,
                            t_inner=58.7e-6,
                            cl_surf=pipeline.material.cp,
                            cl_upper=acoustic_lens.coupling_medium.cp,
                            cl_bottom=acoustic_lens.coupling_medium.cp,
                            outer_r=pipeline.outer_radius
                        )

                    window_corners = [
                        (63.0e-3, theta_scatterer[i] - np.radians(10)),  # north-east
                        (57e-3, theta_scatterer[i] + np.radians(20))  # south-west
                    ]

                    exp_fwhm_w[rep, i], exp_fwhm_h[rep, i], exp_amplitude[rep, i], mask, _ = fwhm(sscan_env, radius_span, focus_angle, window_corners, drawSquare=False,)

                    if theta_scatterer[i] == np.degrees(0) and rep == 0:
                        row_beg, row_end = np.searchsorted(radius_span[::-1], 57e-3), np.searchsorted(radius_span[::-1], 62e-3)
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

                        extent = (np.degrees(focus_angle[0]), np.degrees(focus_angle[-1]), time_grid[-1] * 1e6, time_grid[0] * 1e6)

                        fig, axs = plt.subplots(nrows=1, ncols=3)
                        axs[0].imshow(sscan_env, extent=extent, aspect='auto', interpolation='none')
                        axs[1].imshow(sscan_signal, extent=extent, aspect='auto', interpolation='none')
                        axs[1].axvline(np.degrees(theta_scatterer[i]), ls='--', lw=.5, color='r')

                        axs[2].imshow(noise_signal, extent=extent, aspect='auto', interpolation='none')
                        plt.show(block=False)

                        print(f"CNR = {20 * np.log10(np.abs(signal_mean - noise_mean) / noise_std):.2f} dB")

            std_exp_amplitude = np.std(exp_amplitude, axis=0)
            std_exp_fwhm_w = np.std(exp_fwhm_w, axis=0)
            std_exp_fwhm_h = np.std(exp_fwhm_h, axis=0)

            mean_exp_amplitude = np.mean(exp_amplitude, axis=0)
            mean_exp_fwhm_w = np.mean(exp_fwhm_w, axis=0)
            mean_exp_fwhm_h = np.mean(exp_fwhm_h, axis=0)

            np.save(project_root / "results" / f"mean_exp_amplitude_{delay_law_type}.npy", mean_exp_amplitude)
            np.save(project_root / "results" / f"mean_exp_fwhm_w_{delay_law_type}.npy", mean_exp_fwhm_w)
            np.save(project_root / "results" / f"mean_exp_fwhm_{delay_law_type}.npy", mean_exp_fwhm_h)
            np.save(project_root / "results" / f"std_exp_amplitude_{delay_law_type}.npy", std_exp_amplitude)
            np.save(project_root / "results" / f"std_exp_fwhm_w_{delay_law_type}.npy", std_exp_fwhm_w)
            np.save(project_root / "results" / f"std_exp_fwhm_h_{delay_law_type}.npy", std_exp_fwhm_h)

    #%%
    #
    # exp_amplitude_maximum = np.max([
    #     mean_exp_amplitude_focused_wave,
    #     mean_exp_amplitude_plane_wave
    # ])
    #
    # sim_amplitude_maximum = np.max([
    #     sim_amplitude_plane_wave, sim_amplitude_focused_wave
    # ])
    #
    # # Transform horizontal fwhm to degrees:
    # sim_fwhm_w_plane_wave = np.degrees(sim_fwhm_w_plane_wave)
    # sim_fwhm_w_focused_wave = np.degrees(sim_fwhm_w_focused_wave)
    # mean_exp_fwhm_w_plane_wave = np.degrees(mean_exp_fwhm_w_plane_wave)
    # std_exp_fwhm_w_plane_wave = np.degrees(std_exp_fwhm_w_plane_wave)
    # mean_exp_fwhm_w_focused_wave = np.degrees(mean_exp_fwhm_w_focused_wave)
    # std_exp_fwhm_w_focused_wave = np.degrees(std_exp_fwhm_w_focused_wave)
    #
    # # Transform vertical fwhm to mm:
    # sim_fwhm_h_plane_wave *= 1e3
    # sim_fwhm_h_focused_wave *= 1e3
    # std_exp_fwhm_h_plane_wave *= 1e3
    # mean_exp_fwhm_h_plane_wave *= 1e3
    # std_exp_fwhm_h_focused_wave *= 1e3
    # mean_exp_fwhm_h_focused_wave *= 1e3
    #
    # # Normalize amplitude based on common references:
    # sim_amplitude_focused_wave = sim_amplitude_focused_wave / sim_amplitude_maximum * 100
    # sim_amplitude_plane_wave = sim_amplitude_plane_wave / sim_amplitude_maximum * 100
    # mean_exp_amplitude_focused_wave = mean_exp_amplitude_focused_wave / exp_amplitude_maximum * 100
    # std_exp_amplitude_focused_wave = std_exp_amplitude_focused_wave / exp_amplitude_maximum * 100
    # mean_exp_amplitude_plane_wave = mean_exp_amplitude_plane_wave / exp_amplitude_maximum * 100
    # std_exp_amplitude_plane_wave = std_exp_amplitude_plane_wave / exp_amplitude_maximum * 100
    #
    # theta_scatterer = np.radians(np.arange(0, 40 + 1, 1))
    #
    # # %%
    # thetas_deg = np.degrees(theta_scatterer)
    #
    # fig, ax = plt.subplots(figsize=FIGSIZE)
    #
    # # Horizontal Onda Plana:
    # ax.plot(np.degrees(theta_scatterer), sim_fwhm_w_plane_wave, '-', color='#ff1f23', lw=2)
    # plot_experimental_data(ax, thetas_deg, mean_exp_fwhm_w_plane_wave, std_exp_fwhm_w_plane_wave, style='x-',
    #                        color='#FF1F5B')
    #
    # # Horizontal Onda Focada:
    # ax.plot(np.degrees(theta_scatterer), sim_fwhm_w_focused_wave, '-', color='k', lw=2)
    # plot_experimental_data(ax, thetas_deg, mean_exp_fwhm_w_focused_wave, std_exp_fwhm_w_focused_wave, style='o-',
    #                        color='k')
    #
    # ax.set_xticks(np.arange(0, 40 + 5, 5))
    #
    # ax.set_xlabel("Reflector angular position / (degrees)")
    # ax.set_ylabel(r"Width / (degrees)")
    #
    # ax.grid()
    # plt.tight_layout()
    # plt.savefig(project_root / "figures" / "api_fwhm_horizontal.pdf")
    # plt.show()
    #
    # # %%%
    #
    # fig, ax = plt.subplots(figsize=FIGSIZE)
    # # Vertical Onda Plana:
    # ax.plot(thetas_deg, sim_fwhm_h_plane_wave, '-', color='#FF1F5B', lw=2)
    # plot_experimental_data(ax, thetas_deg, mean_exp_fwhm_h_plane_wave, std_exp_fwhm_h_plane_wave, style='x-',
    #                        color='#FF1F5B')
    #
    # # Vertical Onda Focada:
    # ax.plot(thetas_deg, sim_fwhm_h_focused_wave, '-', color='k', lw=2)
    # plot_experimental_data(ax, thetas_deg, mean_exp_fwhm_h_focused_wave, std_exp_fwhm_h_focused_wave, style='o-',
    #                        color='k')
    #
    # ax.set_xticks(np.arange(0, 40 + 5, 5))
    # ax.set_xlabel("Reflector angular position / (degrees)")
    # ax.set_ylabel(r"Height / (mm)")
    #
    # ax.grid()
    # plt.tight_layout()
    #
    # plt.savefig(project_root / "figures" / "api_fwhm_vertical.pdf")
    # plt.show()
    #
    # # %%%
    #
    # fig, ax = plt.subplots(figsize=FIGSIZE)
    #
    # # Vertical Onda Plana:
    # ax.plot(np.degrees(theta_scatterer), sim_amplitude_plane_wave, '-', color='#FF1F5B', lw=2)
    # plot_experimental_data(ax, thetas_deg, mean_exp_amplitude_plane_wave, std_exp_amplitude_plane_wave, style='x-',
    #                        color='#FF1F5B')
    #
    # # Vertical Onda Focada:
    # ax.plot(np.degrees(theta_scatterer), sim_amplitude_focused_wave, '-', color='k', lw=2)
    # plot_experimental_data(ax, thetas_deg, mean_exp_amplitude_focused_wave, std_exp_amplitude_focused_wave, style='o-',
    #                        color='k')
    #
    # ax.set_yticks([0, 25, 50, 75, 100])
    # ax.set_xticks(np.arange(0, 40 + 5, 5))
    #
    # ax.set_xlabel("Reflector angular position / (degrees)")
    # ax.set_ylabel(r"Normalized Amplitude / (%)")
    #
    # ax.grid()
    # plt.tight_layout()
    # plt.savefig(project_root / "figures" / "api_fwhm_amplitude.pdf")
    # plt.show()
