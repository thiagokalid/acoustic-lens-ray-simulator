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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "experiment_1"
COMPUTE_SIM_RESULTS = True
COMPUTE_EXP_RESULTS = False
FWHM_THRESHOLD_DB = -6

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
        for delay_law_type in ["focused_wave", "plane_wave"]:
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

                sscan = simulate_single_scatterer(x_scatterer[i], z_scatterer[i], simulation, delay_law)
                sscan_env = envelope(sscan, axis=0)[..., 0]

                window_corners = [
                    (70e-3, theta_scatterer[i] - np.radians(14)),  # north-east
                    (45e-3, theta_scatterer[i] + np.radians(14))  # south-west
                ]

                sim_fwhm_w[i], sim_fwhm_h[i], sim_amplitude[i], mask, flaw_area = fwhm(sscan_env, radius_span, focus_angle, window_corners,
                                                            drawSquare=False, thresh=FWHM_THRESHOLD_DB)

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
                CNRs = np.zeros(n_reps, dtype=np.float64)
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
                            t_outer=54.74e-6,
                            t_inner=60.49e-6,
                            cl_surf=pipeline.material.cp,
                            cl_upper=acoustic_lens.coupling_medium.cp,
                            cl_bottom=acoustic_lens.coupling_medium.cp,
                            outer_r=pipeline.outer_radius
                        )
                    else:
                        radius_span = convert_time2radius(
                            tspan=time_grid,
                            t_outer=54.74e-6,
                            t_inner=60.49e-6,
                            cl_surf=pipeline.material.cp,
                            cl_upper=acoustic_lens.coupling_medium.cp,
                            cl_bottom=acoustic_lens.coupling_medium.cp,
                            outer_r=pipeline.outer_radius
                        )

                    window_corners = [
                        (64.0e-3, theta_scatterer[i] - np.radians(10)),  # north-east
                        (57.0e-3, theta_scatterer[i] + np.radians(20))  # south-west
                    ]

                    exp_fwhm_w[rep, i], exp_fwhm_h[rep, i], exp_amplitude[rep, i], mask, _ = fwhm(sscan_env, radius_span, focus_angle, window_corners, drawSquare=False, thresh=FWHM_THRESHOLD_DB)


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
                    if theta_scatterer[i] == np.degrees(0) and rep == 0:
                        extent = (np.degrees(focus_angle[0]), np.degrees(focus_angle[-1]), time_grid[-1] * 1e6, time_grid[0] * 1e6)

                        fig, axs = plt.subplots(nrows=1, ncols=3)
                        manager = plt.get_current_fig_manager()
                        manager.full_screen_toggle()

                        axs[0].imshow(sscan_env, extent=extent, aspect='auto', interpolation='none')
                        axs[1].imshow(mask, extent=extent, aspect='auto', interpolation='none')
                        axs[1].axvline(np.degrees(theta_scatterer[i]), ls='--', lw=.5, color='r')

                        axs[2].imshow(noise_signal, extent=extent, aspect='auto', interpolation='none')
                        plt.show(block=False)
                    CNRs[rep] = 20 * np.log10(np.abs(signal_mean - noise_mean) / noise_std)
                print(f"CNR mean = {np.mean(CNRs)} dB")
                print(f"CNR std = {np.std(CNRs)} dB")

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