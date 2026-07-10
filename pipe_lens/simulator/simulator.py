import numpy as np

from pipe_lens.simulator.raytracer_scenario import RaytracerScenario
from pipe_lens.simulator.utils import fmc_sim_kernel_gaussian
from scipy.signal import hilbert

__all__ = ["Simulator"]

class Simulator:
    def __init__(self, gate_start: float, gate_end: float, fs: float, raytracer: RaytracerScenario | None):
        self.gate_start = gate_start
        self.gate_end = gate_end
        self.fs = fs
        self.raytracer = raytracer
        self.dt = 1/self.fs
        self.tspan = np.arange(self.gate_start, self.gate_end, self.dt)
        self.gate_samples = len(self.tspan)


    def simulate_scatterer(self, xf: float, zf: float) -> np.ndarray:
        if isinstance(xf, float) or isinstance(zf, int):
            xf = np.array([xf])
            zf = np.array([zf])

        opt_rays = self.raytracer.solve(xf, zf)

        # Interval freqs values:
        self.N = len(self.tspan)
        self.f = np.fft.fftfreq(self.N, d=self.dt)
        self.omega = 2 * np.pi * self.f
        self.domega = 2 * np.pi / (self.N * self.dt)


        system_function_w = self.raytracer.physics.transducer.compute_freq_domain_response(self.omega)

        # Compensate for non-zero gate start:
        system_function_w *= np.exp(1j * self.omega * self.gate_start)

        # Continuous to discrete scaling
        system_function_w *= self.domega * self.N / (2 * np.pi)

        # Based on Thompson-Gray ultrasonic measurement model (Schmmer 2.27):
        model = self.raytracer.physics.build_measurement_model(self.omega, opt_rays, self.gate_start)

        # Auld’s measurement model:
        fmc_w = (system_function_w * model.T).T # received-voltage Vo

        # Transform to time-domain signal:
        signal = np.real(np.fft.ifft(fmc_w, axis=0))

        return signal