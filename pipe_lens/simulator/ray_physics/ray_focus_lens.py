# built-in modules
from typing import List, Tuple

# scientific-computing modules
import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from scipy.optimize import minimize

# modules implemented in this repo
from pipe_lens.simulator.ray_physics.ray_lens_physics import RayLensPhysics
from pipe_lens.types import Fluid, Solid, Ray
from pipe_lens.simulator.ray_physics.utils import snell, uhp, roots_bhaskara
from pipe_lens.simulator.bulkwave_physics.diffraction import far_field_directivity_solid
from pipe_lens.simulator.bulkwave_physics.transmission import compute_tension_transmission_coefficient, compute_tension_reflection_coefficient

__all__ = ['RayFocusThroughLens']

class RayFocusThroughLens(RayLensPhysics):
    """
    Classe que implementa a física de propagação de raios considerando uma lente acústica e 
    apenas uma refração na lente.

    """
    def build_measurement_model(self, omega, rays: List[Ray], gate_start=0.):

        n_samples = omega.shape[0]
        n_elem = self.transducer.num_elem

        Ta = np.ones(shape=(n_samples, n_elem, n_elem), dtype=np.complex64)

        # time-of-flight delay:
        P = self.build_propagation_model(omega, rays, self.loss_model.propagation)

        # transmission diffraction correction:
        C1 = self.build_diffraction_tx_model(omega, rays, self.loss_model.transmission_diffraction)

        # reception diffraction + spatial averaging correction:
        C2 = self.build_diffraction_rx_model(omega, rays, self.loss_model.reception_diffraction)

        # spatial attenuation:
        M = self.build_attenuation_model(omega, rays, self.loss_model.attenuation)

        # Computing amplitude:
        for tx in range(n_elem):
            # transmission coefficient:
            T1, _ = self.build_transmission_model(omega, rays[tx], self.loss_model.transmission)
            for rx in range(n_elem):
                # transmission coefficient:
                _, T2 = self.build_transmission_model(omega, rays[rx], self.loss_model.transmission)

                Ta[:, tx, rx] *= P[:, tx, rx] * T1 * T2 * C1[:, tx] * C2[:, rx] * M[:, tx, rx]

        return Ta

    def build_attenuation_model(self, w, rays: List[Ray], flag):
        d_i = np.zeros(shape=(w.shape[0], self.transducer.num_elem, self.transducer.num_elem), dtype=np.complex64)
        for tx, ray_tx in enumerate(rays):
            for rx, ray_rx in enumerate(rays):
                d_i[:, tx, rx] = ray_tx.compute_dist() + ray_rx.compute_dist()

        return 1/np.sqrt(d_i)

    def build_transmission_model(self, omega, rays: Ray, flag: bool):
        if flag:
            # Reflection Aluminium to water:
            R11 = compute_tension_reflection_coefficient(
                inc_ang=rays.ang_i[0], trans_ang=rays.ang_t[0],
                medium1=self.acoustic_lens.wedge_medium, medium2=self.acoustic_lens.coupling_medium,
                inc_mode="L", reflect_mode="L"
            )

            # Transmission Aluminium to water:
            T12 = compute_tension_transmission_coefficient(
                inc_ang=rays.ang_i[1], trans_ang=rays.ang_t[1],
                medium1=self.acoustic_lens.wedge_medium, medium2=self.acoustic_lens.coupling_medium,
                inc_mode="L", trans_mode="L"
            )

            # Transmission Water to steel:
            T23 = compute_tension_transmission_coefficient(
                inc_ang=rays.ang_i[2], trans_ang=rays.ang_t[2],
                medium1=self.acoustic_lens.coupling_medium, medium2=self.pipeline.material,
                inc_mode="L", trans_mode="L"
            )

            T1 = R11 * T12 * T23

            # Steel to water:
            T32 = compute_tension_transmission_coefficient(
                inc_ang=rays.ang_t[2], trans_ang=rays.ang_i[2],
                medium1=self.pipeline.material, medium2=self.acoustic_lens.coupling_medium,
                inc_mode="L", trans_mode="L"
            )

            # Water to aluminium:
            T21 = compute_tension_transmission_coefficient(
                inc_ang=rays.ang_t[1], trans_ang=rays.ang_i[1],
                medium1=self.acoustic_lens.coupling_medium, medium2=self.acoustic_lens.wedge_medium,
                inc_mode="L", trans_mode="L"
            )

            R11 = compute_tension_reflection_coefficient(
                inc_ang=rays.ang_t[0], trans_ang=rays.ang_i[0],
                medium1=self.acoustic_lens.wedge_medium, medium2=self.acoustic_lens.coupling_medium,
                inc_mode="L", reflect_mode="L"
            )

            T2 = T32 * T21 * R11
        else:
            T1 = T2 = 1.

        return T1, T2

    def build_diffraction_tx_model(self, omega, rays: List[Ray], flag: bool):
        Nrays = int(len(rays))
        if flag:
            thetas = np.zeros(Nrays)
            for i, ray in enumerate(rays):
                thetas[i] = np.pi/2 - np.arctan2(np.abs(ray.zray[0] - ray.zray[1]), np.abs(ray.xray[0] - ray.xray[1]))
            return far_field_directivity_solid(2 * np.pi * self.transducer.fc, thetas[:, np.newaxis], self.transducer.element_width, self.acoustic_lens.wedge_medium.cp, self.acoustic_lens.wedge_medium.cs).T
        else:
            return np.ones(shape=(omega.shape[0], Nrays))

    def build_diffraction_rx_model(self, omega, rays: List[Ray], flag):
        return self.build_diffraction_tx_model(omega, rays, flag)

    def compute_tofs(self, solution):
        n_elem = self.transducer.num_elem
        n_focii = solution[0].n_focus


        coord_elements = np.array([self.transducer.xt, self.transducer.zt]).T
        coords_reflectors = np.array([solution[0].xray[-1], solution[0].zray[-1]]).T
        coords_lens = np.zeros((n_elem, 2, n_focii))
        coords_outer = np.zeros((n_elem, 2, n_focii))

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            coords_lens[j, 0, i], coords_lens[j, 1, i] = solution[j].xray[1][i], solution[j].zray[1][i]
            coords_outer[j, 0, i], coords_outer[j, 1, i] = solution[j].xray[2][i], solution[j].zray[2][i]

        coord_elements_mat = np.tile(coord_elements[:, :, np.newaxis], (1, 1, n_focii))
        coord_reflectors_mat = np.tile(coords_reflectors[:, :, np.newaxis], (1, 1, n_elem))

        d1 = norm(coords_lens - coord_elements_mat, axis=1)
        d2 = norm(coords_lens - coords_outer, axis=1)
        d3 = norm(coords_outer - coord_reflectors_mat.T, axis=1)

        oneway_tofs = (d1 / self.c1 + d2 / self.c2 + d3 / self.c3)
        roundtrip_tofs = (
                oneway_tofs[:, None, :] +
                oneway_tofs[None, :, :]
        )
        return roundtrip_tofs.T

    def _dist_kernel(self, xc: float, zc: float, xf: ndarray, yf: ndarray, acurve: float):

        # Ponto x,y da lente avaliado em relação a alfa.
        xlens, ylens = self.acoustic_lens.xy_from_alpha(acurve)

        # Ângulo de incidência na lente com relação ao eixo x
        gamma1 = np.arctan2((ylens - zc), (xlens - xc))
        gamma1 = gamma1 + (gamma1 < 0) * np.pi

        # Ângulo refratado em relação ao eixo x
        gamma2, inc12, ref12 = snell(self.c1, self.c2, gamma1, self.acoustic_lens.dydx_from_alpha(acurve))

        # Coeficientes da reta r1(x) = a_line + x + b_line
        # que coincide com o raio refratado na água:
        a_line = np.tan(uhp(gamma2))
        b_line = ylens - a_line * xlens

        # Calculando a intersecção entre a reta r1(x) e a circunferência dada pelo raio do tubo.
        a = a_line**2 + 1
        b = 2 * a_line * b_line - 2 * (self.pipeline.xcenter + a_line * self.pipeline.zcenter)
        c = b_line ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_line - self.pipeline.outer_radius ** 2

        # Esocolhe o primeir ângulo de incidência:
        xcirc1, xcirc2 = roots_bhaskara(a, b, c)
        ycirc1, ycirc2 = a_line * xcirc1 + b_line, a_line * xcirc2 + b_line
        upper = ycirc1 > ycirc2
        xcirc = xcirc1 * upper + xcirc2 * (1 - upper)
        ycirc = ycirc1 * upper + ycirc2 * (1 - upper)

        # Ângulo de refração, em relação ao eixo x, do raio refratado dentro do tubo:
        gamma3, inc23, ref23 = snell(self.c2, self.c3, gamma2, self.pipeline.dydx(xcirc))
        a3 = np.tan(gamma3)
        b3 = ycirc - a3 * xcirc

        # Reta perpendicular ao raio rafratado dentro do tubo:
        xbottom = -b3 / a3
        a4 = -1 / a3
        b4 = yf - a4 * xf

        xin = (b4 - b3) / (a3 - a4)
        yin = a3 * xin + b3
        dist = (xin - xf)**2 + (yin - yf)**2


        xc_mtx = xc * np.ones_like(xlens)
        zc_mtx = zc * np.ones_like(xlens)

        ang_f = np.pi/2 * np.ones_like(xlens)

        return Ray(
            xray = [xc_mtx, xlens, xcirc, xin],
            zray = [zc_mtx, ylens, ycirc, yin],
            dist2focus = dist,
            ang_i = [inc12, inc23, ang_f],
            ang_t = [ref12, ref23, ang_f],
            medium = (self.medium1, self.medium2, self.medium3),
            modes= self.modes
        )

    def _dist_kernel2(self, xc: float, zc: float, steering_angle: float):
        steering_angle_2 = np.pi/2 - steering_angle

        line_cart = lambda x : np.tan(steering_angle_2) * (x - xc) + zc
        #
        import matplotlib.pyplot as plt
        # plt.gca().plot(self.acoustic_lens.xlens, line_cart(self.acoustic_lens.xlens), color='lime', lw=1.5)

        cost_fun = lambda xx: np.power(line_cart(xx) - self.acoustic_lens.y_from_x(xx, safe=False), 2)
        sol = minimize(cost_fun, np.arctan2(zc, xc), method='L-BFGS-B', bounds=[(np.min(self.acoustic_lens.xlens), np.max(self.acoustic_lens.xlens))])
        print(sol.success)
        xcurve = sol.x
        acurve = self.acoustic_lens.alpha_from_x(xcurve)[0]

        # Ponto x,y da lente avaliado em relação a alfa.
        xlens, ylens = self.acoustic_lens.xy_from_alpha(acurve, safe=False)

        plt.gca().plot(xlens, ylens, 'ob')

        # Ângulo de incidência na lente com relação ao eixo x
        gamma1 = np.arctan2((ylens - zc), (xlens - xc))
        gamma1 = gamma1 + (gamma1 < 0) * np.pi

        # Ângulo refratado em relação ao eixo x
        gamma2, inc12, ref12 = snell(self.c1, self.c2, gamma1, self.acoustic_lens.dydx_from_alpha(acurve, safe=False))

        # Coeficientes da reta r1(x) = a_line + x + b_line
        # que coincide com o raio refratado na água:
        a_line = np.tan(uhp(gamma2))
        b_line = ylens - a_line * xlens

        # Calculando a intersecção entre a reta r1(x) e a circunferência dada pelo raio do tubo.
        a = a_line**2 + 1
        b = 2 * a_line * b_line - 2 * (self.pipeline.xcenter + a_line * self.pipeline.zcenter)
        c = b_line ** 2 + self.pipeline.xcenter ** 2 + self.pipeline.zcenter ** 2 - 2 * self.pipeline.zcenter * b_line - self.pipeline.outer_radius ** 2

        # Esocolhe o primeir ângulo de incidência:
        xcirc1, xcirc2 = roots_bhaskara(a, b, c)
        ycirc1, ycirc2 = a_line * xcirc1 + b_line, a_line * xcirc2 + b_line
        upper = ycirc1 > ycirc2
        xcirc = xcirc1 * upper + xcirc2 * (1 - upper)
        ycirc = ycirc1 * upper + ycirc2 * (1 - upper)

        # Ângulo de refração, em relação ao eixo x, do raio refratado dentro do tubo:
        gamma3, inc23, ref23 = snell(self.c2, self.c3, gamma2, self.pipeline.dydx(xcirc))
        a3 = np.tan(gamma3)
        b3 = ycirc - a3 * xcirc


        xc_mtx = xc * np.ones_like(xlens)
        zc_mtx = zc * np.ones_like(xlens)

        ang_f = np.pi/2 * np.ones_like(xlens)

        return Ray(
            xray = [xc_mtx, xlens, xcirc, a3 * 0 + b3],
            zray = [zc_mtx, ylens, ycirc, a3 * 0 + b3],
            dist2focus = 1,
            ang_i = [inc12, inc23, ang_f],
            ang_t = [ref12, ref23, ang_f],
            medium = (self.medium1, self.medium2, self.medium3),
            modes= self.modes
        )

