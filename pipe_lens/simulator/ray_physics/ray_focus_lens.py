# built-in modules
from typing import List, Tuple

# scientific-computing modules
import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from scipy.optimize import minimize

# modules implemented in this repo
from pipe_lens.simulator.ray_physics.ray_lens_physics import RayLensPhysics
from pipe_lens.types import Ray, COMPLEX
from pipe_lens.simulator.ray_physics.utils import snell, uhp, roots_bhaskara
from pipe_lens.simulator.bulkwave_physics import far_field_directivity_solid
from pipe_lens.simulator.bulkwave_physics import compute_tension_transmission_coefficient
from pipe_lens.simulator.bulkwave_physics import two_media_geom_spread


__all__ = ['RayFocusThroughLens']

class RayFocusThroughLens(RayLensPhysics):
    """
    Classe que implementa a física de propagação de raios considerando uma lente acústica e 
    apenas uma refração na lente.

    """
    def build_measurement_model(self, w, rays: List[Ray], gate_start=0.):

        n_samples = w.shape[0]
        n_elem = self.transducer.num_elem

        Ta = np.ones(shape=(n_samples, n_elem, n_elem), dtype=np.complex64)

        # time-of-flight delay:
        P = self.build_propagation_model(w, rays, self.loss_model.propagation)

        # transmission diffraction correction:
        C1 = self.build_diffraction_tx_model(w, rays, self.loss_model.transmission_diffraction)

        # reception diffraction + spatial averaging correction:
        C2 = self.build_diffraction_rx_model(w, rays, self.loss_model.reception_diffraction)

        # spatial attenuation:
        M = self.build_attenuation_model(w, rays, self.loss_model.attenuation)

        # Computing amplitude:
        for tx in range(n_elem):
            # transmission coefficient:
            T1, _ = self.build_transmission_model(w, rays[tx], self.loss_model.transmission)
            for rx in range(n_elem):
                # transmission coefficient:
                _, T2 = self.build_transmission_model(w, rays[rx], self.loss_model.transmission)

                Ta[:, tx, rx] *= P[:, tx, rx] * T1 * T2 * C1[:, tx] * C2[:, rx] * M[:, tx, rx]

        return Ta

    def build_propagation_model(self, w, rays, enabled: bool):
        tof_mtx = self.compute_tofs(rays) # TOF from TX to RX, thus matrix with size TX by RX
        if enabled:
            # compute the phase-term of the propagation model:
            F = np.exp(-1j * w * tof_mtx.T).T

            # compute the geometric-spreading term
            G = np.ones_like(F)
            for tx, ray_tx in enumerate(rays):
                for rx, ray_rx in enumerate(rays):
                    d1_tx, d2_tx, d3_tx = ray_tx.compute_segment_dist()
                    d1_rx, d2_rx, d3_rx = ray_rx.compute_segment_dist()


                    G[:, tx, rx] = np.multiply(
                        # from tx towards the reflector:
                        two_media_geom_spread(d1_tx, d2_tx, ray_tx.ang_i[0], ray_tx.ang_t[0], self.c1, self.c2),
                        two_media_geom_spread(d2_tx, d3_tx, ray_tx.ang_i[1], ray_tx.ang_t[1], self.c2, self.c3)
                    ) * np.multiply(
                        # from the reflector towards the rx:
                        two_media_geom_spread(d3_rx, d2_rx, ray_rx.ang_t[1], ray_tx.ang_i[1], self.c3, self.c2),
                        two_media_geom_spread(d2_rx, d1_rx, ray_rx.ang_t[0], ray_tx.ang_i[0], self.c2, self.c1),
                    )
            return F * G
        else:
            return np.ones_like(w * tof_mtx, dtype=COMPLEX)

    def build_attenuation_model(self, w, rays: List[Ray], enabled):
        M = np.ones(shape=(w.shape[0], self.transducer.num_elem, self.transducer.num_elem), dtype=np.complex64)
        if enabled:
            for tx, ray_tx in enumerate(rays):
                for rx, ray_rx in enumerate(rays):
                    d1_tx, d2_tx, d3_tx = ray_tx.compute_segment_dist()
                    d1_rx, d2_rx, d3_rx = ray_rx.compute_segment_dist()

                    M[:, tx, rx] = 1.
        return M

    def build_transmission_model(self, w, rays: list[Ray] | Ray, enabled: bool):
        if enabled:
            # Transmission Aluminium to water:
            T12 = compute_tension_transmission_coefficient(
                inc_ang=rays.ang_i[0], trans_ang=rays.ang_t[0],
                medium1=self.acoustic_lens.wedge_medium, medium2=self.acoustic_lens.coupling_medium,
                inc_mode="L", trans_mode="L"
            )

            # Transmission Water to steel:
            T23 = compute_tension_transmission_coefficient(
                inc_ang=rays.ang_i[1], trans_ang=rays.ang_t[1],
                medium1=self.acoustic_lens.coupling_medium, medium2=self.pipeline.material,
                inc_mode="L", trans_mode="L"
            )

            T1 = T12 * T23

            # Steel to water:
            T32 = compute_tension_transmission_coefficient(
                inc_ang=rays.ang_t[1], trans_ang=rays.ang_i[1],
                medium1=self.pipeline.material, medium2=self.acoustic_lens.coupling_medium,
                inc_mode="L", trans_mode="L"
            )

            # Water to aluminium:
            T21 = compute_tension_transmission_coefficient(
                inc_ang=rays.ang_t[0], trans_ang=rays.ang_i[0],
                medium1=self.acoustic_lens.coupling_medium, medium2=self.acoustic_lens.wedge_medium,
                inc_mode="L", trans_mode="L"
            )

            T2 = T32 * T21
        else:
            T1 = T2 = 1.

        return T1, T2

    def build_diffraction_tx_model(self, w, rays: List[Ray], enabled: bool):
        Nrays = int(len(rays))
        if enabled:
            thetas = np.zeros(Nrays)
            for i, ray in enumerate(rays):
                thetas[i] = np.pi/2 - np.arctan2(np.abs(ray.zray[0] - ray.zray[1]), np.abs(ray.xray[0] - ray.xray[1]))
            return far_field_directivity_solid(2 * np.pi * self.transducer.fc, thetas[:, np.newaxis], self.transducer.element_width, self.acoustic_lens.wedge_medium.cp, self.acoustic_lens.wedge_medium.cs).T
        else:
            return np.ones(shape=(w.shape[0], Nrays))

    def build_diffraction_rx_model(self, w, rays: List[Ray], enabled):
        return self.build_diffraction_tx_model(w, rays, enabled)

    def compute_tofs(self, optimal_rays: List[Ray]):
        n_elem = self.transducer.num_elem
        n_focii = optimal_rays[0].n_focus


        coord_elements = np.array([self.transducer.xt, self.transducer.zt]).T
        coords_reflectors = np.array([optimal_rays[0].xray[-1], optimal_rays[0].zray[-1]]).T
        coords_lens = np.zeros((n_elem, 2, n_focii))
        coords_outer = np.zeros((n_elem, 2, n_focii))

        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            coords_lens[j, 0, i], coords_lens[j, 1, i] = optimal_rays[j].xray[1][i], optimal_rays[j].zray[1][i]
            coords_outer[j, 0, i], coords_outer[j, 1, i] = optimal_rays[j].xray[2][i], optimal_rays[j].zray[2][i]

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

    def _dist_kernel(self, xc: float, zc: float, xf: ndarray, zf: ndarray, acurve: ndarray):

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
        b4 = zf - a4 * xf

        xin = (b4 - b3) / (a3 - a4)
        yin = a3 * xin + b3
        dist = (xin - xf)**2 + (yin - zf)**2


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