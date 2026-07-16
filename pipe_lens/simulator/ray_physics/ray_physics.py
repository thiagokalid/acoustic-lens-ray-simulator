# scientific-computing modules
import numpy as np
from numpy import ndarray

# built-in modules
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

# modules implemented in this repo
from pipe_lens.pipeline import Pipeline
from pipe_lens.transducer import Transducer
from pipe_lens.types import COMPLEX, Ray

__all__ = ["RayPhysics"]

@dataclass
class LossModel:
    propagation: bool = True
    transmission: bool = False
    transmission_diffraction: bool = False
    attenuation: bool = False
    reception_diffraction: bool = False

class RayPhysics(ABC):
    """Classe abstrata que irá conter toda física de propagação de ondas, assim como a função-custo que será
    usada para encontrar os raios ótimos.

    Parameters
    ----------
    pipeline : Pipeline
        Tubulação do problema.
    transducer : Transducer
        Transdutor considerado no problema.
    """
    def __init__(self,
                 pipeline: Pipeline,
                 transducer: Transducer,
                 loss_model: LossModel = LossModel()
                 ):

        self.transducer = transducer
        self.pipeline = pipeline
        self.loss_model = loss_model

    def get_cost_function_for_ray(self, xc: float, yc: float, xf: np.ndarray, yf: np.ndarray, eps: float = 1e-4):
        """
        Returns a callable that the Optimizer can use.
        This closes over the specific geometry parameters.
        """

        def cost(alpha):
            # This logic remains in your Physics class (Raytracer)
            return self.__dist_and_derivatives(xc, yc, xf, yf, alpha, eps)

        return cost


    def __dist_and_derivatives(self, xc: float, yc: float, xf: ndarray, yf: ndarray, acurve: ndarray, eps: float):
        '''Computes the squared distance using distalpha as well as the first and
      second derivatives of the squared distance with relation to alpha.'''
        dm = self._dist_kernel(xc, yc, xf, yf, acurve - eps).dist2focus
        dic = self._dist_kernel(xc, yc, xf, yf, acurve)
        d0 = dic.dist2focus
        dp = self._dist_kernel(xc, yc, xf, yf, acurve + eps).dist2focus
        der1 = (dp - dm) * .5 / eps
        der2 = (dm - 2 * d0 + dp) / eps ** 2
        return dic, der1, der2

    ##### Case-specific:
    @abstractmethod
    def _dist_kernel(self, xc: float, zc: float, xf: ndarray, yf: ndarray, acurve: ndarray) -> Ray:
        pass

    @abstractmethod
    def build_measurement_model(self, w, solutions, gate_start: float = 0.):
        pass

    @abstractmethod
    def compute_tofs(self, optimal_rays: List[Ray]):
        pass

    def build_propagation_model(self, w, rays, flag: bool):
        tof_mtx = self.compute_tofs(rays) # TOF from TX to RX, thus matrix with size TX by RX
        if flag:
            return np.exp(-1j * w * tof_mtx.T).T
        else:
            return np.ones_like(w * tof_mtx, dtype=COMPLEX) # Null delay

    def build_transmission_model(self, w, solution, flag: bool):
        if flag:
            raise NotImplementedError
        else:
            T1 = T2 = np.ones_like(w)
            return T1, T2

    def build_diffraction_tx_model(self, w, solution, flag):
        if flag:
            raise NotImplementedError
        else:
            Nsamples = len(w)
            return np.ones(shape=(Nsamples, self.transducer.num_elem))

    def build_attenuation_model(self, w, solution, flag):
        if flag:
            raise NotImplementedError
        else:
            return np.ones(shape=(w.shape[0], self.transducer.num_elem, self.transducer.num_elem))


    def build_system_function_model(self, w, solution, flag):
        if flag:
            raise NotImplementedError
        else:
            return np.ones_like(w)

    def build_flaw_scattering_model(self, w, solution, flag):
        return np.ones_like(w)

    def build_diffraction_rx_model(self, w, solution, flag):
        if flag:
            raise NotImplementedError
        else:
            Nsamples = len(w)
            return np.ones(shape=(Nsamples, self.transducer.num_elem))