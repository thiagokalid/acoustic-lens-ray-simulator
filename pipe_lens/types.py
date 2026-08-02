# built-in modules
from abc import ABC
from typing import Tuple, List, ClassVar, Literal
from copy import deepcopy
from dataclasses import dataclass

# scientific-computing modules
import numpy as np
from numpy import sqrt, square, float64, complex128
from numpy.typing import NDArray, ArrayLike

__all__ = ["FLOAT", "Material", "Fluid", "Solid"]

# default precision for entire module
FLOAT = float64
COMPLEX = complex128

AttenuationCoeffs = tuple[tuple[float, float], ...]
Mode = Literal["L", "T"]
Modes = tuple[Mode, ...]

@dataclass(frozen=True)
class Material(ABC):
    cp: float
    density: float
    attenuation_coeff: AttenuationCoeffs = ((0.0, 0.0),)
    cs: float = 0.0
    name: str = ""

    def __post_init__(self):
        if self.cp <= 0:
            raise ValueError("cp must be positive.")
        if self.cs < 0:
            raise ValueError("cs must be non-negative.")
        if self.density <= 0:
            raise ValueError("density must be positive.")

    @property
    def acoustic_impedance(self) -> float:
        return self.cp * self.density

    def alpha(self, frequency: np.ndarray) -> np.ndarray:
        alpha = np.zeros_like(frequency, dtype=float)
        for a, n in self.attenuation_coeff:
            alpha += a * np.abs(frequency) ** n
        return alpha

@dataclass(frozen=True)
class Fluid(Material):
    type: ClassVar[str] = "fluid"

    def __post_init__(self):
        super().__post_init__()
        if self.cs != 0:
            raise ValueError("A fluid must have cs = 0.")


@dataclass(frozen=True)
class Solid(Material):
    type: ClassVar[str] = "solid"

    def __post_init__(self):
        super().__post_init__()
        if self.cs <= 0:
            raise ValueError("A solid must have cs > 0.")

class Ray:
    def __init__(self, xray, zray, ang_i, ang_t, dist2focus, medium: Tuple[Material, ...], modes : Modes | None = None, tof = None):
        self.xray = np.asarray(xray)
        self.zray = np.asarray(zray)
        self.ang_i = np.asarray(ang_i)
        self.ang_t = np.asarray(ang_t)
        self.dist2focus = np.asarray(dist2focus)
        self.medium = medium
        self.n_medium = len(medium)
        self.n_focus = zray[-1].shape[-1]
        self._tof = tof
        if modes is None:
            self.modes = ['L'] * len(medium)
        else:
            if set(modes) <= {"L", "T"}:
                self.modes = modes
            else:
                raise ValueError("Modes not allowed.")

    @property
    def tof(self):
        if self._tof is None:
            self._tof = self.compute_tof()
        return self._tof

    @property
    def segment_dist(self):
        """ Calcula a distância total percorrida do raio em cada segmento.

        Returns
        -------
        total_dist: list
            Lista de Array 2D com distância do(s) raio(s).
        """
        total_dist = list()

        for i in range(self.n_medium):
            x1 = self.xray[i]
            z1 = self.zray[i]

            x2 = self.xray[i + 1]
            z2 = self.zray[i + 1]

            total_dist.append(sqrt(square(x1 - x2) + square(z1 - z2)))

        return tuple(total_dist)

    def compute_dist(self):
        """ Calcula a distância total percorrida do raio.

        Returns
        -------
        total_dist: ndarray
            Array 2D com distância do(s) raio(s).
        """
        total_dist = 0.0

        for i in range(self.n_medium):
            x1 = self.xray[i]
            z1 = self.zray[i]

            x2 = self.xray[i + 1]
            z2 = self.zray[i + 1]

            total_dist += sqrt(square(x1 - x2) + square(z1 - z2))

        return total_dist

    def compute_tof(self):
        """ Calcula o tempo de vôo do ponto inicial ao final do raio.

        Returns
        -------

        """
        total_time = 0.0

        for i in range(self.n_medium):
            x1 = self.xray[i]
            z1 = self.zray[i]

            x2 = self.xray[i + 1]
            z2 = self.zray[i + 1]

            d_i = sqrt(square(x1 - x2) + square(z1 - z2))

            speed = self.medium[i].cs if self.modes[i] == 'T' else self.medium[i].cp
            total_time += d_i / speed

        return total_time

def pick_best_rays(ray1: Ray, ray2: Ray, mask: np.ndarray[bool]) -> tuple[Ray, NDArray[np.bool_]]:
    """ Compara dois objetos do tipo `Ray` com base em seu tempo de vôo, e retorna o que tiver menor tempo de vôo.

    Parameters
    ----------
    ray1: `Ray`
        Raio 1 a ser comparado
    ray2: `Ray`
        Raio 2 a ser comparado
    mask: ndarray[bool]
        An array that restricts the comparisson only to a few rays.

    Returns
    -------
    new_ray: `Ray`
        Resultado de menor TOF entre `ray1` e `ray2`
    best_rays: ndarray
        Máscara que definiu qual foi o melhor raio.
    """
    new_ray = deepcopy(ray1)
    best_rays = ray1.dist2focus > ray2.dist2focus
    best_rays = best_rays * mask # merge both masks
    new_ray.xray[:, best_rays] = ray2.xray[:, best_rays]
    new_ray.zray[:, best_rays] = ray2.zray[:, best_rays]
    new_ray.ang_i[:, best_rays] = ray2.ang_i[:, best_rays]
    new_ray.ang_t[:, best_rays] = ray2.ang_t[:, best_rays]
    new_ray.dist2focus[best_rays] = ray2.dist2focus[best_rays]
    return new_ray, best_rays

