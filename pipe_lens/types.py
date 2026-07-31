# built-in modules
from abc import ABC
from typing import Tuple, List
from copy import deepcopy

# scientific-computing modules
import numpy as np
from numpy import sqrt, square, float64, complex128
from numpy.typing import NDArray, ArrayLike

__all__ = ["FLOAT", "Material", "Fluid", "Solid"]

# default precision for entire module
FLOAT = float64
COMPLEX = complex128

class Material(ABC):
    def __init__(self, cp: float, density: float, attenuation_coeff: Tuple[Tuple[float | int, float | int], ...] = ((0., 0.), ), cs: float = 0, name: str=""):
        if cp <= 0:
            raise ValueError("cp (pressure wave) speed must be positive.")
        if cs < 0:
            raise ValueError("cs (shear wave) speed must non-negative.")
        if density <= 0:
            raise ValueError("Density must be positive.")

        self._cp = cp
        self._cs = cs
        self._density = density
        self._acoustic_impedance = self._cp * self._density
        self._name = name
        self._attenuation_coeff = attenuation_coeff
        self._type = ""

    @property
    def cp(self) -> float:
        return self._cp

    @property
    def cs(self) -> float:
        return self._cs

    @property
    def density(self) -> float:
        return self._density


    def alpha(self, frequency):
        # based on page 113, Krautkrämer, J., & Krautkrämer, H. (2013).
        # Ultrasonic testing of materials. Springer Science & Business Media.
        alpha = np.ones_like(frequency)
        for coeff_pair in self._attenuation_coeff:
            alpha += coeff_pair[0] * np.power(np.abs(frequency), coeff_pair[1])
        return alpha


    @property
    def acoustic_impedance(self) -> float:
        return self._acoustic_impedance

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["cp"], data["density"], data["name"])

    def to_dict(self) -> dict:
        return {
            "name": self._name,
            "type": self._type,
            "cp": self._cp,
            "density": self._density,
            "cs": self._cs,
            "alpha": self._alpha,
        }

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(cp={self._cp:.2f} m/s, "
                f"density={self._density:.2f} kg/m^3, "
                f"acoustic_impedance={self._acoustic_impedance * 1e-6:.2f} MRayl,"
                f"attenuatino coefficient= {self._alpha:.2f} s^2 Np/m)")


    def __eq__(self, other) -> bool:
        return self.cp == other.cp and self.density == other.density and self.name == other.name and self.cs == other.cs and self._alpha == other._alpha

class Fluid(Material):
    """
    Classe que contém informação sobre fluidos.

    Parameters
    ----------
    cp : float
        Velocidade da onda de pressão (longitudinal) no sólido, em m/s.
    density : float
        Densidade do material, em kg/m³.
    """
    def __init__(self, cp: float, density: float, attenuation_coeff: Tuple[Tuple[float | int, float | int], ...] = ((0., 0.),), name : str = ""):
        super().__init__(cp, density, attenuation_coeff=attenuation_coeff, cs=0., name=name)
        self._type = "fluid"

    @property
    def cs(self) -> float:
        raise ValueError("Fluid does not have shear-wave velocity.")


class Solid(Material):
    """
    Classe que contém informação sobre materiais sólidos.

    Parameters
    ----------
    cp : float
        Velocidade da onda de pressão (longitudinal) no sólido, em m/s.
    cs : float
        Velocidade da onda de cisalhamento (transversal) no sólido, em m/s. Por padrão é metade de ``cp``.
    density : float
        Densidade do material, em kg/m³.
    """
    def __init__(self, cp: float, cs: float, density: float, attenuation_coeff: Tuple[Tuple[float | int, float | int], ...] = ((1., 0.),), name : str = ""):
        if cs <= 0:
            raise ValueError("cs (shear wave) speed cannot be negative.")
        super().__init__(cp, density, cs=cs, attenuation_coeff=attenuation_coeff, name=name)
        self._type = "solid"

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["cp"], data["cs"], data["density"], data["name"])

class Ray:
    """Classe que armazena informações de um caminho de raios.

    Parameters
    ----------
    xray: list of ndarray
        Lista de coordenadas `x` dos pontos onde o raio interage com superfícies.
    zray: list of ndarray
        Lista de coordenadas `z` dos pontos onde o raio interage com superfícies.
    ang_i: list of ndarray
        Lista com os ângulos incidentes dos raios nas superfícies.
    ang_t: list of ndarray
        Lista com os ângulos transmitidos ou refletidos dos raios nas superfícies.
    dist2focus: ndarray
        Array 2D que representa a distância entre o último trecho do raio e o foco.
    medoium: tuple of Material
        Tupla com objeto tipo `Material` dos meios onda há propagação de cada trecho do raio.
    modes: tuple of str
        Tupla com o modo de propagação em cada trecho do raio ("L" ou "T").
    tof: ndarray
        Array 2D que representa o tempo de vôo do raio até o ponto final. Caso não seja fornecido, será calculado com
        base nos modos e meios fornecidos.
    """
    def __init__(self, xray, zray, ang_i, ang_t, dist2focus, medium: Tuple[Material, ...], modes : Tuple[str, ...] | None = None, tof = None):
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

    def __index__(self):
        pass

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

