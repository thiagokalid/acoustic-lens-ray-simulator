import numpy as np
from numpy import sqrt
from pipe_lens.types import Solid
from pipe_lens.geometric_utils import circle_cartesian

__all__ = ["Pipeline"]

class Pipeline:
    """
    Objeto que contém informações sobre corpo de prova cilíndrico.

    Parameters
    ----------
    material: Solid
        Propriedades do material que é feito o tubo. Considera-se o tubo feito apenas de um único material.
    outer_radius : float
        Raio externo da tubulação, em metros.
    wall_thickness : float
        Espessura da parede do tubo, em metros.
    center : tuple[float, float]
        Coordenada onde o centro do tubo está (em metros).

    """
    def __init__(self, material: Solid, outer_radius: float, wall_thickness, center: tuple[float, float] = (0., 0.)):
        self.outer_radius = outer_radius
        self.inner_radius = outer_radius - wall_thickness
        self.wall_width = wall_thickness
        self.c = material.cp
        self.rho = material.density
        self.material = material
        self.xcenter = center[0]
        self.zcenter = center[1]

        self.xint, self.zint = circle_cartesian(self.inner_radius, self.xcenter, self.zcenter)
        self.xout, self.zout = circle_cartesian(self.outer_radius, self.xcenter, self.zcenter)

    def xz_from_alpha(self, alpha: float) -> [float, float]:
        """Retorna coordenadas (x,z) de um ponto pertencente à superfície externa do tubo na posição alfa. Alfa é medido
        em relação ao centro do tubo, onde alfa = 0 é às 0h ou 12h.

        Parameters
        ----------
        alpha : float
            Posição angular a ser retornada a coordenada (x,z).

        Returns
        -------
        x, y: [float, float]
            Coordenadas (x,z) correspondente ao alfa.
        """
        raise NotImplementedError

    def dydx_from_alpha(self, alpha: float) -> float:
        """Retorna a derivada no ponto pertencente à superfície externa do tubo na posição alfa. Alfa é medido
        em relação ao centro do tubo, onde alfa = 0 é às 0h ou 12h.

        Parameters
        ----------
        alpha : float
            Posição angular a ser retornada a coordenada (x,z).

        Returns
        -------
        dzdx : float
            Derivada de z em relação a x no ponto alfa.
        """
        raise NotImplementedError

    def y_from_x(self, x):
        inside = self.outer_radius ** 2 - (x - self.xcenter) ** 2
        return self.zcenter + np.sqrt(inside)

    def dydx(self, x, mode='full'):
        """Retorna a derivada no ponto pertencente à superfície externa do tubo na posição x.

        Parameters
        ----------
        x : float
            Posição x a ser avaliada a derivada.

        Returns
        -------
        dzdx : float
            Derivada de z em relação a x avaliada em x.
        """
        dy = -(x - self.xcenter)
        dx = sqrt(self.outer_radius ** 2 - (x - self.xcenter) ** 2)
        if mode == "full":
            return dy/dx
        elif mode == 'partial':
            return dy, dx

    def find_ray_intersection(self, x0_ray, z0_ray, ray_slope, surface="outer", epsilon=1e-12):
        if surface == "outer":
            radius = self.outer_radius
        elif surface == "inner":
            radius = self.inner_radius
        else:
            raise ValueError("Invalid surface")

        # Analytical formula for intesection between a line and a circle
        dx = np.cos(ray_slope)
        dz = np.sin(ray_slope)

        a = dx ** 2 + dz ** 2
        b = 2 * ((x0_ray - self.xcenter) * dx + (z0_ray - self.zcenter) * dz)
        c = (x0_ray - self.xcenter) ** 2 + (z0_ray - self.zcenter) ** 2 - radius ** 2

        delta = b ** 2 - 4 * a * c
        sol1 = (-b + np.sqrt(delta)) / (2 * a)
        sol2 = (-b - np.sqrt(delta)) / (2 * a)

        sol1[sol1 < epsilon] = np.nan
        sol2[sol2 < epsilon] = np.nan

        sol_hit = np.nanmin(np.asarray([sol1, sol2]), axis=0)

        p_hit = np.asarray([
            x0_ray + sol_hit * dx,
            z0_ray + sol_hit * dz
        ], dtype=np.float32)

        return p_hit