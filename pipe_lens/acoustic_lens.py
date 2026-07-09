import numpy as np
from numpy import cos, sin, sqrt, pi, arcsin, linspace
from pipe_lens.types import Material, Fluid, Solid
from pipe_lens.utils import convert_parameters
from scipy.optimize import brentq
__all__ = ['AcousticLens']


class AcousticLens:
    """
    Modelo analítico para o cálculo do formato de uma lente acústica para inspeções em tubos.

    A lente é definida de modo que o tempo total de percurso (material da sapata + coluna do meio acoplante)
     seja constante para todos os ângulos de inspeção ``alpha``.

    Parameters
    ----------
    wedge_material : Material
        Material da sapata (típicamente um sólido). A classe utiliza apenas a velocidade longitudinal ``cp`` e a
        densidade.
    coupling_medium : Material
        Meio externo à lente (água, por exemplo). Deve possuir velocidade longitudinal ``cp`` e a
        densidade.
    d : float
        Distância vertical entre o transdutor e o centro do tubo. Considera-se o centro do tubo a origem do sistema
        de coordenadas.
    alpha_max : float
        Ângulo máximo permitido da varredura sectorial, em radianos.
        A lente é válida em ``alpha ∈ [-alpha_max, +alpha_max]``.
    alpha_0 : float
        Ângulo de referência usado para impor a condição de contorno.
        Deve satisfazer ``|alpha_0| ≤ alpha_max``.
    h0 : float
        Distância entre lente e centro do sistema de coordenadas (0,0) no ângulo de referência ``alpha_0``. É uma
        condição de contorno do problema.
    pipe_outer_radius : float
        Raio externo nominal do tubo sob inspeção. Apesar da lente acústica funcionar para diâmetros diferentes do
        nominal, ela tem uma operação ótima no valor de projeto.

    Raises
    ------
    ValueError
        Se ``alpha_0`` estiver fora do intervalo permitido.
    ValueError
        Se ``h0 ≥ d``, o que torna a geometria fisicamente inválida.

    """

    def __init__(self, wedge_medium: Material, coupling_medium: Material,
                 d: float, alpha_max: float, alpha_0: float, h0: float, pipe_outer_radius: float = None, alpha_min: float = 0.0, tolerance: float = 1e-4):

        self.d = d
        self.c1 = wedge_medium.cp
        self.c2 = coupling_medium.cp
        self.wedge_medium = wedge_medium
        self.coupling_medium = coupling_medium
        self.alpha_min = alpha_min
        self.tol = tolerance

        if np.abs(alpha_0) > alpha_max:
            raise ValueError('alpha_0 fora do intervalo permitido [-alpha_max, +alpha_max].')

        self.alpha_0 = alpha_0

        if h0 >= d:
            raise ValueError("h0 deve ser estritamente menor que d.")

        self.h0 = h0
        self.z0 = self.d - self.h0
        self.l0 = self.d - self.h0
        self.rho1 = wedge_medium.density
        self.rho2 = coupling_medium.density
        self.pipe_outer_radius = pipe_outer_radius
        if self.pipe_outer_radius is not None:
            self.t_outer = 2 * (self.z0 / self.c1 + (self.h0 - self.pipe_outer_radius) / self.c2)
        else:
            self.t_outer = None

        # Pontos geométricos usados na equação de Fermat
        x0, z0 = h0 * cos(pi/2 - alpha_0), h0 * sin(pi/2 - alpha_0)
        xt, zt = 0, d

        self.T = sqrt((x0 - xt)**2 + (z0 - zt)**2) / self.c1 + self.h0 / self.c2

        self.alpha_max = alpha_max
        self.a = (self.c1 / self.c2)**2 - 1
        self.b = lambda alpha: 2 * d * cos(alpha) - 2 * self.T * self.c1**2 / self.c2
        self.c = (self.c1 * self.T)**2 - d**2


        # Arbitrary discretization:
        self.xlens, self.zlens = self.xy_from_alpha(
            linspace(-self.alpha_max, self.alpha_max, 1000)
        )

    def h(self, alpha, safe=True):
        """
        Calcula a distância entre a lente e o centro do sistema de coordenadas para um ângulo de inspeção ``alpha``.

        Parameters
        ----------
        alpha : float ou array_like
            Ângulo(s) de inspeção da tubulação, em radianos.

        Returns
        -------
        float ou ndarray
            Distância entre a lente e o centro do sistema de coordenadas correspondente ao(s) ângulo(s) fornecido(s).

        Raises
        ------
        ValueError
            Se algum ``alpha`` estiver fora de ``[-alpha_max, +alpha_max]``.
        """
        if np.any(np.logical_or(np.abs(alpha) > (self.alpha_max + self.tol), np.abs(alpha) < (self.alpha_min - self.tol))) and safe:
            raise ValueError('alpha fora do intervalo permitido [-alpha_max, +alpha_max].')

        return (-self.b(alpha) - sqrt(self.b(alpha)**2 - 4 * self.a * self.c)) / (2 * self.a)

    def dhda(self, alpha, safe=True):
        """
        Calcula a derivada ``dh/dalpha`` da superfície da lente.

        Parameters
        ----------
        alpha : float ou ndarray
            Ângulo(s) de inspeção, em radianos.

        Returns
        -------
        float ou ndarray
            Valor de ``dh/dalpha`` no(s) ponto(s) especificado(s).

        Raises
        ------
        ValueError
            Se algum valor estiver fora da faixa angular permitida.

        Notes
        -----
        Implementa as expressões apresentadas no Apêndice A.2.2, Equações (A.19a)–(A.19b) do artigo
        ´´An acoustic lens for displacement-free sectorial inspection of pipes with ultrasonic phased arrays´´
        """
        if safe:
            if np.any(np.logical_or(np.abs(alpha) > (self.alpha_max + self.tol), np.abs(alpha) < (self.alpha_min - self.tol))):
                raise ValueError('alpha fora do intervalo permitido [-alpha_max, +alpha_max].')

        return -1 / (2 * self.a) * (
            -2 * self.d * sin(alpha)
            + 0.5 * (1 / sqrt(self.b(alpha)**2 - 4 * self.a * self.c))
            * (-4 * self.b(alpha) * self.d * sin(alpha))
        )

    def xy_from_alpha(self, alpha, safe=True):
        """
        Converte um ângulo de inspeção ``alpha`` para coordenadas cartesianas
        da superfície da lente.

        Parameters
        -----------
        alpha : float ou ndarray
            Ângulo(s) de inspeção, em radianos.

        Returns
        -------
        x : float ou ndarray
            Coordenada horizontal do ponto pertencente à superfície da lente.
        y : float ou ndarray
            Coordenada vertical do ponto pertencente à superfície da lente.

        Raises
        ------
        ValueError
            Se ``alpha`` estiver fora dos limites definidos.
        """
        z = self.h(alpha, safe)
        x = z * sin(alpha)
        y = z * cos(alpha)
        return x, y

    def dydx_from_alpha(self, alpha, mode='full', safe=True):
        """
        Calcula a inclinação ``dy/dx`` da lente para um ângulo ``alpha``.

        Parameters
        ------------
        alpha : float ou ndarray
            Ângulo(s) de inspeção, em radianos.
        mode : {'full', 'partial'}, opcional
            * ``'full'`` : retorna apenas ``dy/dx``.
            * ``'partial'`` : retorna ``(dy, dx)`` separadamente.

        Returns
        --------
        float ou ndarray
            Se ``mode='full'``: a razão ``dy/dx``.
        (dy, dx) : tuple de ndarrays
            Se ``mode='partial'``: derivadas separadas.

        Raises
        ------
        ValueError
            Se ``alpha`` estiver fora da faixa permitida.
        ValueError
            Se ``mode`` for inválido.
        """
        h_ = self.h(alpha, safe)

        dy = self.dhda(alpha, safe) * np.cos(alpha) - h_ * np.sin(alpha)
        dx = self.dhda(alpha, safe) * np.sin(alpha) + h_ * np.cos(alpha)

        if mode == 'full':
            return dy / dx
        elif mode == 'partial':
            return dy, dx
        else:
            raise ValueError("mode deve ser 'full' ou 'partial'.")

    def y_from_x(self, x, safe=True):
        alphas = self.alpha_from_x(x)
        _, y = self.xy_from_alpha(alphas, safe)
        return y

    def alpha_from_x(self, x_):
        if isinstance(x_, float) or len(x_) == 1:
            x = [x_]
        else:
            x = x_

        alphas = np.zeros_like(x)

        for i, xx in enumerate(x):
            target = np.abs(xx)

            def f(alpha):
                return self.xy_from_alpha(alpha)[0] - target

            alphas[i] = brentq(f, self.alpha_min, self.alpha_max)

        return alphas

    def dydx(self, x, mode='full'):
        alphas = self.alpha_from_x(x)
        return self.dydx_from_alpha(alphas, mode=mode)


    def pipe2steering_angle(self, alpha):
        """
        Converte o ângulo de inspeção (com relação ao centro do tubo) ``alpha`` no ângulo de disparo ``beta``
        (com relação ao centro do transdutor).

        Parameters
        ----------
        alpha : float ou ndarray
            Ângulo(s) de inspeção, em radianos.

        Returns
        -------
        beta : float ou ndarray
            Ângulo de disparo do raio partindo do centro do transdutor..

        """
        x = self.h(alpha) * sin(alpha)
        y = self.h(alpha) * cos(alpha)
        r = sqrt(x**2 + (self.d - y)**2)
        beta = arcsin(self.h(alpha) * sin(alpha) / r)
        return beta

    @classmethod
    def from_gustavo_parameters(cls, c, wc, r_outer, wedge_medium: Material, coupling_medium: Material):
        d = 2 * c
        h0, _, alpha_max = convert_parameters(c, wc, r_outer, wedge_medium.cp, coupling_medium.cp)
        return cls(
            wedge_medium=wedge_medium,
            coupling_medium=coupling_medium,
            d=d,
            alpha_max=alpha_max,
            alpha_0=0,
            h0=h0,
            pipe_outer_radius=r_outer
        )

    def to_dict(self):
        design_parameters = {
            "h0": self.h0 * 1e3,
            "d": self.d * 1e3,
            "wedge_medium": self.wedge_medium.to_dict(),
            "coupling_medium": self.coupling_medium.to_dict(),
            "alpha_max": np.degrees(self.alpha_max),
            "alpha_0": np.degrees(self.alpha_0),
        }

        if self.pipe_outer_radius is not None:
            design_parameters["pipe_outer_radius"] = self.pipe_outer_radius * 1e3

        return design_parameters

    @classmethod
    def from_dict(cls, data : dict):
        wedge_medium = data.get("wedge_medium")
        wedge_medium = Solid.from_dict(wedge_medium) if wedge_medium.get("type") == "solid" else Fluid.from_dict(wedge_medium)

        coupling_medium = data.get("coupling_medium")
        coupling_medium = Solid.from_dict(coupling_medium) if coupling_medium.get("type") == "solid" else Fluid.from_dict(coupling_medium)

        return cls(
            wedge_medium=wedge_medium,
            coupling_medium=coupling_medium,
            d=data["d"] * 1e-3,
            alpha_max=np.radians(data["alpha_max"]),
            alpha_0=np.radians(data["alpha_0"]),
            h0=data["h0"] * 1e-3,
            pipe_outer_radius=data["pipe_outer_radius"] * 1e-3 if data["pipe_outer_radius"] is not None else None,

        )

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"wedge_material={self.wedge_medium!r}, "
            f"coupling_medium={self.coupling_medium!r}, "
            f"d={self.d}, "
            f"alpha_max={self.alpha_max}, "
            f"alpha_0={self.alpha_0}, "
            f"h0={self.h0}"
            f")"
        )

    def __str__(self):
        return (
            "AcousticLens\n"
            f"  wedge material   : {self.wedge_medium}\n"
            f"  coupling medium  : {self.coupling_medium}\n"
            f"  d                : {self.d:.3f} m\n"
            f"  alpha_max        : {self.alpha_max:.3f} rad\n"
            f"  alpha_0          : {self.alpha_0:.3f} rad\n"
            f"  h0               : {self.h0:.3f} m"
        )
