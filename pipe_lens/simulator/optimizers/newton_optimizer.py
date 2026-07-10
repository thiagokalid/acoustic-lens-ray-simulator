# built-in modules
from typing import Callable, Tuple
import numpy as np
import warnings

# pypi helper modules
from tqdm import tqdm

# modules implemented in this repo
from pipe_lens.types import Ray, pick_best_rays
from pipe_lens.simulator.optimizers.optimizer import RayOptimizer

CostFunction = Callable[[float], Tuple[Ray, float, float]]

__all__ = ['NewtonOptimizer']

class NewtonOptimizer(RayOptimizer):
    """Otimizador baseado em Newton-raphson que encontra o caminho ótimo via método iterativo.

    Parameters
    ----------
    alpha_max: float
        Ângulo limite superior na busca por raios candidatos.
    max_iter: int
        Número máximo de itarações do método numérico.
    eps: float
        Tolerância de parada do método numérico.
    alpha_min: float
        Ângulo limite inferior na busca por raios candidatos.
    learning_rate: float
        Fator que multiplica o passo do método de Newton
    dist_tolerance: float
        Limiar de distância entre raio refratado e foco que define se método convergiu ou não.
    init_guess: float
        Vetor com chutes inicias para o ângulo do Newton. Caso não seja fornecido, o chute inicial será a posição angular
        do próprio foco.
    verbose: bool
        Flag que define se irá ser mostrado informações como progresso, prints de debug, etc.
    """
    def __init__(self, alpha_max: float, max_iter: int = 10, eps: float = 1e-4, alpha_min: float = 0., learning_rate: float = 1, dist_tolerance: float = 1e-6, init_guess: np.ndarray | None = None, verbose: bool = False):
        super().__init__(alpha_max, alpha_min, verbose)
        if init_guess is None:
            self.init_guess = init_guess
        else:
            if (np.any(init_guess < alpha_min) or np.any(init_guess > alpha_max)):
                raise ValueError('Invalid initial guess.')
            else:
                self.init_guess = init_guess
        self.max_iter = max_iter
        self.eps = eps
        if learning_rate < 0:
            raise ValueError('Learning rate must be positive.')
        self.learning_rate = learning_rate

        if dist_tolerance < 0:
            raise ValueError('Distance tolerance must be positive.')
        self.dist_tolerance = dist_tolerance

    def optimize(self, cost_func: CostFunction, init_guess: float | np.ndarray) -> Ray:
        alpha = init_guess

        for i in range(self.max_iter):
            # The cost_func now returns the dict AND the derivatives
            alpha = np.sign(alpha) * np.clip(np.abs(alpha), self.alpha_min, self.alpha_max)
            ray_candidate, d1, d2 = cost_func(alpha)

            # Newton Step
            alpha -= self.learning_rate * d1 / d2
            alpha[np.isnan(alpha)] = init_guess[np.isnan(alpha)]

            # Check which rays have converged according to pre-defined tolerance:
            converged_rays = ray_candidate.dist2focus < self.dist_tolerance

            if i == 0:
                best_ray = ray_candidate
            else:
                # regular iteration
                best_ray, _ = pick_best_rays(best_ray, ray_candidate, converged_rays)

            if np.all(converged_rays):
                # all rays have converged
                break

        return best_ray

    def optimize_batch(self, transducer_coords, target_coords, cost_func_factory):
        N_elem = len(transducer_coords[0])
        xc, yc = transducer_coords
        xf, yf = target_coords

        results = [None] * N_elem

        for i in tqdm(range(0, N_elem // 2), disable=not self.verbose):
            pairs = [
                int(N_elem // 2 - i - 1), # marching towards the left
                int(N_elem // 2 + i) # marching towards the right
            ]
            sign = [1, -1]
            for c, p in zip(sign, pairs):
                if i == 0 or self.init_guess is not None:
                    # First iteration initial-guess given by user:
                    init_guess = np.arctan2(xf, yf)
                else:
                    # Compute next inital guess
                    ray_p = results[p + c]
                    x_mirror, z_mirror = ray_p.xray[1], ray_p.zray[1]
                    init_guess = np.arctan2(x_mirror, z_mirror)

                # Compute the optimal path for a given transducer (xc,yc) focus (xf, yf) pair:
                results[p] = self.optimize(
                    cost_func=cost_func_factory(xc[p], yc[p], xf, yf, self.eps),
                    init_guess=init_guess
                )

                bad_indices = results[p].dist2focus > self.dist_tolerance

                if np.count_nonzero(bad_indices) > 0:
                    warnings.warn("Some rays have not converged.", RuntimeWarning)
        if N_elem % 2 != 0:
            results[-1] = self.optimize(
                    cost_func=cost_func_factory(xc[-1], yc[-1], xf, yf, self.eps),
                    init_guess=init_guess
            )

        return results