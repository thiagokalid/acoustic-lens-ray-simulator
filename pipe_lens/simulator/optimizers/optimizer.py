from typing import Callable, Tuple, Dict, Any, List
from abc import ABC, abstractmethod
import numpy as np

__all__ = ['RayOptimizer']

CostFunction = Callable[[float], Tuple[Dict[str, Any], float, float]]

class RayOptimizer(ABC):
    """
    Classe abstrata base para otimizadores que encontram o raio que respeita a lei de Snell no
    simulador por ray-tracing.

    Parameters
    ----------
    alpha_max : float
        Ângulo limite superior na busca por raios candidatos (em radianos).
    alpha_min : float
        Ângulo limite inferior na busca por raios candidatos (em radianos).
    verbose: bool
        Flag que define se irá ser mostrado informações como progresso, prints de debug, etc.
    """
    def __init__(self, alpha_max: float, alpha_min: float = 0., verbose: bool = False):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.verbose = verbose

    @abstractmethod
    def optimize(self,
                 cost_func: CostFunction) -> Dict[str, Any]:
        """Encontra o raio que respeita a lei de Snell via minimização da ``cost_func``. A função-custo irá conter
        toda a física de propagação de ondas.

        Parameters
        ----------
        cost_func : CostFunction
            Função-custo que contém a física de propagação de onda.
        initial_guess : float
            Chute inicial do método, em radianos.

        Returns
        -------
        Dict[str, Any]
            Resultados dos caminhos ótimos compilados em um dicionário.
        """
        pass

    @abstractmethod
    def optimize_batch(self,
                       transducer_coords: Tuple[np.ndarray, np.ndarray],
                       target_coords: Tuple[np.ndarray, np.ndarray],
                       cost_func_factory: Callable) -> List[Any]:
        """Encontra o raio que respeita a lei de Snell para vários pares elementos e pontos de foco.

        Parameters
        ----------
        transducer_coords : Tuple[np.ndarray, np.ndarray]
            Coordendas do centro dos elementos do transdutor, em metros.
        target_coords : Tuple[np.ndarray, np.ndarray]
            Coordendas dos pontos de chegada do raio, em metros.
        cost_func_factory : Callable
            Função-custo que será minimizada para encontrar o caminho ótimo que respeita a lei de Snell.

        Returns
        -------
        List[Dict]
            Lista com resultados dos caminhos ótimos compilados em um dicionário.
        """
        pass
