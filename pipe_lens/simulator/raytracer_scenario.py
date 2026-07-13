import numpy as np

from pipe_lens.simulator.ray_physics.ray_physics import RayPhysics
from pipe_lens.simulator.optimizers.optimizer import RayOptimizer

__all__ = ["RaytracerScenario"]

class RaytracerScenario:
    """
    Classe responsável por gerenciar uma instância do simulador por ray-tracing.

    Parameters
    ----------
    physics: RaytracerPhysics
        Objeto que irá conter toda física de propagação de ondas do problema.
    optimizer : RaytracerOptimizer
        Método que irá encontrar o caminho que respeita a lei de Snell baseado na física do problema.
        
    """
    def __init__(self,
                 physics: RayPhysics,
                 optimizer: RayOptimizer,
                 ):
        self.physics = physics
        self.optimizer = optimizer

    def solve(self, xf: np.ndarray, zf: np.ndarray):
        """
        Método que encontra, para um par de coordenadas (xf, zf) qual o caminho que respeita a lei de Snell que tem como fim
        esses pontos, e início o centro dos elementos do transdutor.

        Parameters
        ----------
        xf : ndarray
            Coordenadas x dos pontos destino que deseja-se encontrar o caminho ótimo, em metros.
        zf : ndarray
            Coordenadas z dos pontos destino que deseja-se encontrar o caminho ótimo, em metros.

        Returns
        -------
        results : list[Dict]
            Lista de dicionários contendo os pontos em cada superfície onde há refração e outras informações adicionais.
        """
        if isinstance(xf, float) or isinstance(zf, int):
            xf = np.array([xf])
            zf = np.array([zf])

        xc, yc = self.physics.transducer.coords

        # Example of simple delegation:
        results = self.optimizer.optimize_batch(
            transducer_coords=(xc, yc),
            target_coords=(xf, zf),
            cost_func_factory=self.physics.get_cost_function_for_ray
        )

        return results