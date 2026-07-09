# built-in modules
from abc import ABC
from copy import deepcopy

# scientific-computing modules
import numpy as np

# modules implemented in this repo
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.pipeline import Pipeline
from pipe_lens.simulator.ray_physics.ray_physics import RayPhysics, LossModel
from pipe_lens.transducer import Transducer

__all__ = ["RayLensPhysics"]

class RayLensPhysics(RayPhysics, ABC):
    """Classe abstrata que irá conter toda física de propagação de ondas, assim como a função-custo que será
    usada para encontrar os raios ótimos.

    Parameters
    ----------
    acoustic_lens : AcousticLens
        Lente acústica usada no problema.
    pipeline : Pipeline
        Tubulação do problema.
    transducer : Transducer
        Transdutor considerado no problema.
    """
    def __init__(
        self,
        acoustic_lens: AcousticLens,
        pipeline: Pipeline,
        transducer: Transducer,
        loss_model: LossModel | None = None,
        modes: tuple[str, str, str] = ("L", "L", "L")
    ):
        super().__init__(pipeline, transducer, loss_model)

        self.acoustic_lens = acoustic_lens

        self.medium1 = deepcopy(acoustic_lens.wedge_medium)
        self.medium2 = deepcopy(acoustic_lens.coupling_medium)
        self.medium3 = deepcopy(pipeline.material)

        if set(modes) <= {"L", "T"}:
            self.modes = modes
        else:
            raise ValueError("Modes not allowed.")


    @property
    def c1(self):
        return self.medium1.cp if self.modes[0] == "L" else self.medium1.cs

    @property
    def c2(self):
        return self.medium2.cp if self.modes[1] == "L" else self.medium2.cs

    @property
    def c3(self):
        return self.medium3.cp if self.modes[2] == "L" else self.medium3.cs

    @c1.setter
    def c1(self, value):
        if self.modes[0] == "L":
            self.medium1._cp = value
        else:
            self.medium1._cs = value

    @c2.setter
    def c2(self, value):
        if self.modes[1] == "L":
            self.medium2._cp = value
        else:
            self.medium2._cs = value

    @c3.setter
    def c3(self, value):
        if self.modes[2] == "L":
            self.medium3._cp = value
        else:
            self.medium3._cs = value


