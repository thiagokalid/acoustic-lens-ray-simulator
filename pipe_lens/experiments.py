import numpy as np
from .types import Solid, Fluid
from .transducer import Transducer
from .pipeline import Pipeline
from .utils import convert_parameters
from .acoustic_lens import AcousticLens

__all__ = ['experimental_setup1']

def experimental_setup1():
    # Materials related:
    steel = Solid(cp=5900, cs=3200, density=7850, attenuation_coeff=[(0., 0.)])
    aluminium = Solid(cp=6332.94, cs=3128.18, density=2730, attenuation_coeff=[(0., 0.)])
    water = Fluid(cp=1430, density=997, attenuation_coeff=[(25e-15, 2.)])

    # Transducer related:
    pitch = .5e-3
    num_elem = 64
    element_gap = .1e-3
    central_freq = 5e6
    bw = .4
    bwr = -6
    pulse_type = "gaussian"

    # Pipe related:
    outer_radius = 70e-3
    pipe_material = steel
    wall_thickness = 18.5e-3
    center = (0., 0.)

    # Acoustic-lens related:
    wedge_material = aluminium
    coupling_medium = water
    d = 2 * 85e-3
    # h0 = 90e-3
    h0, _, _ = convert_parameters(c=d / 2, wc=8.6e-3, R=outer_radius, v1=wedge_material.cp, v2=coupling_medium.cp)
    alpha_max = np.radians(50)
    alpha_0 = 0


    # Create pipeline:
    pipeline = Pipeline(pipe_material, outer_radius, wall_thickness, center)

    # Create acoustic lens:
    acoustic_lens = AcousticLens(wedge_material, coupling_medium, d, alpha_max, alpha_0, h0, outer_radius)

    # Create transducer:
    transducer = Transducer(pitch, num_elem, element_gap, central_freq, bw, bwr, pulse_type,
                            center=(0., acoustic_lens.d))


    return steel, aluminium, water, acoustic_lens, transducer, pipeline