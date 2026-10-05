import numpy as np
from numpy.typing import NDArray

def apply_time_reference(delay_law, method):
    match method:
        case 'max':
            return delay_law.max() - delay_law
        case 'mean':
            return delay_law.mean() - delay_law
        case _:
            raise ValueError

def planewave_delaylaw(speed: float, xt: NDArray, steering_angles_rad: NDArray, method='mean') -> NDArray:
    n_elements = len(xt)
    steering_angles_rad = [steering_angles_rad] if isinstance(steering_angles_rad, float) else steering_angles_rad
    n_angles = len(steering_angles_rad)
    delay_law = np.zeros(shape=(n_elements, n_angles))

    for i, ang in enumerate(steering_angles_rad):
        delay_law[:, i] = -xt * np.sin(ang)/ speed

    return apply_time_reference(delay_law, method)


def focusedwave_delaylaw(time_of_flight: NDArray, method = 'mean') -> NDArray:
    return apply_time_reference(time_of_flight, method)


