import numpy as np

def sinc_(theta):
    return np.sin(theta) / theta

def far_field_directivity(theta, element_width, lambda_w):
    theta = np.asarray(theta, dtype=np.complex64)
    return sinc_(np.pi * element_width * np.sin(theta) / lambda_w)

def F0(zeta, cL, cS):
    zeta = np.asarray(zeta, dtype=np.complex64)
    r = cL / cS
    term1 = (2 * zeta**2 - r**2)**2
    term2 = 4 * zeta**2 * np.sqrt(zeta**2 - 1) * np.sqrt(zeta**2 - r**2)
    return term1 - term2


def directivity_longitudinal_wave(theta, cp, cs):
    r = cp / cs
    numerator = (r**2 - 2 * np.sin(theta)**2) * np.cos(theta)
    denominator = F0(np.sin(theta), cp, cs)
    return numerator / denominator


def far_field_directivity_solid(omega, theta, element_width, cp, cs):
    lambda_w = 2 * np.pi * cp / omega
    Df = np.nan_to_num(far_field_directivity(theta, element_width, lambda_w), copy=True, nan=1.)
    DL = np.nan_to_num(directivity_longitudinal_wave(theta, cp, cs), copy=True, nan=1.)

    return Df * DL