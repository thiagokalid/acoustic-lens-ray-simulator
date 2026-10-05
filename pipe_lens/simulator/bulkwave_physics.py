from numpy import sin, cos, arcsin, power, sqrt
from pipe_lens.types import Material, Fluid, Solid
from pipe_lens.types import Ray
from numpy.typing import NDArray
from typing import Tuple

import numpy as np


# Diffraction related phenomena:

def sinc_(theta):
    return np.sin(theta) / theta

def far_field_directivity(theta, element_width, lambda_w):
    theta = np.asarray(theta, dtype=np.complex64)
    return sinc_(np.pi / lambda_w * element_width * np.sin(theta) )

def F0(zeta, cL, cS):
    zeta = np.asarray(zeta, dtype=np.complex64)
    r = cL / cS
    term1 = (2 * zeta**2 - r**2)**2
    term2 = 4 * zeta**2 * sqrt(zeta**2 - 1) * sqrt(zeta**2 - r**2)
    return term1 - term2


def directivity_longitudinal_wave(theta, cp, cs):
    r = cp / cs
    numerator = (r**2 - 2 * np.sin(theta)**2) * np.cos(theta)
    denominator = F0(np.sin(theta), cp, cs)
    return numerator / denominator


def far_field_directivity_solid(omega, theta, element_width, cp, cs, baffle_type="soft"):
    lambda_w = 2 * np.pi * cp / omega
    Df = np.nan_to_num(far_field_directivity(theta, element_width, lambda_w), copy=True, nan=1.)
    DL = np.nan_to_num(directivity_longitudinal_wave(theta, cp, cs), copy=True, nan=1.)
    return Df * DL * compute_baffling(theta, baffle_type)

def compute_baffling(theta, baffle_type):
    match baffle_type:
        case "rigid":
            return np.ones_like(theta)
        case "soft":
            return np.cos(theta)
        case _:
            raise ValueError("Invalid baffle type.")

# Propagation related phenomena:

def propagation_delay(delta_t):
    return np.exp(-delta_t * delta_t)

def two_media_geom_spread(r1, r2, theta1, theta2, c1, c2):
    term = c2/c1 * (np.cos(theta1)**2)/(np.cos(theta2)**2)
    den = np.sqrt(r1 + term * r2)
    return 1/den

def one_medium_geom_spread(propagation_distance):
    return 1/sqrt(propagation_distance)

def compute_attenuation(omega, medium, distance):
    frequencies = omega / 2 * np.pi
    alpha = medium.alpha(frequencies)
    return np.exp(-alpha * distance)

def compute_ray_attenuation(omega: np.ndarray, ray: Ray) -> NDArray[np.float32]:
    segmentwise_attenuation = [compute_attenuation(omega, medium, segment) for segment, medium in zip(ray.segment_dist, ray.medium)]
    return np.prod(np.asarray(segmentwise_attenuation), axis=0)

# Transmission related phenomena:

def validate_ang(theta, eps=np.radians(1)):
    theta = np.asarray(theta)

    if np.any(np.abs(theta) > np.pi/2 + eps):
        raise ValueError("Invalid angle value.")

    # Snap values very close to the limits
    theta = np.where(np.abs(theta) < eps, 0.0, theta)
    theta = np.where(np.abs(theta - np.pi) < eps, np.pi, theta)

    return theta

def compute_tension_transmission_coefficient(inc_ang, trans_ang, medium1: Material, medium2: Material, inc_mode="L", trans_mode="L"):
    if isinstance(medium1, Fluid) and isinstance(medium2, Fluid):
        Tpp_12 = fluid2fluid_t_coeff(inc_ang, trans_ang, medium1, medium2)
        coeff = Tpp_12
    elif isinstance(medium1, Fluid) and isinstance(medium2, Solid):
        Tpp_12 = fluid2solid_t_coeff(inc_ang, trans_ang, medium1, medium2)
        coeff = Tpp_12
    elif isinstance(medium1, Solid) and isinstance(medium2, Solid):
        Tpp_12 = solid2solid_t_coeff(inc_ang, trans_ang, medium1, medium2)
        coeff = Tpp_12
    elif isinstance(medium1, Solid) and isinstance(medium2, Fluid):
        # 2 -> 1 where inc_ang at 2 and trans_ang at 2. m1 = solid and m2 = liquid
        theta_p_1 = trans_ang
        theta_p_2 = inc_ang
        m1 = medium2
        m2 = medium1

        # Stolke's relation:
        Tpp_12 = fluid2solid_t_coeff(theta_p_1, theta_p_2, m1, m2)
        Tpp_21 = m1.acoustic_impedance / m2.acoustic_impedance * np.cos(theta_p_2) / np.cos(theta_p_1) * Tpp_12

        # Result:
        coeff = Tpp_21
    else:
        raise ValueError("Medium must be either Solid or Fluid or mode is invalid with medium type.")

    return np.abs(coeff)

def compute_tension_reflection_coefficient(inc_ang, trans_ang, medium1: Material, medium2: Material, inc_mode="L", reflect_mode="L"):
    """ Função que calcula o coeficiente de transmissão baseado em frações de tensão normal, ou seja, a fração entre
    amplitude da pressão da onda incidente para a pressão da onda transmitida de um meio 1 para um meio 2.

    Parameters
    ----------
    inc_ang:
        Ângulo de incidência da onda P no meio 1.
    trans_ang:
        Ângulo de incidência da onda P no meio 2.
    medium1: data_type.Material
        Material do meio de onde parte a onda.
    medium2: data_type.Material
        Material do meio onde a onda seria/é transmitida.

    Returns
    -------
    Rpp_12:
        Array 1-D com valores que representam a fração entre a pressão da onda incidente e da onda refletida.

    """
    if isinstance(medium1, Fluid) and isinstance(medium2, Fluid):
        Rpp_12 = fluid2fluid_r_coeff(inc_ang, trans_ang, medium1, medium2)
    elif isinstance(medium1, Fluid) and isinstance(medium2, Solid):
        Rpp_12 = fluid2solid_r_coeff(inc_ang, trans_ang, medium1, medium2)
    elif isinstance(medium1, Solid) and isinstance(medium2, Solid):
        Rpp_12 = solid2solid_r_coeff(inc_ang, trans_ang, medium1, medium2)
    elif isinstance(medium1, Solid) and isinstance(medium2, Fluid):
        Rpp_21 = fluid2solid_r_coeff(trans_ang, inc_ang, medium2, medium1)
        Rpp_12 = -Rpp_21 # Schmer Eq. 6.80 (Stokes Relation)
    else:
        raise ValueError("Medium must be either Solid or Fluid.")

    return Rpp_12


def fluid2solid_t_coeff(inc_ang, trans_ang, medium1: Fluid, medium2: Solid):
    theta_p1 = abs(inc_ang)
    theta_p2 = abs(trans_ang)

    # Extract meaningful variables regarding the media:
    cp1, rho1 = medium1.cp, medium1.density
    cp2, cs2, rho2 = medium2.cp, medium2.cs, medium2.density

    # Equation 6.118:
    theta_s2 = arcsin(cs2 * sin(theta_p2) / cp2)

    # Equation 6.120 from Fundamentals of Ultrasonic (Schmerr 2016)
    # Delta 1 is analogous with the acoustic impedance Z1:
    delta1 = cos(theta_p2)
    # Delta 2 is analogous with the acoustic impedance Z2:
    delta2 = \
        (rho2 * cp2 * cos(theta_p1)) / (rho1 * cp1) * (
                power(cos(2 * theta_s2), 2) + (cs2 ** 2 * sin(2 * theta_s2) * sin(2 * theta_p2)) / (
                cp2 ** 2)
        )

    delta = delta1 + delta2

    # Transmission coefficient from an incident wave of type P (pressure) to a transmitted wave of type P (pressure):
    Tpp_12 = -(2 * rho2 * cp2 * cos(theta_p1) * cos(2 * theta_s2)) / (rho1 * cp1 * delta)  # Equation 6.122a

    return Tpp_12

def fluid2solid_r_coeff(inc_ang, trans_ang, medium1: Fluid, medium2: Solid):
    theta_p1 = abs(inc_ang)
    theta_p2 = abs(trans_ang)

    # Extract meaningful variables regarding the media:
    cp1, rho1 = medium1.cp, medium1.density
    cp2, cs2, rho2 = medium2.cp, medium2.cs, medium2.density

    # Equation 6.118:
    theta_s2 = arcsin(cs2 * sin(theta_p2) / cp2)

    # Equation 6.120 from Fundamentals of Ultrasonic (Schmerr 2016)
    # Delta 1 is analogous with the acoustic impedance Z1:
    delta1 = cos(theta_p2)
    # Delta 2 is analogous with the acoustic impedance Z2:
    delta2 = \
        (rho2 * cp2 * cos(theta_p1)) / (rho1 * cp1) * (
                power(cos(2 * theta_s2), 2) + (cs2 ** 2 * sin(2 * theta_s2) * sin(2 * theta_p2)) / (
                cp2 ** 2)
        )

    Rpp_12 = (delta2 - delta1) / (delta2 + delta1)

    return Rpp_12

def fluid2fluid_t_coeff(inc_ang, trans_ang, medium1: Fluid, medium2: Fluid):
    theta_i = abs(inc_ang)
    theta_t = abs(trans_ang)

    # Extract meaningful variables regarding the media:
    c1, rho1 = medium1.cp, medium1.density
    c2, rho2 = medium2.cp, medium2.density

    # Transmission coefficient:
    Tp = (2 * rho2 * c2 * cos(theta_i))/(rho2*c2*cos(theta_i) + rho1*c1*cos(theta_t))

    return Tp

def fluid2fluid_r_coeff(inc_ang, trans_ang, medium1: Fluid, medium2: Fluid):
    theta_i = abs(inc_ang)
    theta_t = abs(trans_ang)

    # Extract meaningful variables regarding the media:
    c1, rho1 = medium1.cp, medium1.density
    c2, rho2 = medium2.cp, medium2.density

    # Transmission coefficient:
    Rp = (rho2*c2*cos(theta_i) - rho1*c1*cos(theta_t))/(rho2*c2*cos(theta_i) + rho1*c1*cos(theta_t))

    return Rp

def solid2solid_t_coeff(inc_ang, trans_ang, medium1: Solid, medium2: Solid):
    theta_p1 = abs(inc_ang)
    theta_p2 = abs(trans_ang)

    # Extract meaningful variables regarding the media:
    cp1, cs1, rho1 = medium1.cp, medium1.cs, medium1.density
    cp2, cs2, rho2 = medium2.cp, medium2.cs, medium2.density

    # Snell's Law to find shear angles
    theta_s1 = arcsin(cs1 * sin(theta_p1) / cp1)
    theta_s2 = arcsin(cs2 * sin(theta_p2) / cp2)

    # Intermediate terms from Eqs. (6.162) and (6.163)
    d1 = (cs1 / cp1) * sin(2 * theta_s1) * sin(theta_p1) + cos(2 * theta_s1) * cos(theta_s1)
    d2 = (cs1 / cp1) * sin(2 * theta_p1) * sin(theta_s1) + cos(2 * theta_s1) * cos(theta_p1)

    l1 = (cs1 / cp2) * sin(2 * theta_s1) * sin(theta_p2) + (rho2 / rho1) * cos(2 * theta_s2) * cos(theta_s1)
    m1 = (cs1 / cs2) * sin(2 * theta_s1) * cos(theta_s2) - (rho2 / rho1) * sin(2 * theta_s2) * cos(theta_s1)
    l2 = (cs1 / cp2) * cos(2 * theta_s1) * sin(theta_p2) - (rho2 / rho1) * cos(2 * theta_s2) * sin(theta_s1)
    m2 = (cs1 / cs2) * cos(2 * theta_s1) * cos(theta_s2) + (rho2 / rho1) * sin(2 * theta_s2) * sin(theta_s1)
    l3 = (cp1 / cp2) * cos(2 * theta_s1) * cos(theta_p2) + (rho2 * cs2 ** 2) / (rho1 * cp2 ** 2) * sin(
        2 * theta_p2) * sin(theta_p1)
    m3 = -(cp1 / cs2) * cos(2 * theta_s1) * sin(theta_s2) + rho2 / rho1 * cos(2 * theta_s2) * sin(theta_p1)
    l4 = -(cs1 ** 2) / (cp1 * cp2) * sin(2 * theta_p1) * cos(theta_p2) + (rho2 * cs2 ** 2) / (
                rho1 * cp2 ** 2) * sin(
        2 * theta_p2) * cos(theta_p1)
    m4 = (cs1 ** 2) / (cp1 * cs2) * sin(2 * theta_p1) * sin(theta_s2) + rho2 / rho1 * cos(2 * theta_s2) * cos(
        theta_p1)

    # Denominator from Eq. (6.166)
    delta = (l2 / d1 + l4 / d2) * (m1 / d1 + m3 / d2) - (l1 / d1 + l3 / d2) * (m2 / d1 + m4 / d2)

    #
    delta1 = cs1 / cp1 * np.sin(2 * theta_s1) * np.sin(theta_p1) + np.cos(2 * theta_s1) * np.cos(theta_s1)
    delta2 = cs1 / cp1 * np.sin(2 * theta_p1) * np.sin(theta_s1) + np.cos(2 * theta_s1) * np.cos(theta_p1)

    # Potential amplitude ratios from Eq. (6.164)
    At_Ai = -2 * ((m2 / delta1) + (m4 / delta2)) / delta

    # Convert to transmission coefficient:
    Tpp_12 = At_Ai * rho2 / rho1  # Eq. 4.28

    return Tpp_12

def solid2solid_r_coeff(inc_ang, trans_ang, medium1: Solid, medium2: Solid):
    theta_p1 = abs(inc_ang)
    theta_p2 = abs(trans_ang)

    # Extract meaningful variables regarding the media:
    cp1, cs1, rho1 = medium1.cp, medium1.cs, medium1.density
    cp2, cs2, rho2 = medium2.cp, medium2.cs, medium2.density

    # Snell's Law to find shear angles
    theta_s1 = arcsin(cs1 * sin(theta_p1) / cp1)
    theta_s2 = arcsin(cs2 * sin(theta_p2) / cp2)

    # Intermediate terms from Eqs. (6.162) and (6.163)
    d1 = (cs1 / cp1) * sin(2 * theta_s1) * sin(theta_p1) + cos(2 * theta_s1) * cos(theta_s1)
    d2 = (cs1 / cp1) * sin(2 * theta_p1) * sin(theta_s1) + cos(2 * theta_s1) * cos(theta_p1)

    l1 = (cs1 / cp2) * sin(2 * theta_s1) * sin(theta_p2) + (rho2 / rho1) * cos(2 * theta_s2) * cos(theta_s1)
    m1 = (cs1 / cs2) * sin(2 * theta_s1) * cos(theta_s2) - (rho2 / rho1) * sin(2 * theta_s2) * cos(theta_s1)
    l2 = (cs1 / cp2) * cos(2 * theta_s1) * sin(theta_p2) - (rho2 / rho1) * cos(2 * theta_s2) * sin(theta_s1)
    m2 = (cs1 / cs2) * cos(2 * theta_s1) * cos(theta_s2) + (rho2 / rho1) * sin(2 * theta_s2) * sin(theta_s1)
    l3 = (cp1 / cp2) * cos(2 * theta_s1) * cos(theta_p2) + (rho2 * cs2 ** 2) / (rho1 * cp2 ** 2) * sin(
        2 * theta_p2) * sin(theta_p1)
    m3 = -(cp1 / cs2) * cos(2 * theta_s1) * sin(theta_s2) + rho2 / rho1 * cos(2 * theta_s2) * sin(theta_p1)
    l4 = -(cs1 ** 2) / (cp1 * cp2) * sin(2 * theta_p1) * cos(theta_p2) + (rho2 * cs2 ** 2) / (
            rho1 * cp2 ** 2) * sin(
        2 * theta_p2) * cos(theta_p1)
    m4 = (cs1 ** 2) / (cp1 * cs2) * sin(2 * theta_p1) * sin(theta_s2) + rho2 / rho1 * cos(2 * theta_s2) * cos(
        theta_p1)

    # Denominator from Eq. (6.166)
    delta = (l2 / d1 + l4 / d2) * (m1 / d1 + m3 / d2) - (l1 / d1 + l3 / d2) * (m2 / d1 + m4 / d2)

    # Potential amplitude ratios from Eq. (6.164)
    Ar_Ai = ((l2 / d1 - l4 / d2) * (m1 / d1 - m3 / d2) - (l1 / d1 - l3 / d2) * (m2 / d1 + m4 / d2)) / delta

    # Convert potential amplitude ratio to reflection and transmission coefficient (Eq. 4.28):
    Rpp_12 = Ar_Ai * rho1 / rho1

    return Rpp_12