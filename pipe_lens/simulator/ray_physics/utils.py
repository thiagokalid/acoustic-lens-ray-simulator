import  numpy as np
from numpy import pi, sin, cos

# def snell(incident, normal, c1, c2):
#     incident = np.asarray(incident, dtype=float)
#     normal = np.asarray(normal, dtype=float)
#
#     single_input = incident.ndim == 1
#
#     if single_input:
#         incident = incident[None, :]
#         normal = normal[None, :]
#
#     # Normalize vectors
#     i = incident / np.linalg.norm(incident, axis=1, keepdims=True)
#     n = normal / np.linalg.norm(normal, axis=1, keepdims=True)
#
#     mu = c2/c1
#
#     #
#     cos_theta1 = -np.sum(n * i, axis=1)
#
#     # Replace n by its negative whenever yields cos_theta1 < 0:
#     n[cos_theta1 < 0] = -n[cos_theta1 < 0]
#
#     # Recompute cos(theta1):
#     cos_theta1 = -np.sum(n * i, axis=1)
#
#     radicand = 1 - mu**2 * (1 - cos_theta1)**2
#     cos_theta2 = np.sqrt(radicand)
#
#     v_refract = mu * i + np.tile(mu * cos_theta1 - cos_theta2, reps=(2, 1)).T * n
#     v_reflect = i + 2 * np.tile(cos_theta1, reps=(2, 1)).T * n
#
#     T = v_refract
#     R = v_reflect
#
#     theta_i = angle_between(incident, normal)
#     theta_r = angle_between(R, normal)
#     theta_t = angle_between(T, normal)
#
#     return T, R, theta_i, theta_t, theta_r

def circle_cartesian(radius, xc, zc, alpha_step=0.1):
    alpha_step_rad = np.deg2rad(alpha_step)
    alpha = np.arange(0, 2 * np.pi, alpha_step_rad)

    x = xc + radius * np.cos(alpha)
    z = zc + radius * np.sin(alpha)

    return x, z


def roots_bhaskara(a, b, c):
    """
    Resolve a equação quadrática de segundo grau usando a fórmula de Bhaskara.

    Parameters
    ----------
    a : float
        Coeficiente quadrático.
    b : float
        Coeficiente linear.
    c : float
        Termo constante.

    Returns
    -------
    x1 : float
        Primeira raiz real.
    x2 : float
        Segunda raiz real.

    Notes
    -----
    A função não verifica se o discriminante é negativo.
    """
    sqrt_delta = np.sqrt(np.square(b) - 4 * a * c)
    den = 2 * a
    x1 = (-b + sqrt_delta) / den
    x2 = (-b - sqrt_delta) / den
    return x1, x2



def rhp(x):
    '''Projects an angle to the Right Half Plane [-pi/2; pi/2]'''
    x = np.mod(x, pi)
    x = x - (x > pi / 2) * pi
    x = x + (x < -pi / 2) * pi
    return x


def uhp(x):
    '''Projects an angle to the Upper Half Plane [0; pi]'''
    x = rhp(x)
    x = x + (x < 0) * pi
    return x

def snell(v1, v2, gamma1, dydx):
    """Computes the the new angle  after the refraction
  of the first angle with the Snell's law (top-down)"""
    gamma1 = uhp(gamma1)
    slope = rhp(np.arctan(dydx))
    normal = slope + pi / 2
    theta1 = gamma1 - normal
    arg = sin(theta1) * v2 / v1
    bad_index = np.abs(arg) > 1
    # Forcing the argument to be always within the [-1, 1] interval:
    arg[bad_index] = np.tanh(arg[bad_index])
    theta2 = np.arcsin(arg)
    gamma2 = slope - pi / 2 + theta2
    return gamma2, theta1, theta2