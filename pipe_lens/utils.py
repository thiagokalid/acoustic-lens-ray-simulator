import numpy as np

import json
import pathlib

__all__ = [
    'api_func',
    'api_polar',
    'api',
    'fwhm',
    'convert_corners_time2idx',
]


def api_func(img, t, ang, corner_in_time, thresh=.5, drawSquare=True):
    corners = convert_corners_time2idx(t, ang, corner_in_time)
    north_east_corner = corners[0]
    south_west_corner = corners[1]
    img_cropped = img[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]]
    local_max = np.max(img_cropped)
    maxLocationCoord = np.where(img_cropped == local_max)
    maxLocation = maxLocationCoord[1] + north_east_corner[1]
    img_cropped_masked = img_cropped > thresh * local_max
    img_masked = np.zeros_like(img)
    img_masked[north_east_corner[0]:south_west_corner[0],
    north_east_corner[1]:south_west_corner[1]] += img_cropped_masked
    api = np.sum(img_masked * 1.0) / len(img_masked)

    if drawSquare:
        width = 1
        scale_factor = int(img.shape[0] / img.shape[1])
        img_masked[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: north_east_corner[1] + width] = 1

        img_masked[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        south_west_corner[1] - width: south_west_corner[1] + width] = 1

        img_masked[north_east_corner[0] - width * scale_factor: north_east_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

        img_masked[south_west_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

    return api, maxLocation, img_masked, local_max


def api_polar(sscan, r_span, theta_deg_span, corners_in, thresh=-6, drawSquare=True):
    # Corner in [(r1, theta1), (r2, theta2)] in (mm, deg)

    # Converts (mm, deg) to (index, index):
    corners = convert_corners_time2idx(r_span, theta_deg_span, corners_in)  # Corner in index
    north_east_corner = corners[0]
    south_west_corner = corners[1]

    # Crops only the API Region of Interest (ROI):
    img_cropped = sscan[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]]

    # Obtain the maximum amplitude within the ROI:
    maximumAmplitude = np.max(img_cropped)

    # Creates a binary image where only pixels above thesh_db are highlighted
    pixels_above_threshold_within_roi = img_cropped >= maximumAmplitude * 10 ** (thresh / 20)
    pixels_above_threshold = np.zeros_like(sscan, dtype=bool)
    pixels_above_threshold[north_east_corner[0]:south_west_corner[0],
    north_east_corner[1]:south_west_corner[1]] += pixels_above_threshold_within_roi

    if np.any(pixels_above_threshold == True):
        maxLocationWithinROI = np.where(img_cropped == maximumAmplitude)
        maxAbsoluteLocation = maxLocationWithinROI[1] + north_east_corner[1]
    else:
        maxAbsoluteLocation = None

    # Compute the area of each pixel considering:
    # * All pixels above threshold belongs to the tube wall;
    # * The tube center is aligned with (0, 0) of the coordinate system;
    # * The sampling is constant along radius and angle axis;
    r_borders = moving_average(r_span)
    diff_i = r_borders[0] - r_span[0]
    diff_f = r_span[-1] - r_borders[-1]
    r_borders = np.append(r_span[0] - diff_i, r_borders)
    r_borders = np.append(r_borders, r_span[-1] + diff_f)

    #
    theta_borders = moving_average(theta_deg_span)
    diff_i = theta_borders[0] - theta_deg_span[0]
    diff_f = theta_deg_span[-1] - theta_borders[-1]
    theta_borders = np.append(theta_deg_span[0] - diff_i, theta_borders)
    theta_borders = np.append(theta_borders, theta_deg_span[-1] + diff_f)

    pixels_area = np.zeros_like(sscan)
    pixels_coords = np.zeros(shape=(*sscan.shape, 2))
    for i in range(sscan.shape[0]):
        for j in range(sscan.shape[1]):
            inner_radius = r_borders[i]
            outer_radius = r_borders[i + 1]
            theta1 = theta_borders[j]
            theta2 = theta_borders[j + 1]
            diff_theta = np.abs(theta2 - theta1)

            #
            pixels_area[i, j] = (outer_radius * 2 - inner_radius * 2) * np.pi * diff_theta / 360

            pixel_radius = (outer_radius + inner_radius) / 2
            pixel_ang = (theta1 + theta2) / 2
            pixels_coords[i, j, 0] = pixel_radius * np.cos(pixel_ang * np.pi / 360)
            pixels_coords[i, j, 1] = pixel_radius * np.sin(pixel_ang * np.pi / 360)
    pixels_area = np.abs(pixels_area)
    #
    api = np.sum(pixels_area[pixels_above_threshold])
    estimated_height = np.max(pixels_coords[pixels_above_threshold, 0]) - np.min(pixels_coords[pixels_above_threshold, 0])
    estimated_width = np.max(pixels_coords[pixels_above_threshold, 1]) - np.min(pixels_coords[pixels_above_threshold, 1])

    if drawSquare:
        width = 2
        scale_factor = int(sscan.shape[0] / sscan.shape[1])
        pixels_above_threshold[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: north_east_corner[1] + width] = 1

        pixels_above_threshold[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        south_west_corner[1] - width: south_west_corner[1] + width] = 1

        pixels_above_threshold[north_east_corner[0] - width * scale_factor: north_east_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

        pixels_above_threshold[south_west_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

    return api, maxAbsoluteLocation, pixels_above_threshold, maximumAmplitude, estimated_width, estimated_height

def convert_time2radius(tspan, t_outer, t_inner, cl_surf, cl_upper, cl_bottom=None, outer_r=70):
    # Converts the 'tspan' in us into a 'rspan' in mm, assuming that the outer-surface is located at 't_outer'
    # us, the inner-surface is located at 't_inner' us.
    if cl_bottom is None:
        cl_bottom = cl_upper
    if t_inner < t_outer:
        raise ValueError('t_inner must be smaller than t_outer')
    # Finds where the object profile begins and ends in index:
    tspan_copy = np.copy(tspan)
    beg_idx = np.argmin((tspan_copy - t_outer) ** 2)
    end_idx = np.argmin((tspan_copy - t_inner) ** 2)
    tspan_copy -= tspan_copy[beg_idx]
    tspan_copy *= -1

    # Converts the tspan into rspan
    rspan = np.zeros_like(tspan_copy)
    rspan[:beg_idx] = cl_bottom * (tspan_copy[:beg_idx] / 2) + outer_r
    rspan[beg_idx:end_idx] = cl_surf * (tspan_copy[beg_idx:end_idx] / 2) + outer_r
    rspan[end_idx:] = cl_upper * ((tspan_copy[end_idx:]) / 2)
    rspan[end_idx:] -= rspan[end_idx:].max()
    rspan[end_idx:] = rspan[end_idx:] + rspan[end_idx - 1]

    return rspan

def api(img, xaxis, zaxis, corners, thresh=-6, drawSquare=False):
    #corners = (superior-esquerdo, inferior-direito) considerando a imagem na visualizaçaõ eixo z para baixo e x para direita
    NE_corner, SW_corner = corners

    if (NE_corner[0] >= SW_corner[0] or NE_corner[1] <= SW_corner[0]):
        raise ValueError("Invalid Corners")

    xmin, xmax = NE_corner[0], SW_corner[0]
    zmin, zmax = NE_corner[1], SW_corner[1]

    i0, i1 = np.argmin(np.abs(xaxis - xmin)), np.argmin(np.abs(xaxis - xmax))
    j0, j1 = np.argmin(np.abs(zaxis - zmin)), np.argmin(np.abs(zaxis - zmax))

    pixel_area = (xaxis[1] - xaxis[0]) * (zaxis[1] - zaxis[0])

    croped_region = img[j0 : j1 + 1, i0 : i1 + 1]
    local_maximum = np.max(croped_region)
    binary_mask = croped_region >= local_maximum * 10 ** (thresh / 20)
    api = np.sum(binary_mask) * pixel_area

    return api, local_maximum, binary_mask

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

def fwhm(signal, xspan=None):
    # Compute the Full Width Half Maximum assuming only peak exist within the signal.
    # If xspan is not given, FWHM will be computed in number of samples.

    if xspan is None:
        xspan = np.arange(len(signal))

    return half_max_x(xspan, signal)


def convert_parameters(c, wc, R, v1, v2):
    z0 = -(2 * c * v2 - v1 * v2 * (((2 * c - (
            v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
            - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                    1 / 2) + v2 * wc * (
                    - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                    1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) / (2 * c * v1 ** 2)) ** 2 - (
                                            v1 ** 2 * (v2 ** 2 / v1 ** 2 - 1) * ((R + wc) ** 2 - c ** 2 + (c - (
                                            v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                            - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                    1 / 2) + v2 * wc * (
                                                    - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                    1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) / (
                                                                                                                   2 * c * v1 ** 2)) ** 2 - (
                                                                                         v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                                                         - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                 1 / 2) + v2 * wc * (
                                                                                                 - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                 1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) ** 2 / (
                                                                                         4 * c ** 2 * v1 ** 4)) ** 2) / (
                                            4 * c ** 2 * v2 ** 2)) ** (1 / 2) / v1 + ((
                                                                                              v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                                                              - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                      1 / 2) + v2 * wc * (
                                                                                                      - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                      1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) ** 2 / (
                                                                                              4 * c ** 2 * v1 ** 4) - (
                                                                                              v1 ** 2 * (
                                                                                              v2 ** 2 / v1 ** 2 - 1) * (
                                                                                                      (
                                                                                                              R + wc) ** 2 - c ** 2 + (
                                                                                                              c - (
                                                                                                              v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                                                                              - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                                      1 / 2) + v2 * wc * (
                                                                                                                      - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                                      1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) / (
                                                                                                                      2 * c * v1 ** 2)) ** 2 - (
                                                                                                              v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                                                                              - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                                      1 / 2) + v2 * wc * (
                                                                                                                      - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                                      1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) ** 2 / (
                                                                                                              4 * c ** 2 * v1 ** 4)) ** 2) / (
                                                                                              4 * c ** 2 * v2 ** 2)) ** (
                                           1 / 2) / v2)) / (v1 - v2)
    r0 = 2 * c + (2 * c * v2 - v1 * v2 * (((2 * c - (
            v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
            - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                    1 / 2) + v2 * wc * (
                    - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                    1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) / (2 * c * v1 ** 2)) ** 2 - (
                                                   v1 ** 2 * (v2 ** 2 / v1 ** 2 - 1) * (
                                                   (R + wc) ** 2 - c ** 2 + (c - (
                                                   v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                   - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                           1 / 2) + v2 * wc * (
                                                           - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                           1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) / (
                                                                                     2 * c * v1 ** 2)) ** 2 - (
                                                           v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                           - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                   1 / 2) + v2 * wc * (
                                                                   - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                   1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) ** 2 / (
                                                           4 * c ** 2 * v1 ** 4)) ** 2) / (
                                                   4 * c ** 2 * v2 ** 2)) ** (1 / 2) / v1 + ((
                                                                                                     v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                                                                     - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                             1 / 2) + v2 * wc * (
                                                                                                             - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                             1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) ** 2 / (
                                                                                                     4 * c ** 2 * v1 ** 4) - (
                                                                                                     v1 ** 2 * (
                                                                                                     v2 ** 2 / v1 ** 2 - 1) * (
                                                                                                             (
                                                                                                                     R + wc) ** 2 - c ** 2 + (
                                                                                                                     c - (
                                                                                                                     v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                                                                                     - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                                             1 / 2) + v2 * wc * (
                                                                                                                             - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                                             1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) / (
                                                                                                                             2 * c * v1 ** 2)) ** 2 - (
                                                                                                                     v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                                                                                     - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                                             1 / 2) + v2 * wc * (
                                                                                                                             - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                                                             1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) ** 2 / (
                                                                                                                     4 * c ** 2 * v1 ** 4)) ** 2) / (
                                                                                                     4 * c ** 2 * v2 ** 2)) ** (
                                                  1 / 2) / v2)) / (v1 - v2)
    max_alpha = -np.arctan((v1 ** 3 * (1 - v2 ** 2 / v1 ** 2) ** (1 / 2) * ((R + wc) ** 2 - c ** 2 + (c - (
            v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
            - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                    1 / 2) + v2 * wc * (
                    - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                    1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) / (2 * c * v1 ** 2)) ** 2 - (
                                                                                    v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                                                                    - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                            1 / 2) + v2 * wc * (
                                                                                            - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                                                                            1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc) ** 2 / (
                                                                                    4 * c ** 2 * v1 ** 4))) / (
                                   v2 * (
                                   v1 ** 2 * wc ** 2 - v2 ** 2 * wc ** 2 + R ** 2 * v1 ** 2 - R ** 2 * v2 ** 2 + R * v2 * (
                                   - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                           1 / 2) + v2 * wc * (
                                           - R ** 2 * v1 ** 2 + R ** 2 * v2 ** 2 - 2 * R * v1 ** 2 * wc + 2 * R * v2 ** 2 * wc + 4 * c ** 2 * v1 ** 2 - v1 ** 2 * wc ** 2 + v2 ** 2 * wc ** 2) ** (
                                           1 / 2) + 2 * R * v1 ** 2 * wc - 2 * R * v2 ** 2 * wc)))
    return z0, r0, max_alpha

def get_lens_params(version: str = ""):
    valid_versions = [
        "1.0.1", "1.1.0", "1.2.0", "1.4.1A", "1.4.1B", "1.5.0", "2.0.0", "2.0.1", "2.1.0"
    ]

    module_path = pathlib.Path(__file__)

    if version not in valid_versions:
        raise ValueError(
            f"Versão {version} não corresponde a um modelo de lente existente. Escolha entre {valid_versions}.")

    with open(module_path / "lenses_types.json", 'r') as file:
        lens_params = json.load(file)[version]

    # Define the order of keys
    keys = ["wc", "c", "c1", "c2", "offset", "r_outer", "wall_thickness", "rho1", "rho2", "alpha_max"]

    # Unpack values directly into variables
    wc, c, c1, c2, offset, r_outer, wall_thickness, rho1, rho2, alpha_max = (
        lens_params.get(k) for k in keys
    )

    return wc, c, c1, c2, offset, r_outer, wall_thickness, rho1, rho2, alpha_max

def convert_time2idx(t, value):
    return np.argmin(np.power(t - value, 2))


def convert_corners_time2idx(t, ang, corner_in_time):
    north_east_corner_row, north_east_corner_column = corner_in_time[0]
    south_west_corner_row, south_west_corner_column = corner_in_time[1]
    corner_in_idx = \
        [
            (convert_time2idx(t, north_east_corner_row), convert_time2idx(ang, north_east_corner_column)),
            (convert_time2idx(t, south_west_corner_row), convert_time2idx(ang, south_west_corner_column))
        ]
    return corner_in_idx

def moving_average(vector, n=2):
    n_elem = vector.shape[0]
    output = np.zeros(shape=n_elem - n + 1)
    for i in range(0, n_elem - n + 1):
        output[i] = np.mean(vector[i:i + n])
    return output
