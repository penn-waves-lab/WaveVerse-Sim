#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the WaveVerse authors (2026): extended radio materials.
"""
Instantiates a set of radio materials for the scene objects.
These materials are from Table 3 of the Recommendation ITU-R P.2040-2.
"""
from functools import partial

import numpy as np
from .radio_material import RadioMaterial
from . import scene

def instantiate_itu_materials(dtype):
    #########################################
    # Vacuum (~ air)
    #########################################

    def vacuum_properties(f_hz): # pylint: disable=unused-argument
        return (1.0, 0.0)

    rm = RadioMaterial("vacuum",
                       frequency_update_callback=vacuum_properties,
                       dtype=dtype)
    scene.Scene().add(rm)


    #########################################
    # Concrete
    #########################################

    def concrete_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 100.:
            return (-1.0, -1.0)

        relative_permittivity = 5.24
        conductivity = 0.0462*np.power(f_ghz, 0.7822)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_concrete",
                      frequency_update_callback=concrete_properties,
                      dtype=dtype)
    scene.Scene().add(rm)

    ##########################################
    # Brick
    ##########################################

    def brick_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 40.:
            return (-1.0, -1.0)

        relative_permittivity = 3.91
        conductivity = 0.0238*np.power(f_ghz, 0.16)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_brick",
                       frequency_update_callback=brick_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Plasterboard
    #########################################

    def plasterboard_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 100.:
            return (-1.0, -1.0)


        relative_permittivity = 2.73
        conductivity = 0.0085*np.power(f_ghz, 0.9395)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_plasterboard",
                       frequency_update_callback=plasterboard_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Wood
    #########################################

    def wood_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 0.001 or f_ghz > 100.:
            return (-1.0, -1.0)

        relative_permittivity = 1.99
        conductivity = 0.0047*np.power(f_ghz, 1.0718)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_wood",
                       frequency_update_callback=wood_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Glass
    #########################################

    def glass_properties(f_hz):
        f_ghz = f_hz / 1e9
        if 0.1 <= f_ghz <= 100.:
            relative_permittivity = 6.31
            conductivity = 0.0036*np.power(f_ghz, 1.3394)
            return (relative_permittivity, conductivity)
        elif 220. <= f_ghz <= 450.:
            relative_permittivity = 5.79
            conductivity = 0.0004*np.power(f_ghz, 1.658)
            return (relative_permittivity, conductivity)
        else:
            return (-1.0, -1.0)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_glass",
                       frequency_update_callback=glass_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Ceiling board
    #########################################

    def ceiling_board_properties(f_hz):
        f_ghz = f_hz / 1e9
        if 1. <= f_ghz <= 100.:
            relative_permittivity = 1.48
            conductivity = 0.0011*np.power(f_ghz, 1.0750)
            return (relative_permittivity, conductivity)
        elif 220. <= f_ghz <= 450.:
            relative_permittivity = 1.52
            conductivity = 0.0029*np.power(f_ghz, 1.029)
            return (relative_permittivity, conductivity)
        else:
            return (-1.0, -1.0)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_ceiling_board",
                       frequency_update_callback=ceiling_board_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Chipboard
    #########################################

    def chipboard_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 100.0:
            return (-1.0, -1.0)

        relative_permittivity = 2.58
        conductivity = 0.0217*np.power(f_ghz, 0.7800)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_chipboard",
                       frequency_update_callback=chipboard_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Plywood
    #########################################

    def plywood_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 40.0:
            return (-1.0, -1.0)

        relative_permittivity = 2.71
        conductivity = 0.33
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_plywood",
                       frequency_update_callback=plywood_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Marble
    #########################################

    def marble_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 60.0:
            return (-1.0, -1.0)

        relative_permittivity = 7.074
        conductivity = 0.0055*np.power(f_ghz, 0.9262)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_marble",
                       frequency_update_callback=marble_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Floorboard
    #########################################

    def floorboard_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 50.0 or f_ghz > 100.0:
            return (-1.0, -1.0)

        relative_permittivity = 3.66
        conductivity = 0.0044*np.power(f_ghz, 1.3515)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_floorboard",
                       frequency_update_callback=floorboard_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Metal
    #########################################

    def metal_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 100.0:
            return (-1.0, -1.0)

        relative_permittivity = 1.0
        conductivity = 1e7
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_metal",
                       frequency_update_callback=metal_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Very dry ground
    #########################################

    def very_dry_ground_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 10.0:
            return (-1.0, -1.0)

        relative_permittivity = 3.0
        conductivity = 0.00015*np.power(f_ghz, 2.52)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_very_dry_ground",
                       frequency_update_callback=very_dry_ground_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Medium dry ground
    #########################################

    def medium_dry_ground_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 10.0:
            return (-1.0, -1.0)

        relative_permittivity = 15.0*np.power(f_ghz, -0.1)
        conductivity = 0.035*np.power(f_ghz, 1.63)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_medium_dry_ground",
                       frequency_update_callback=medium_dry_ground_properties,
                       dtype=dtype)
    scene.Scene().add(rm)

    #########################################
    # Wet ground
    #########################################

    def wet_ground_properties(f_hz):
        f_ghz = f_hz / 1e9
        if f_ghz < 1.0 or f_ghz > 10.0:
            return (-1.0, -1.0)

        relative_permittivity = 30.0*np.power(f_ghz, -0.4)
        conductivity = 0.15*np.power(f_ghz, 1.30)
        return (relative_permittivity, conductivity)

    # Materials parameters will be updated when the frequency is set
    rm = RadioMaterial("itu_wet_ground",
                       frequency_update_callback=wet_ground_properties,
                       dtype=dtype)
    scene.Scene().add(rm)


    #########################################
    # extended materials
    #########################################
    extended_itu_materials_properties = {
        "plastic": {(1.0, 100.): (2.4, 0.0, 0.002, 1.1)},
        'fabric': {(1.0, 100.): (1.8, 0.0, 0.0015, 1.2)},
        "leather": {(1.0, 100.): (2.1, 0.0, 0.003, 1.1)},
        "carpet": {(1.0, 100.): (1.5, 0.0, 0.001, 1.3)},
        "paper": {(1.0, 100.): (2.0, 0.0, 0.0025, 1.15)},
        "rubber": {(1.0, 100.): (2.8, 0.0, 0.004, 1.2)},
        "wax": {(1.0, 100.): (2.2, 0.0, 0.001, 1.0)},
        "wicker": {(1.0, 100.): (1.85, 0.0, 0.003, 1.1)},
        "ceramic": {(1.0, 100.): (8.5, 0.0, 0.015, 1.2)},
        "food": {(1.0, 100.): (4.5, 0.0, 0.03, 1.4)},
        # extrapolate built-in materials by valid frequency ranges
        "concrete_ex": {(1.0, 100.): (5.24, 0.0, 0.0462, 0.7822)},
        "brick_ex": {(1.0, 100.): (3.91, 0.0, 0.0238, 0.16)},
        "plasterboard_ex": {(1.0, 100.): (2.73, 0.0, 0.0085, 0.9395)},
        "wood_ex": {(0.001, 100.): (1.99, 0.0, 0.0047, 1.0718)},
        "glass_ex": {(0.1, 100.): (6.31, 0.0, 0.0036, 1.3394),
                     (220., 450.): (5.79, 0.0, 0.0004, 1.658)},
        "ceiling_board_ex": {(1., 100.): (1.48, 0.0, 0.0011, 1.0750),
                             (220., 450.): (1.52, 0.0, 0.0029, 1.029)},
        "chipboard_ex": {(1.0, 100.): (2.58, 0.0, 0.0217, 0.7800)},
        "plywood_ex": {(1.0, 100.): (2.71, 0.0, 0.33, 0.0)},
        "marble_ex": {(1.0, 100.): (7.074, 0.0, 0.0055, 0.9262)},
        "floorboard_ex": {(1.0, 100.): (3.66, 0.0, 0.0044, 1.3515)},
        "metal_ex": {(1.0, 100.): (1.0, 0.0, 1e7, 0.0)},
        "very_dry_ground_ex": {(1.0, 100.): (3.0, 0.0, 0.00015, 2.52)},
        "medium_dry_ground_ex": {(1.0, 100.): (15.0, -0.1, 0.035, 1.63)},
        "wet_ground_ex": {(1.0, 100.): (30.0, -0.4, 0.15, 1.30)}
    }

    def itu_freq_cb(f, name, props):
        f_ghz = f / 1e9

        # Extract the properties to use according to the frequency
        # If the frequency is in none of the valid ranges, an exception is raised
        valid_freq = False
        for f_ranges, params in props.items():
            if f_ranges[0] < f_ghz < f_ranges[1]:
                a, b, c, d = params
                valid_freq = True
                break
        if not valid_freq:
            msg = f"Properties of ITU material '{name}' are not defined for" \
                  " this frequency"
            raise ValueError(msg)

        # Evaluate the material properties
        eta_r = a * np.power(f_ghz, b)
        sigma = c * np.power(f_ghz, d)

        return eta_r, sigma

    for material_name, material_freq_parameters in extended_itu_materials_properties.items():
        rm = RadioMaterial(
            f"itu_{material_name}",
            frequency_update_callback=partial(itu_freq_cb, name=material_name, props=material_freq_parameters),
            dtype=dtype
        )
        scene.Scene().add(rm)
