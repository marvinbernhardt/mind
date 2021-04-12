#!/bin/env python3
################################################################################
#              ,--.   ,--. ,--.             ,--.                               #
#              |   `.'   | `--' ,--,--,   ,-|  |                               #
#              |  |'.'|  | ,--. |      \ ' .-. |                               #
#              |  |   |  | |  | |  ||  | \ `-' |                               #
#              `--'   `--' `--' `--''--'  `---'                                #
#                                                                              #
# ** A Minimal Lennard-Jones Fluid Molecular Dynamics Python Program **        #
#                                                                              #
#                                                                              #
#                                                                              #
# Authors: Pu Du, Marvin Bernhardt                                             #
# Website: https://github.com/marvinbernhardt/mind                             #
# Fork of: https://github.com/ipudu/mind                                       #
################################################################################

import time
import numpy as np
from numba import jit


@jit
def velocity_verlet(N, dt, rx, ry, rz, vx, vy, vz, fx, fy, fz):
    """Verloctiy verlet algorithm."""
    dt2 = dt * dt
    for i in range(N):
        rx[i] += vx[i] * dt + 0.5 * dt2 * fx[i]
        ry[i] += vy[i] * dt + 0.5 * dt2 * fy[i]
        rz[i] += vz[i] * dt + 0.5 * dt2 * fz[i]

        vx[i] += 0.5 * dt * fx[i]
        vy[i] += 0.5 * dt * fy[i]
        vz[i] += 0.5 * dt * fz[i]


@jit
def wrap_into_box(N, box, rx, ry, rz):
    """Wrap the coordinates."""
    for i in range(N):
        if rx[i] < 0.0:
            rx[i] += box[0]
        if rx[i] > box[0]:
            rx[i] -= box[0]
        if ry[i] < 0.0:
            ry[i] += box[1]
        if ry[i] > box[1]:
            ry[i] -= box[1]
        if rz[i] < 0.0:
            rz[i] += box[2]
        if rz[i] > box[2]:
            rz[i] -= box[2]


@jit
def correct_distance_pbc(box, box_half, dx, dy, dz):
    if dx > box_half[0]:
        dx -= box[0]
    elif dx < -box_half[0]:
        dx += box[0]
    if dy > box_half[1]:
        dy -= box[1]
    elif dy < -box_half[1]:
        dy += box[1]
    if dz > box_half[2]:
        dz -= box[2]
    elif dz < -box_half[2]:
        dz += box[2]
    return dx, dy, dz


@jit
def potential_energy(N, box, cut_off, rx, ry, rz, fx, fy, fz):
    """Calculate the potential energy and forces."""
    fx.fill(0)
    fy.fill(0)
    fz.fill(0)
    cut_off_squared = cut_off**2

    box_half = box / 2.0
    e = 0.0
    for i in range(N-1):
        for j in range(i+1, N):
            dx = rx[i] - rx[j]
            dy = ry[i] - ry[j]
            dz = rz[i] - rz[j]

            dx, dy, dz = correct_distance_pbc(box, box_half, dx, dy, dz)

            r2 = dx * dx + dy * dy + dz * dz

            if r2 < cut_off_squared:
                r6i = 1.0 / (r2 * r2 * r2)
                e += 4 * (r6i * r6i - r6i)
                f = 48 * (r6i * r6i - 0.5 * r6i)
                fx[i] += dx * f / r2
                fx[j] -= dx * f / r2
                fy[i] += dy * f / r2
                fy[j] -= dy * f / r2
                fz[i] += dz * f / r2
                fz[j] -= dz * f / r2
    return e


@jit
def kinetic_energy(N, dt, vx, vy, vz, fx, fy, fz):
    """Calculate the kinetic energy and does second half of Velocity verlet."""
    e = 0.0
    for i in range(N):
        vx[i] += 0.5 * dt * fx[i]
        vy[i] += 0.5 * dt * fy[i]
        vz[i] += 0.5 * dt * fz[i]
        e += vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
    e *= 0.5
    return e


@jit
def berendsen_thermostat(N, dt, T, tau, KE, vx, vy, vz):
    """Apply Berendsen thermostat."""
    lamb = np.sqrt(1 + dt / tau * (T / (2.0 * KE / 3.0 / N) - 1.0))
    for i in range(N):
        vx[i] *= lamb
        vy[i] *= lamb
        vz[i] *= lamb


def output_thermo(s, PE, KE, TE, T):
    """Print thermo information."""
    print("Step: {:9d} PE = {:10.4f} | KE = {:10.4f} | TE  = {:10.4f} | T = {:8.3f}"
          "".format(s, PE, KE, TE, T))


def mdrun(md_setup):
    """main MD function"""
    # unpack positions
    rx, ry, rz = np.array(md_setup['start_r']).copy()
    # unpack or initialize velocities
    N = len(md_setup['start_r'][0])
    if md_setup['start_v'] is None:
        vx = np.zeros(N)
        vy = np.zeros(N)
        vz = np.zeros(N)
    else:
        vx, vy, vz = np.array(md_setup['start_v']).copy()
    # initialize forces
    fx = np.zeros(N)
    fy = np.zeros(N)
    fz = np.zeros(N)

    traj = np.empty((md_setup['n_steps'], 3, N))
    traj.fill(np.nan)
    energies = np.empty((md_setup['n_steps'], 4))  # T, KE, PE, unused
    energies.fill(np.nan)

    for s in range(md_setup['n_steps']):
        velocity_verlet(N, md_setup['dt'], rx, ry, rz, vx, vy, vz, fx, fy, fz)
        wrap_into_box(N, md_setup['box'], rx, ry, rz)
        PE = potential_energy(N, md_setup['box'], md_setup['cut_off'], rx, ry, rz,
                              fx, fy, fz)
        KE = kinetic_energy(N, md_setup['dt'], vx, vy, vz,
                            fx, fy, fz)
        berendsen_thermostat(N, md_setup['dt'], md_setup['T'], md_setup['tau'], KE,
                             vx, vy, vz)
        T = KE / (3/2 * N)
        if s % md_setup['print_every_n_steps'] == 0:
            TE = PE + KE
            output_thermo(s, PE, KE, TE, T)
        if s % md_setup['save_traj_every_n_steps'] == 0:
            traj[s, 0, :] = rx
            traj[s, 1, :] = ry
            traj[s, 2, :] = rz
        if s % md_setup['save_energies_every_n_steps'] == 0:
            energies[s, 0] = T
            energies[s, 1] = KE
            energies[s, 2] = PE
    return traj, energies


if (__name__ == '__main__'):
    # initialize a cubic box with particles on a grid
    L = 6.0
    box = np.array([L, L, L])
    x_ = np.linspace(0., box[0], num=6, endpoint=False)
    y_ = np.linspace(0., box[1], num=6, endpoint=False)
    z_ = np.linspace(0., box[2], num=6, endpoint=False)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    rx = x.flatten()
    ry = y.flatten()
    rz = z.flatten()
    # MD setup
    md_setup = {
        'box': box,
        'start_r': (rx, ry, rz),
        'start_v': None,
        'dt': 0.001,
        'cut_off': 2.5,
        'n_steps': 10000,
        'T': 1.0,
        'tau': 0.1,
        'T_damp': 1,
        'print_every_n_steps': 100,
        'save_traj_every_n_steps': 1,
        'save_energies_every_n_steps': 1,
    }
    # run md
    t_start = time.perf_counter()
    traj, energies = mdrun(md_setup)
    np.savez_compressed('md_out.npz', traj=traj, energies=energies)
    t_end = time.perf_counter()
    print("Total looping time = {:.2f} seconds.".format(t_end - t_start))
