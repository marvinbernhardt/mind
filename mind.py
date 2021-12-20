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
from numba import njit


@njit
def velocity_verlet(N, dt, m, rx, ry, rz, vx, vy, vz, fx, fy, fz):
    """Verloctiy verlet algorithm."""
    dt2 = dt * dt
    for i in range(N):
        rx[i] += vx[i] * dt + 0.5 * dt2 * fx[i] / m
        ry[i] += vy[i] * dt + 0.5 * dt2 * fy[i] / m
        rz[i] += vz[i] * dt + 0.5 * dt2 * fz[i] / m

        vx[i] += 0.5 * dt * fx[i] / m
        vy[i] += 0.5 * dt * fy[i] / m
        vz[i] += 0.5 * dt * fz[i] / m


@njit
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
        # if the particle is still not in the box, it moved too fast
        if (rx[i] < 0.0 or rx[i] > box[0]
                or ry[i] < 0.0 or ry[i] > box[1]
                or rz[i] < 0.0 or rz[i] > box[2]):
            return False
    return True


@njit
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


@njit
def potential_energy_LJ(N, box, cut_off, rx, ry, rz, fx, fy, fz):
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


@njit
def potential_energy_table(N, box, cut_off, rx, ry, rz, fx, fy, fz, u_table,
                           fr_table, min_r, Delta_r):
    """Calculate the potential energy and forces using tabulated potential which is
    linearly interpolated in r².
    """
    fx.fill(0)
    fy.fill(0)
    fz.fill(0)
    cut_off_squared = cut_off**2

    box_half = box / 2.0
    Delta_r_half = Delta_r / 2
    e = 0.0
    for i in range(N-1):
        for j in range(i+1, N):
            dx = rx[i] - rx[j]
            dy = ry[i] - ry[j]
            dz = rz[i] - rz[j]

            dx, dy, dz = correct_distance_pbc(box, box_half, dx, dy, dz)

            r2 = dx * dx + dy * dy + dz * dz
            r = np.sqrt(r2)

            if r2 < cut_off_squared:
                index = int((r - min_r + Delta_r_half) // Delta_r)
                e += u_table[index]
                fr = fr_table[index]
                fx[i] += dx * fr / r2
                fx[j] -= dx * fr / r2
                fy[i] += dy * fr / r2
                fy[j] -= dy * fr / r2
                fz[i] += dz * fr / r2
                fz[j] -= dz * fr / r2
    return e


@njit
def check_arrays_for_nans(*arrays):
    """Check arrays for any NaN."""
    for array in arrays:
        if np.isnan(np.dot(array, array)):
            return True
    return False


@njit
def kinetic_energy(N, dt, m, vx, vy, vz, fx, fy, fz):
    """Calculate the kinetic energy and does second half of Velocity verlet."""
    e = 0.0
    for i in range(N):
        vx[i] += 0.5 * dt * fx[i] / m
        vy[i] += 0.5 * dt * fy[i] / m
        vz[i] += 0.5 * dt * fz[i] / m
        e += m * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
    e *= 0.5
    return e


@njit
def berendsen_thermostat(N, dt, T, tau, KE, k_B, vx, vy, vz):
    """Apply Berendsen thermostat."""
    lamb = np.sqrt(1 + dt / tau * (T / (2.0 * KE / 3.0 / N / k_B) - 1.0))
    for i in range(N):
        vx[i] *= lamb
        vy[i] *= lamb
        vz[i] *= lamb


def output_thermo(s, PE, KE, TE, T):
    """Print thermo information."""
    print(("Step: {:9d} PE = {:10.4f} | KE = {:10.4f} | TE  = {:10.4f} | T = {:8.3f}"
           ).format(s, PE, KE, TE, T))


def mdrun(md_setup):
    """Run molecular dynamics simulation.

    Arguments:
        md_setup: A dictionary with all data needed for the simulation.
        Three of its entries can be Numpy arrays:
        'box':     x, y, and z, length of the orthonormal box. Numpy array with
                   shape (3).
        'start_r': Start configuration, Numpy array with shape (3, N). N is the number
                   of atoms.
        'start_v': Start velocitiy, either None (all velocities zero) or Numpy array
                   with shape (3, N).
        'r_u_table': A 2D numpy array defining the pair potential. First dimension is r,
                     second is u(r).

    Returns:
        A tuple with two elements.
        traj:     The trajectory, a Numpy array with shape (md_setup['n_steps']
                  // md_setup['save_traj_every_n_steps'], 3, N).
        energies: The calculated energies, a Numpy array with shape (md_setup['n_steps']
                  // md_setup['save_energies_every_n_steps'], 5)).
                  The four dimensions are T, E_kin, and E_pot.
    """
    # unpack table
    if 'r_u_table' in md_setup:
        r_table = md_setup['r_u_table'].copy()[0, :]
        u_table = md_setup['r_u_table'].copy()[1, :]
        Delta_r = r_table[1] - r_table[0]
        min_r = np.min(r_table)
        # get force from potential
        fr_table = -np.array([0, *list(1/2 * (np.diff(u_table[1:])
                                              + np.diff(u_table[:-1]))), 0])
        fr_table /= Delta_r  # /Δr
        # multiply now by r, so later we divide by r²
        fr_table *= r_table
        if r_table[-1] < md_setup['cut_off']:
            raise Exception("the last r of r_u_table has be larger than cut-off")
        if not np.allclose(r_table[-1], md_setup['cut_off']):
            print("Warning: the last r of r_u_table is not equal to cut_off")
    else:
        u_table = None
        fr_table = None
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
    # copy box
    box = np.copy(md_setup['box'])
    # prepare traj and energies
    traj = np.empty((md_setup['n_steps'] // md_setup['save_traj_every_n_steps'], 3, N))
    traj.fill(np.nan)
    energies = np.empty((md_setup['n_steps'] // md_setup['save_energies_every_n_steps'],
                         4))  # T, KE, PE, unused
    energies.fill(np.nan)
    # determine k_B
    if 'k_B' in md_setup:
        k_B = float(md_setup['k_B'])
    else:
        k_B = 1.0
    # determine mass
    if 'm' in md_setup:
        m = float(md_setup['m'])
    else:
        m = 1.0

    for s in range(md_setup['n_steps']):
        velocity_verlet(N, md_setup['dt'], m, rx, ry, rz, vx, vy, vz, fx, fy, fz)
        wrap_successful = wrap_into_box(N, box, rx, ry, rz)
        if not wrap_successful:
            raise Exception(f"""
At step {s}: Some particles moved too fast and could not be wraped back into the box.
Time step too large? Stopping.""")
        # if no table, do LJ, else tabulated
        if u_table is None:
            PE = potential_energy_LJ(N, box, md_setup['cut_off'], rx, ry, rz,
                                     fx, fy, fz)
        else:
            PE = potential_energy_table(N, box, md_setup['cut_off'], rx, ry, rz,
                                        fx, fy, fz, u_table, fr_table, min_r, Delta_r)
        if check_arrays_for_nans(fx, fy, fz):
            raise Exception(f"""
At step {s}: Some force is NaN. Are particles overlapping? Stopping""")
        KE = kinetic_energy(N, md_setup['dt'], m, vx, vy, vz, fx, fy, fz)
        if check_arrays_for_nans(vx, vy, vz):
            raise Exception(f"""
At step {s}: Some velocity is NaN. Are particles overlapping? Stopping""")
        berendsen_thermostat(N, md_setup['dt'], md_setup['T'], md_setup['tau'], KE, k_B,
                             vx, vy, vz)
        # if we print or save energies this step -> calculate them
        if (s % md_setup['print_every_n_steps'] == 0
                or s % md_setup['save_energies_every_n_steps'] == 0):
            TE = PE + KE
            T = KE / (k_B * 3/2 * N)
        # print output
        if s % md_setup['print_every_n_steps'] == 0:
            output_thermo(s, PE, KE, TE, T)
        # save energies
        if s % md_setup['save_energies_every_n_steps'] == 0:
            s_ener = s // md_setup['save_energies_every_n_steps']
            energies[s_ener, 0] = T
            energies[s_ener, 1] = KE
            energies[s_ener, 2] = PE
        # save trajectory
        if s % md_setup['save_traj_every_n_steps'] == 0:
            s_traj = s // md_setup['save_traj_every_n_steps']
            traj[s_traj, 0, :] = rx
            traj[s_traj, 1, :] = ry
            traj[s_traj, 2, :] = rz
    return traj, energies


def gen_bcc_crystal(num, rho):
    """Initalize a box with bcc grid."""
    N = num**3 * 2  # 2 atoms per unit cell
    L = (N / rho)**(1/3)
    box = np.array([L, L, L])
    x_ = np.linspace(0., box[0], num=num, endpoint=False)
    y_ = np.linspace(0., box[1], num=num, endpoint=False)
    z_ = np.linspace(0., box[2], num=num, endpoint=False)

    rx = []
    ry = []
    rz = []
    for x in x_:
        for y in y_:
            for z in z_:
                for offset in [(0, 0, 0), (L / num / 2, L / num / 2, L / num / 2)]:
                    rx.append(x + offset[0])
                    ry.append(y + offset[1])
                    rz.append(z + offset[2])
    return box, rx, ry, rz


if __name__ == '__main__':
    box, rx, ry, rz = gen_bcc_crystal(num=6, rho=0.85)
    # MD setup
    md_setup = {
        'box': box,
        'start_r': (rx, ry, rz),
        'start_v': None,
        'dt': 0.002,
        'cut_off': 2.5,
        'n_steps': 10000,
        'T': 1.0,
        'tau': 0.1,
        'print_every_n_steps': 1000,
        'save_traj_every_n_steps': 100,
        'save_energies_every_n_steps': 1,
    }
    # run md
    t_start = time.perf_counter()
    traj, energies = mdrun(md_setup)
    np.savez_compressed('md_out.npz', traj=traj, box=md_setup['box'], energies=energies)
    t_end = time.perf_counter()
    print("Total looping time = {:.2f} seconds.".format(t_end - t_start))
