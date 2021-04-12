# Mind
A Minimal Lennard-Jones Fluid Molecular Dynamics Python Program

## Usage

Simplely,

```bash
$ ./mind.py
```

Alternatively, define a setup in Python and run it there:

```Python
L = 6.0
box = np.array([L, L, L])
x_ = np.linspace(0., box[0], num=6, endpoint=False)
y_ = np.linspace(0., box[1], num=6, endpoint=False)
z_ = np.linspace(0., box[2], num=6, endpoint=False)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
rx = x.flatten()
ry = y.flatten()
rz = z.flatten()

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

t_start = time.perf_counter()
traj, energies = mdrun(md_setup)
np.savez_compressed('md_out.npz', traj=traj, energies=energies)
t_end = time.perf_counter()
print("Total looping time = {:.2f} seconds.".format(t_end - t_start))
```

## Requirements

- Numpy
- Numba

## License

All code is licensed under the GPL, v3 or later. See LICENSE file for details.
