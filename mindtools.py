import mdtraj as md
import nglview as nv
import numpy as np


def create_mdtraj_traj(traj, box):
    """Convert Numpy to MDTraj trajectory.

    All atoms will be argon atoms.

    Keyword arguments:
    traj -- Numpy array with shape (n_steps, 3, n_atoms) containing positions in nm
    box -- Numpy array with shape (3) in nm

    Returns:
    mdtraj.Trajectory
    """
    # create an argon topology
    top = md.Topology()
    chain = top.add_chain()
    for i in range(traj.shape[2]):
        residue = top.add_residue('AR', chain)
        top.add_atom('Ar', md.element.argon, residue)
    # create trajectory
    n_frames = len(traj)
    t = md.Trajectory(traj.swapaxes(1, 2), topology=top,
                      unitcell_lengths=np.repeat(box[np.newaxis, ...], n_frames,
                                                 axis=0),
                      unitcell_angles=np.ones((n_frames, 3))*90.0)
    return t


def show_mdtraj_trajectory(traj, aspect_ratio=5):
    """Show MDTraj trajectory with constant box.

    All atoms will be shown as argon atoms.

    Keyword arguments:
    traj -- MDTraj trajectory
    aspect_ratio -- parameter chaning the sphere size

    Returns:
    nglview.widget.NGLWidget
    """
    # create view and show trajectory
    view = nv.show.show_mdtraj(traj)
    view.clear_representations()
    view.add_representation('spacefill')
    view.add_unitcell()
    view.gui_style = 'ngl'
    return view
