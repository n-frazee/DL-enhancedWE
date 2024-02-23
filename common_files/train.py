# get_coord.py
#
# A script to output coordinates as auxdata in WESTPA on the fly.
# Coordinates should be outputted into 'iterations/iter_{i:08d}/auxdata/coord'
# for the haMSM or other uses, under the shape (n_segs, n_frames, n_atoms, 3).
#
# Make sure you declare the dataset in west.cfg and specify `loader: npy_loader`
# under executable/dataset. You will have to run this script in runseg.sh and copy
# the output file using `cp coord.npy $WEST_COORD_RETURN`.


import mdtraj
import numpy
import os.path


# Loading the trajectories...
parent_traj = mdtraj.load("train.nc", top="hmr.prmtop")
ref_file = mdtraj.load("ref.pdb", top="hmr.prmtop")

# Start loading and stuff

backbone = ref_file.top.select('backbone')
ref_file = ref_file.atom_slice(backbone)
backbone = parent_traj.top.select('backbone')
full_traj = parent_traj.atom_slice(backbone)

full_traj = full_traj.superpose(ref_file)
all_coords = full_traj._xyz * 10

if 0:
    # Alternate loading just in case things are weird, like traj and parent don't share the same topology
    parent_traj = parent_traj.superpose(ref_file, atom_indices=ref_slice)
    seg_traj = seg_traj.superpose(ref_file, atom_indices=ref_slice)
    all_coords = numpy.squeeze(numpy.concatenate((parent_traj._xyz, seg_traj._xyz)))
    
# print(all_coords.shape)
numpy.save("train.npy", numpy.transpose(all_coords, [0, 2, 1]))
