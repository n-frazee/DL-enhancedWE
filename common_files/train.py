import mdtraj
import numpy


# Load the traj and reference files
traj_file = mdtraj.load("train.nc", top="hmr.prmtop")
ref_file = mdtraj.load("reference.pdb", top="hmr.prmtop")

# Select the CA atoms from the traj and ref
selection = ref_file.top.select('name CA')
ref_file = ref_file.atom_slice(selection)
selection = traj_file.top.select('name CA')
full_traj = traj_file.atom_slice(selection)

# Align the traj to the ref
full_traj = full_traj.superpose(ref_file)
# Get the xyz coords and multiply by 10 to convert to angstroms
all_coords = full_traj._xyz * 10
    
# Save the coords in the proper format
numpy.save("../common_files/train.npy", numpy.transpose(all_coords, [0, 2, 1]))
