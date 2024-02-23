import mdtraj as md
import numpy as np
from itertools import combinations
import os
import sys
# ref = md.load("ref.pdb")

# heavy = ref.topology.select_atom_indices("heavy")
# heavy_pairs = np.array(
#     [
#         (i, j)
#         for (i, j) in combinations(heavy, 2)
#         if abs(ref.topology.atom(i).residue.index - ref.topology.atom(j).residue.index)
#         > 3
#     ]
# )

# r0 = md.compute_distances(ref, heavy_pairs)[0]
# signature = heavy_pairs[r0 < 0.45]
# r0 = r0[r0 < 0.45]

# np.save("signature_ref.npy", signature)
# np.save("r0_ref.npy", r0)


def _best_hummer_q(trajectory, signature, r0):
    r = md.compute_distances(trajectory, signature)
    q = np.mean(1.0 / (1 + np.exp(50. * (r - 1.8 * r0))), axis=1)
    return q

signature = np.load(os.path.expandvars("$WEST_SIM_ROOT/common_files/signature_ref.npy"))  # This is just a list of atom pairs of what distances to look at 
r0 = np.load(os.path.expandvars('$WEST_SIM_ROOT/common_files/r0_ref.npy'))  # This is just the distances of signature atoms in the reference structure

topology_path = os.path.expandvars("$WEST_SIM_ROOT/common_files/hmr.prmtop")
traj_path = os.path.expandvars("$WEST_CURRENT_SEG_DATA_REF/seg.nc")
parent_path = os.path.expandvars("$WEST_CURRENT_SEG_DATA_REF/parent.ncrst")

if os.path.isfile(traj_path):
    parent_traj = md.load(parent_path, top=topology_path)
    seg_traj = md.load(traj_path, top=topology_path)
    full_traj = parent_traj.join(seg_traj)
else:
    traj_path = os.path.expandvars("$WEST_STRUCT_DATA_REF")
    full_traj = md.load(traj_path, top=topology_path)
    
np.savetxt(sys.argv[1], _best_hummer_q(full_traj, signature, r0))
def inA(trajectory):
    return _best_hummer_q(trajectory, signature, r0) >= .99

def inB(trajectory):
    return _best_hummer_q(trajectory, signature, r0) <= .01
