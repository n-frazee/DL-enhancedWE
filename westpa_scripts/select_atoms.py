import loos
import loos.pyloos
import numpy as np
import sys

system_file = sys.argv[1]
traj_file = sys.argv[2]

system = loos.createSystem(system_file)
traj = loos.pyloos.Trajectory(traj_file, system)

sel = loos.selectAtoms(system, sel_string)

coords = np.zeros((len(traj), len(sel), 3))
frame_num = 0
for frame in traj:
    atom_num = 0
    for atom in sel:
        coords[frame_num][atom_num] = [x for x in atom.coords()]
        atom_num += 1
    frame_num += 1

np.savetxt("coords.txt", np.ravel(coords))
