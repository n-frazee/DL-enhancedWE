import numpy as np
import pickle
from synd.core import load_model
import MDAnalysis as md
from MDAnalysis.coordinates.memory import MemoryReader


model = load_model('synd_model/ntl9_folding.synd')

model.rng = np.random.default_rng()
print(model.rng.random())

restart = np.loadtxt('parent.txt', dtype=int)

discrete_trajectory = model.generate_trajectory(
    initial_states=np.array([121]),
    # 1 step is 10 ps
    n_steps=10
)

print(f'discrete: {discrete_trajectory}')


with open('synd_model/coord_map.pkl', 'rb') as infile:
    coord_map = pickle.load(infile)

model.add_backmapper(
    coord_map.get, 
    name='coord'
)

atomistic_trajectory = model.backmap(discrete_trajectory[:,:], 'coord') * 10

#print(f'atomistic: {atomistic_trajectory}')

u = md.Universe('synd_model/ntl9.pdb')
u.load_new(atomistic_trajectory[0], format=MemoryReader)
u.select_atoms('all').write('seg.nc', frames='all')

np.savetxt('seg.txt', discrete_trajectory)