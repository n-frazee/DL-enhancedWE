import westpa
import logging
from sklearn import cluster
import torch
import numpy as np
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
import operator
from scipy.sparse import coo_matrix
import pickle
import synd
from westpa.core.h5io import tostr
from os.path import expandvars


log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)


def compute_sparse_contact_map(coords: np.ndarray) -> np.ndarray:
    """Compute the sparse contact maps for a set of coordinate frames."""
    # Compute a distance matrix for each frame
    #distance_matrices = [distance_matrix(frame, frame) for frame in coords]
    # Convert the distance matrices to contact maps (binary matrices of 0s and 1s)
    #contact_maps = np.array(distance_matrices) < self.cfg.contact_cutoff
    # Convert the contact maps to sparse matrices for memory efficiency (using COO format)
    coo_contact_maps = [coo_matrix(contact_map) for contact_map in coords]
    # Collect the rows and cols of the COO matrices (since the values are all 1s, we don't need them)
    rows = [coo.row.astype("int16") for coo in coo_contact_maps]
    cols = [coo.col.astype("int16") for coo in coo_contact_maps]
    # Concatenate the rows and cols into a single array representing the contact maps
    return [np.concatenate(row_col) for row_col in zip(rows, cols)]


def load_synd_model(synd_model_path, backmap_path):
    synd_model = synd.core.load_model(synd_model_path)

    with open(backmap_path, 'rb') as infile:
        dmatrix_map = pickle.load(infile)

    synd_model.add_backmapper(dmatrix_map.get, name='dmatrix')

    return synd_model


def ls_kmeans(we_driver, ibin, n_iter=0, n_clusters=5, **kwargs):
    # Sort the segments in seg_id order... just making sure.

    current_iteration = westpa.rc.get_data_manager().current_iteration
    print(f'looking at bin {ibin} for iteration {n_iter}')
    #print(we_driver.next_iter_binning[ibin])
    #print([segment for segment in we_driver.next_iter_binning[ibin]])
    bin = we_driver.next_iter_binning[ibin]

    print(f'{len(bin)} segments in this bin')
    if n_iter <= 1:
    #    # Don't do grouping in the first iteration/initialization, can't get the dmatrices.
        return [{i for i in bin}]
        #bin = np.array(sorted(we_driver.next_iter_binning[ibin], key=operator.attrgetter('parent_id')), dtype=np.object_)

    # Grab all the d-matrices from the last iteration (n_iter-1), will be in seg_id order by default.
    # TODO: Hope we have some way to simplify this... WESTPA 3 function...

    dcoords = np.concatenate(we_driver.get_prev_dcoords(1))
    ibstate_group = westpa.rc.get_data_manager().get_iter_group(n_iter-1)['ibstates']

    #print(ibstate_group)
    synd_model_path = expandvars(kwargs['synd_model'])
    backmap_path = expandvars(kwargs['dmatrix_map'])
    
    chosen_dcoords = []
    for idx, seg in enumerate(bin):
        #print(seg.wtg_parent_ids)
        if seg.parent_id < 0:
            istate_id = ibstate_group['istate_index']['basis_state_id', -int(seg.parent_id + 1)]
            #print(istate_id)
            auxref = int(tostr(ibstate_group['bstate_index']['auxref', istate_id]))
            #print(auxref)
            try:
                dmatrix = synd_model.backmap([auxref])
            except UnboundLocalError:
                synd_model = load_synd_model(synd_model_path, backmap_path)
                dmatrix = synd_model.backmap([auxref])

            chosen_dcoords.append(dmatrix)  # This will likely mismatch when recycling... Need a way to get the dmatrix for ibstates.
        else:
            chosen_dcoords.append(dcoords[seg.parent_id]) 

    scoords = compute_sparse_contact_map(chosen_dcoords)

    X = we_driver.autoencoder.predict(scoords)

    # Perform the K-means clustering
    if len(bin) < n_clusters:
        n_clusters = len(bin)

    km = cluster.KMeans(n_clusters=n_clusters).fit(X[0])
    cluster_centers_indices = km.cluster_centers_
    labels = km.labels_

    # Log the cluster centers
    westpa.rc.pstatus(f'cluster centers: {np.sort(cluster_centers_indices)}')
    westpa.rc.pflush()

    # Make the dictionary where each key is a group.
    groups = dict()
    for label, segment in zip(labels, bin):
        try:
            groups[label].add(segment)
        except KeyError:
            groups[label] = set([segment])

    print(f'Labels for the segments: {labels}')
    print(f'clusters present: {groups.keys()}')

    # This assertion will fail if all the structures in this bin ended up in the same cluster.
    #assert len(groups.keys()) == n_clusters
    return groups.values()

