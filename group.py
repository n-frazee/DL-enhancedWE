import westpa
import logging
from sklearn import cluster
import torch
import numpy as np
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
import operator
from scipy.sparse import coo_matrix
import pickle
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

def binless_ls_kmeans(coords, n_clusters, splitting, iters_back_to_cluster=10, **kwargs):

    # Sort the segments in seg_id order... just making sure.

    #print(coords.shape)
    #print(coords)

    we_driver = westpa.rc.get_we_driver()
    sim_manager = westpa.rc.get_sim_manager()

    current_iteration = westpa.rc.get_data_manager().current_iteration
    n_iter = we_driver.niter

    #print(n_iter)
    #print(we_driver.segments)

    #print(f'looking at bin {ibin} for iteration {n_iter}')
    #print(we_driver.next_iter_binning[ibin])
    #print([segment for segment in we_driver.next_iter_binning[ibin]])
    #bin = we_driver.next_iter_binning[ibin]
    n_segs = len(we_driver.segments)

    assert len(coords) == n_segs

    #print(f'{len(bin)} segments in this bin')
    #if n_iter <= 1:
    #    # Don't do grouping in the first iteration/initialization, can't get the dmatrices.
    #    return [{i for i in bin}]
        #bin = np.array(sorted(we_driver.next_iter_binning[ibin], key=operator.attrgetter('parent_id')), dtype=np.object_)

    # Grab all the d-matrices from the last iteration (n_iter-1), will be in seg_id order by default.
    # TODO: Hope we have some way to simplify this... WESTPA 3 function...

    ibstate_group = westpa.rc.get_data_manager().we_h5file['ibstates/0']

    if n_iter > 1:
        past_dcoords = []
        if iters_back_to_cluster == 0:
            past_dcoords = []
        elif n_iter <= iters_back_to_cluster:
            iters_back_to_cluster = n_iter -1
            past_dcoords = np.concatenate(we_driver.get_prev_dcoords(iters_back_to_cluster, upperbound=n_iter))

        # Get current iter dcoords
        curr_dcoords = np.concatenate(we_driver.get_prev_dcoords(1))
    else:  # building the list from scratch, during first iter
        past_dcoords, curr_dcoords = [], []
        for segment in we_driver.segments:
            istate_id = ibstate_group['istate_index']['basis_state_id', int(segment.parent_id)]
            #print(istate_id)
            auxref = int(tostr(ibstate_group['bstate_index']['auxref', istate_id]))
            #print(auxref)

            dmatrix = we_driver.synd_model.backmap([auxref], mapper='dmatrix')
            curr_dcoords.append(dmatrix[0])
            
    #print(dcoords)

    #print(len(curr_dcoords))
    #print(n_segs)

    chosen_dcoords = []
    to_pop = []
    for idx, segment in enumerate(we_driver.segments):
        #print(segment)
        #print(seg.wtg_parent_ids)
        if segment.parent_id < 0:
            istate_id = ibstate_group['istate_index']['basis_state_id', -int(segment.parent_id + 1)]
            #print(istate_id)
            auxref = int(tostr(ibstate_group['bstate_index']['auxref', istate_id]))
            #print(auxref)

            dmatrix = we_driver.synd_model.backmap([auxref], mapper='dmatrix')
            chosen_dcoords.append(dmatrix[0])
        else:
            #print(idx)
            #print(segment.parent_id)
            chosen_dcoords.append(curr_dcoords[segment.parent_id])
            to_pop.append(segment.parent_id)
    
    curr_dcoords = np.asarray(curr_dcoords)
    if len(to_pop) > 0:
        curr_dcoords = np.delete(curr_dcoords, to_pop, axis=0)
    final = [np.asarray(i) for i in [chosen_dcoords, curr_dcoords, past_dcoords] if len(i) > 0]

    #print(len(chosen_dcoords))
    #print(f'curr_dcoords shape: {curr_dcoords.shape}')
    #print(len(past_dcoords))
    #print(final)
    #print(len(final))

    #print('yay')
    #if len(final) > 1:
    #    print(f'{final[0].shape}, {final[1].shape}')

    scoords = compute_sparse_contact_map(np.vstack((final)))

    #print(len(scoords))

    X = we_driver.autoencoder.predict(scoords)

    # Perform the K-means clustering
    km = cluster.KMeans(n_clusters=n_clusters).fit(X[0])
    cluster_centers_indices = km.cluster_centers_
    labels = km.labels_

    # Log the cluster centers
    if splitting:
        westpa.rc.pstatus(f'cluster centers: {np.sort(cluster_centers_indices)}')
        westpa.rc.pflush()

    return labels

    # Make the dictionary where each key is a group.
    #groups = dict()
    #for label, segment in zip(labels, bin):
    #    try:
    #        groups[label].add(segment)
    #    except KeyError:
    #        groups[label] = set([segment])

    #print(f'Labels for the segments: {labels}')
    #print(f'clusters present: {groups.keys()}')

    # This assertion will fail if all the structures in this bin ended up in the same cluster.
    #assert len(groups.keys()) == n_clusters
    #return groups.values()

    #X = numpy.array(coords)
    #if X.shape[0] == 1:
    #    X = X.reshape(-1,1)
    #km = cluster.KMeans(n_clusters=n_clusters).fit(X)   
    #cluster_centers_indices = km.cluster_centers_
    #labels = km.labels_
    #if splitting:
    #    print("cluster centers:", numpy.sort(cluster_centers_indices))
    #return labels




def ls_kmeans(we_driver, ibin, n_iter=0, n_clusters=5, iters_back_to_cluster=10, **kwargs):

    # Sort the segments in seg_id order... just making sure.

    current_iteration = westpa.rc.get_data_manager().current_iteration
    print(f'looking at bin {ibin} for iteration {n_iter}')
    #print(we_driver.next_iter_binning[ibin])
    #print([segment for segment in we_driver.next_iter_binning[ibin]])
    bin = we_driver.next_iter_binning[ibin]
    #n_segs = len(bin)

    # print(f'{len(bin)} segments in this bin')
    #if n_iter <= 1:
    #    # Don't do grouping in the first iteration/initialization, can't get the dmatrices.
    #    return [{i for i in bin}]
        #bin = np.array(sorted(we_driver.next_iter_binning[ibin], key=operator.attrgetter('parent_id')), dtype=np.object_)

    # Grab all the d-matrices from the last iteration (n_iter-1), will be in seg_id order by default.
    # TODO: Hope we have some way to simplify this... WESTPA 3 function...

    # ibstate_group = westpa.rc.get_data_manager().we_h5file['ibstates/0']

    # if n_iter > 1:
    #     past_dcoords = []
    #     if iters_back_to_cluster == 0:
    #         past_dcoords = []
    #     elif n_iter <= iters_back_to_cluster:
    #         iters_back_to_cluster = n_iter -1
    #         past_dcoords = np.concatenate(we_driver.get_prev_dcoords(iters_back_to_cluster, upperbound=n_iter))

    #     # Get current iter dcoords
    #     curr_dcoords = np.concatenate(we_driver.get_prev_dcoords(1))
    # else:  # building the list from scratch, during first iter
    #     past_dcoords, curr_dcoords = [], []
    #     for segment in bin:
    #         istate_id = ibstate_group['istate_index']['basis_state_id', int(segment.parent_id)]
    #         #print(istate_id)
    #         auxref = int(tostr(ibstate_group['bstate_index']['auxref', istate_id]))
    #         #print(auxref)

    #         dmatrix = we_driver.synd_model.backmap([auxref], mapper='dmatrix')
    #         curr_dcoords.append(dmatrix[0])
            
    #print(dcoords)

    #print(len(curr_dcoords))
    #print(n_segs)

    # chosen_dcoords = []
    # to_pop = []
    # for idx, segment in enumerate(bin):
    #     #print(segment)
    #     #print(seg.wtg_parent_ids)
    #     if segment.parent_id < 0:
    #         istate_id = ibstate_group['istate_index']['basis_state_id', -int(segment.parent_id + 1)]
    #         #print(istate_id)
    #         auxref = int(tostr(ibstate_group['bstate_index']['auxref', istate_id]))
    #         #print(auxref)

    #         dmatrix = we_driver.synd_model.backmap([auxref], mapper='dmatrix')
    #         chosen_dcoords.append(dmatrix[0])
    #     else:
    #         #print(idx)
    #         #print(segment.parent_id)
    #         chosen_dcoords.append(curr_dcoords[segment.parent_id])
    #         to_pop.append(segment.parent_id)
    # 
    # curr_dcoords = np.asarray(curr_dcoords)
    # if len(to_pop) > 0:
    #     curr_dcoords = np.delete(curr_dcoords, to_pop, axis=0)
    # final = [np.asarray(i) for i in [chosen_dcoords, curr_dcoords, past_dcoords] if len(i) > 0]

    #print(len(chosen_dcoords))
    #print(f'curr_dcoords shape: {curr_dcoords.shape}')
    #print(len(past_dcoords))
    #print(final)
    #print(len(final))

    #print('yay')
    #if len(final) > 1:
    #    print(f'{final[0].shape}, {final[1].shape}')

    #scoords = compute_sparse_contact_map(np.vstack((final)))

    #print(len(scoords))

    #X = we_driver.autoencoder.predict(scoords)

    # Perform the K-means clustering
    if len(bin) < n_clusters:
        n_clusters = len(bin)

    labels = we_driver.rng.choice(list(range(n_clusters)), len(bin))
    #km = cluster.KMeans(n_clusters=n_clusters).fit(X[0])
    #cluster_centers_indices = km.cluster_centers_
    #labels = km.labels_

    # Log the cluster centers
    #westpa.rc.pstatus(f'cluster centers: {np.sort(cluster_centers_indices)}')
    #westpa.rc.pflush()

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

