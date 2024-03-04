import westpa
import logging
from sklearn import cluster
import torch
import numpy as np
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
import operator
from scipy.sparse import coo_matrix
import pickle

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


def ls_kmeans(we_driver, ibin, n_iter=0, n_clusters=5, **kwargs):
    # Sort the segments in seg_id order... just making sure.

    current_iteration = westpa.rc.get_data_manager().current_iteration
    print(f'looking at bin {ibin} for iteration {n_iter}')
    #print(we_driver.next_iter_binning[ibin])
    #print([segment for segment in we_driver.next_iter_binning[ibin]])
    bin = we_driver.next_iter_binning[ibin]

    print(f'{len(bin)} segments in this bin')
    if n_iter <= 1:
        # Don't do grouping in the first iteration/initialization, can't get the dmatrices.
        return [{i for i in bin}]
        #bin = np.array(sorted(we_driver.next_iter_binning[ibin], key=operator.attrgetter('parent_id')), dtype=np.object_)

    # Grab all the d-matrices from the last iteration (n_iter-1), will be in seg_id order by default.
    # TODO: Hope we have some way to simplify this... WESTPA 3 function...
    dcoords = np.concatenate(we_driver.get_prev_dcoords(1))

    chosen_dcoords = []
    for idx, seg in enumerate(bin):
        #print(seg.wtg_parent_ids)
        if seg.parent_id < 0:
            chosen_dcoords.append(dcoords[seg.parent_id])  # This will likely mismatch when recycling... Need a way to get the dmatrix for ibstates.
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


def kmeans(coords, n_clusters, splitting, **kwargs):
    X = np.array(coords)
    if X.shape[0] == 1:
        X = X.reshape(-1,1)
    km = cluster.KMeans(n_clusters=n_clusters).fit(X)   
    cluster_centers_indices = km.cluster_centers_
    labels = km.labels_
    if splitting:
        print("cluster centers:", np.sort(cluster_centers_indices))
    return labels


def walkers_by_history(we_driver, ibin, hist_length=25, **kwargs):
    '''Groups walkers inside of a bin according to their history.

    Creates a group, which takes the same data format as a bin, and then passes into the
    normal split/merge functions.'''
    # Pass in the bin object instead of the index
    log.debug('using group.walkers_by_history')
    log.debug('history length: {!r}'.format(hist_length))
    bin = we_driver.next_iter_binning[ibin]
    groups = dict()
    for segment in bin:
        if segment.n_iter > 1:
            ##par_iter = we_driver._find_parent_n(segment, hist_length)
            par_iter = _find_parent_n(segment, hist_length)
        else:
            par_iter = (0, 1)
        try:
            groups[par_iter].add(segment)
        except KeyError:
            groups[par_iter] = set([segment])
    return groups.values()


def _find_parent_n(segment, n):
    iteration = (segment.n_iter - 1)
    parent_id = segment.parent_id
    try:
        if (len(segment.id_hist) < n):
            segment.id_hist = [(parent_id, iteration)] + segment.id_hist
        else:
            segment.id_hist.reverse()
            segment.id_hist.append((segment.parent_id, (segment.n_iter - 1)))
            segment.id_hist.reverse()
            del segment.id_hist[n:]
        parent_id = segment.id_hist[(len(segment.id_hist) - 1)][0]
        iteration = segment.id_hist[(len(segment.id_hist) - 1)][1]
    except AttributeError:
        data_manager = westpa.rc.get_data_manager()
        i = 0
        while (i < n) and parent_id >= 0:
            seg_id = parent_id
            iter_group = data_manager.get_iter_group(iteration)
            seg_index = iter_group['seg_index']
            parent_id = seg_index[seg_id]['parent_id']
            try:
                segment.id_hist.append((parent_id, iteration))
            except AttributeError:
                segment.id_hist = list()
                segment.id_hist.append((parent_id, iteration))
            iteration -= 1
            i += 1
        iteration += 1
    return (parent_id, iteration)


def walkers_test(we_driver, ibin, **kwargs):
    log.debug('using odld_system._group_walkers_identity')
    bin_set = we_driver.next_iter_binning[ibin]

    log.debug('bin_set: {!r}'.format(bin_set)) #KFW 

    list_bins = [set()]
    log.debug('list_bins empty: {!r}'.format(list_bins)) #KFW

    for i in bin_set:

        log.debug('>>> i: {!r}'.format(i)) #KFW

        list_bins[0].add(i)

    log.debug('list_bins: {!r}'.format(list_bins)) #KFW
    return list_bins


def walkers_by_color(we_driver, ibin, states, **kwargs):
    '''Groups walkers inside of a bin according to a user defined state definition.
    Must be n-dimensional.

    Creates a group, which takes the same data format as a bin, and then passes into the
    normal split/merge functions.'''
    # Pass in the bin object instead of the index
    log.debug('using group.walkers_by_color')
    log.debug('state definitions: {!r}'.format(states))
    # Generate a dictionary which contains bin indices for the states.
    states_ibin = {}
    for i in states.keys():
        for pcoord in states[i]:
            try:
                states_ibin[i].append(we_driver.bin_mapper.assign([pcoord])[0])
            except:
                states_ibin[i] = []
                states_ibin[i].append(we_driver.bin_mapper.assign([pcoord])[0])
    for state in states_ibin:
        states_ibin[state] = list(set(states_ibin[state]))
    log.debug('state bins: {!r}'.format(states_ibin))
    bin = we_driver.next_iter_binning[ibin]
    groups = dict()
    for segment in bin:
        color = we_driver.bin_mapper.assign([segment.pcoord[0,:]])[0]
        for i in states_ibin.keys():
            if color in set(states_ibin[i]):
                segment.data['color'] = np.float64(i)
            else:
                segment.data['color'] = -1
        try:
            groups[segment.data['color']].add(segment)
        except KeyError:
            groups[segment.data['color']] = set([segment])
    return groups.values()

    
def color_data_loader(fieldname, data_filename, segment, single_point):
    '''Groups walkers inside of a bin according to a user defined state definition.
    Must be n-dimensional.

    Creates a group, which takes the same data format as a bin, and then passes into the
    normal split/merge functions.'''
    # Pass in the bin object instead of the index
    # Generate a dictionary which contains bin indices for the states.
    we_driver = westpa.rc.get_we_driver()
    states = we_driver.subgroup_function_kwargs['states']
    system = westpa.rc.get_system_driver()
    states_ibin = {}
    for i in states.keys():
        for pcoord in states[i]:
            try:
                states_ibin[i].append(system.bin_mapper.assign([pcoord])[0])
            except:
                states_ibin[i] = []
                states_ibin[i].append(system.bin_mapper.assign([pcoord])[0])
    for state in states_ibin:
        states_ibin[state] = list(set(states_ibin[state]))
    color = system.bin_mapper.assign([segment.pcoord[0,:]])[0]
    
    for i in states_ibin.keys():
        if color in set(states_ibin[i]):
            segment.data['color'] = np.float64(i)
        else:
            segment.data['color'] = -1


