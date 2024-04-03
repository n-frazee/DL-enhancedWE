import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import warnings
import MDAnalysis as mda
import subprocess
import os

def calculate_counters(c_total, n_objects, c_threshold=None, w_factor="fraction"):
    """Calculate 1-similarity, 0-similarity, and dissimilarity counters

    Parameters
    ---------
    c_total : array-like of shape (n_objects, n_features)
        Vector containing the sums of each column of the fingerprint matrix.

    n_objects : int
        Number of objects to be compared.

    c_threshold : {None, 'dissimilar', int}
        Coincidence threshold.
        None : Default, c_threshold = n_objects % 2
        'dissimilar' : c_threshold = np.ceil(n_objects / 2)
        int : Integer number < n_objects
        float : Real number in the (0 , 1) interval. Indicates the % of the total data that will serve as threshold.

    w_factor : {"fraction", "power_n"}
        Type of weight function that will be used.
        'fraction' : similarity = d[k]/n
                     dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
        'power_n' : similarity = n**-(n_objects - d[k])
                    dissimilarity = n**-(d[k] - n_objects % 2)
        other values : similarity = dissimilarity = 1

    Returns
    -------
    counters : dict
        Dictionary with the weighted and non-weighted counters.
    """
    # Assign c_threshold
    if not c_threshold:
        c_threshold = n_objects % 2
    if c_threshold == 'dissimilar':
        c_threshold = np.ceil(n_objects / 2)
    if c_threshold == 'min':
        c_threshold = n_objects % 2
    if isinstance(c_threshold, int):
        if c_threshold >= n_objects:
            raise ValueError("c_threshold cannot be equal or greater than n_objects.")
        c_threshold = c_threshold
    if 0 < c_threshold < 1:
        c_threshold *= n_objects

    # Set w_factor
    if w_factor:
        if "power" in w_factor:
            power = int(w_factor.split("_")[-1])
            def f_s(d):
                return power**-float(n_objects - d)

            def f_d(d):
                return power**-float(d - n_objects % 2)
        elif w_factor == "fraction":
            def f_s(d):
                return d/n_objects

            def f_d(d):
                return 1 - (d - n_objects % 2)/n_objects
        else:
            def f_s(d):
                return 1

            def f_d(d):
                return 1
    else:
        def f_s(d):
            return 1

        def f_d(d):
            return 1

    # Calculate a, d, b + c

    a_indices = 2 * c_total - n_objects > c_threshold
    d_indices = n_objects - 2 * c_total > c_threshold
    dis_indices = np.abs(2 * c_total - n_objects) <= c_threshold

    a = np.sum(a_indices)
    d = np.sum(d_indices)
    total_dis = np.sum(dis_indices)

    a_w_array = f_s(2 * c_total[a_indices] - n_objects)
    d_w_array = f_s(abs(2 * c_total[d_indices] - n_objects))
    total_w_dis_array = f_d(abs(2 * c_total[dis_indices] - n_objects))

    w_a = np.sum(a_w_array)
    w_d = np.sum(d_w_array)
    total_w_dis = np.sum(total_w_dis_array)

    total_sim = a + d
    total_w_sim = w_a + w_d
    p = total_sim + total_dis
    w_p = total_w_sim + total_w_dis

    counters = {"a": a, "w_a": w_a, "d": d, "w_d": w_d,
                "total_sim": total_sim, "total_w_sim": total_w_sim,
                "total_dis": total_dis, "total_w_dis": total_w_dis,
                "p": p, "w_p": w_p}
    return counters


def gen_traj_numpy(prmtopFileName, trajFileName, atomSel):
    """Reads in a trajectory and returns a 2D numpy array of the coordinates 
    of the selected atoms.
    
    Parameters
    ----------
    prmtopFileName : str
        The file path of the topology file.
    trajFileName : str
        The file path of the trajectory file.
    atomSel : str
        The atom selection string. For example, 'resid 3:12 and name N H CA C O'.
        View details in the MDAnalysis documentation: 
        https://docs.mdanalysis.org/stable/documentation_pages/selections.html

    Returns
    -------
    traj_numpy : np.ndarray
        The 2D numpy array of the coordinates of the selected atoms.
        
    Examples
    --------
    >>> traj_numpy = gen_traj_numpy('aligned_tau.pdb', 'aligned_tau.dcd', 
                                    'resid 3:12 and name N CA C')
    """
    coord = mda.Universe(prmtopFileName,trajFileName)
    print('Number of atoms in trajectory:', coord.atoms.n_atoms)
    print('Number of frames in trajectory:', coord.trajectory.n_frames)
    atomSel = coord.select_atoms(atomSel)
    print('Number of atoms in selection:', atomSel.n_atoms)
    # Create traj data of the atom selection
    traj_numpy = np.empty((coord.trajectory.n_frames,atomSel.n_atoms, 3), dtype=float)
    # Loop every frame and store the coordinates of the atom selection
    for ts in coord.trajectory:
        traj_numpy[ts.frame,:] = atomSel.positions
    # Flatten 3D array to 2D array
    traj_numpy = traj_numpy.reshape(traj_numpy.shape[0],-1)
    return traj_numpy


def gen_sim_dict(c_total, n_objects, c_threshold=None, w_factor="fraction"):
    """Generate a dictionary with the similarity indices
    
    Parameters
    ----------
    c_total : array-like of shape (n_objects, n_features)
        Vector containing the sums of each column of the fingerprint matrix.
    n_objects : int
        Number of objects to be compared.
    c_threshold : {None, 'dissimilar', int}
        Coincidence threshold.
    w_factor : {"fraction", "power_n"}
        Type of weight function that will be used.
    
    Returns
    -------
    dict
        Dictionary with the similarity indices.
    
    Notes
    -----
    Available indices:
    BUB: Baroni-Urbani-Buser, Fai: Faith, Gle: Gleason, Ja: Jaccard,
    JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
    SM: Sokal-Michener, SSn: Sokal-Sneath n
    """
    
    counters = calculate_counters(c_total, n_objects, c_threshold=c_threshold, 
                                  w_factor=w_factor)
    bub_nw = ((counters['w_a'] * counters['w_d']) ** 0.5 + counters['w_a'])/\
             ((counters['a'] * counters['d']) ** 0.5 + counters['a'] + counters['total_dis'])
    fai_nw = (counters['w_a'] + 0.5 * counters['w_d'])/\
             (counters['p'])
    gle_nw = (2 * counters['w_a'])/\
             (2 * counters['a'] + counters['total_dis'])
    ja_nw = (3 * counters['w_a'])/\
            (3 * counters['a'] + counters['total_dis'])
    jt_nw = (counters['w_a'])/\
            (counters['a'] + counters['total_dis'])
    rt_nw = (counters['total_w_sim'])/\
            (counters['p'] + counters['total_dis'])
    rr_nw = (counters['w_a'])/\
            (counters['p'])
    sm_nw = (counters['total_w_sim'])/\
            (counters['p'])
    ss1_nw = (counters['w_a'])/\
             (counters['a'] + 2 * counters['total_dis'])
    ss2_nw = (2 * counters['total_w_sim'])/\
             (counters['p'] + counters['total_sim'])

    Indices = {'BUB':bub_nw, 'Fai':fai_nw, 'Gle':gle_nw, 'Ja':ja_nw, 'JT':jt_nw, 
               'RT':rt_nw, 'RR':rr_nw, 'SM':sm_nw, 'SS1':ss1_nw, 'SS2':ss2_nw}
    return Indices


def mean_sq_dev(matrix, N_atoms):
    """O(N) Mean square deviation (MSD) calculation for n-ary objects.
    
    Parameters
    ----------
    matrix : array-like of shape (n_samples, n_features)
        Data matrix.
    N_atoms : int
        Number of atoms in the system.
    
    Returns
    -------
    float
        normalized MSD value.
    """
    N = len(matrix)
    sq_data = matrix ** 2
    c_sum = np.sum(matrix, axis=0)
    sq_sum = np.sum(sq_data, axis=0)
    msd = np.sum(2 * (N * sq_sum - c_sum ** 2)) / (N ** 2)
    norm_msd = msd / N_atoms
    return norm_msd


def msd_condensed(c_sum, sq_sum, N, N_atoms):
    """Condensed version of 'mean_sq_dev'.

    Parameters
    ----------
    c_sum : array-like of shape (n_features,)
        Column sum of the data. 
    sq_sum : array-like of shape (n_features,)
        Column sum of the squared data.
    N : int
        Number of data points.
    N_atoms : int
        Number of atoms in the system.
    
    Returns
    -------
    float
        normalized MSD value.
    """
    msd = np.sum(2 * (N * sq_sum - c_sum ** 2)) / (N ** 2)
    norm_msd = msd / N_atoms
    return norm_msd


def extended_comparison(matrix, data_type='full', metric='MSD', N=None, N_atoms=1, 
                        **kwargs):
    """Calculate the extended comparison of a dataset. 
    
    Parameters
    ----------
    matrix : {array-like of shape (n_samples, n_features), tuple/list of length 1 or 2}
        Input data matrix.
        For 'full', use numpy.ndarray of shape (n_samples, n_features).
        For 'condensed', use tuple/list of length 1 (c_sum) or 2 (c_sum, sq_sum).
    data_type : {'full', 'condensed'}, optional
        Type of data inputted. Defaults to 'full'.
        Options:
            - 'full': Use numpy.ndarray of shape (n_samples, n_features).
            - 'condensed': Use tuple/list of length 1 (c_sum) or 2 (c_sum, sq_sum).
    metric : {'MSD', 'BUB', 'Fai', 'Gle', 'Ja', 'JT', 'RT', 'RR', 'SM', 'SS1', 'SS2'}
        Metric to use for the extended comparison. Defaults to 'MSD'.
        Available metrics:
        Mean square deviation (MSD), Bhattacharyya's U coefficient (BUB),
        Faiman's coefficient (Fai), Gleason's coefficient (Gle),
        Jaccard's coefficient (Ja), Jaccard-Tanimoto coefficient (JT),
        Rogers-Tanimoto coefficient (RT), Russell-Rao coefficient (RR),
        Simpson's coefficient (SM), Sokal-Sneath 1 coefficient (SS1),
        Sokal-Sneath 2 coefficient (SS2).
    N : int, optional
        Number of data points. Defaults to None.
    N_atoms : int, optional
        Number of atoms in the system. Defaults to 1.
    **kwargs
        c_threshold : int, optional
            Coincidence threshold. Defaults to None.
        w_factor : {'fraction', 'power_n'}, optional
            Type of weight function that will be used. Defaults to 'fraction'.
            See `esim_modules.calculate_counters` for more information.
    
    Raises
    ------
    TypeError
        If data is not a numpy.ndarray or tuple/list of length 2.
    
    Returns
    -------
    float
        Extended comparison value.
    """
    if data_type == 'full':
        if not isinstance(matrix, np.ndarray):
            raise TypeError('data must be a numpy.ndarray')
        c_sum = np.sum(matrix, axis=0)
        if not N:
            N = len(matrix)
        if metric == 'MSD':
            sq_data = matrix ** 2
            sq_sum = np.sum(sq_data, axis=0)
        
    elif data_type == 'condensed':
        if not isinstance(matrix, (tuple, list)):
            raise TypeError('data must be a tuple or list of length 1 or 2')
        c_sum = matrix[0]
        if metric == 'MSD':
            sq_sum = matrix[1]
    if metric == 'MSD':
        return msd_condensed(c_sum, sq_sum, N, N_atoms)
    else:
            if 'c_threshold' in kwargs:
                c_threshold = kwargs['c_threshold']
            else:
                c_threshold = None
            if 'w_factor' in kwargs:
                w_factor = kwargs['w_factor']
            else:
                w_factor = 'fraction'
            esim_dict = gen_sim_dict(c_sum, n_objects=N, c_threshold=c_threshold, w_factor=w_factor)
            
            return 1 - esim_dict[metric]


def calculate_comp_sim(matrix, metric, N_atoms=1):
    """Complementary similarity is calculates the similarity of a set 
    without one object or observation using metrics in the extended comparison.
    The greater the complementary similarity, the more representative the object is.
    
    Parameters
    ----------
    matrix : array-like
        Data matrix.
    metric : {'MSD', 'RR', 'JT', 'SM', etc}
        Metric used for extended comparisons. See `extended_comparison` for details.
    N_atoms : int, optional
        Number of atoms in the system. Defaults to 1.
    
    Returns
    -------
    numpy.ndarray
        Array of complementary similarities for each object.
    """
    if metric == 'MSD' and N_atoms == 1:
        warnings.warn('N_atoms is being specified as 1. Please change if N_atoms is not 1.')
    N = len(matrix)
    sq_data_total = matrix ** 2
    c_sum_total = np.sum(matrix, axis = 0)
    sq_sum_total = np.sum(sq_data_total, axis=0)
    comp_sims = np.zeros((len(matrix), 2))
    for i, object in enumerate(matrix):
        object_square = object ** 2
        value = extended_comparison([c_sum_total - object, sq_sum_total - object_square],
                                    data_type='condensed', metric=metric, 
                                    N=N - 1, N_atoms=N_atoms)
        comp_sims[i] = [i, value]
    comp_sims = np.asarray(comp_sims)
    return comp_sims


def calculate_medoid(matrix, metric, N_atoms=1):
    """Calculates the medoid of a dataset using the metrics in extended comparison.
    Medoid is the most representative object of a set.

    Parameters
    ----------
    matrix : array-like
        Data matrix.
    metric : {'MSD', 'RR', 'JT', 'SM', etc}
        Metric used for extended comparisons. See `extended_comparison` for details.
    N_atoms : int, optional
        Number of atoms in the system. Defaults to 1.
    
    Returns
    -------
    int
        The index of the medoid in the dataset.
    """
    if metric == 'MSD' and N_atoms == 1:
        warnings.warn('N_atoms is being specified as 1. Please change if N_atoms is not 1.')
    N = len(matrix)
    sq_data_total = matrix ** 2
    c_sum_total = np.sum(matrix, axis=0)
    sq_sum_total = np.sum(sq_data_total, axis=0)  
    index = len(matrix) + 1
    max_dissim = -1
    for i, object in enumerate(matrix):
        object_square = object ** 2
        value = extended_comparison([c_sum_total - object, sq_sum_total - object_square],
                                    data_type='condensed', metric=metric, 
                                    N=N - 1, N_atoms=N_atoms)
        if value > max_dissim:
            max_dissim = value
            index = i
        else:
            pass
    return index


def calculate_outlier(matrix, metric, N_atoms=1):
    """Calculates the outliers of a dataset using the metrics in extended comparison.
    Outliers are the least representative objects of a set.

    Parameters
    ----------
    matrix : array-like
        Data matrix.
    metric : {'MSD', 'RR', 'JT', 'SM', etc}
        Metric used for extended comparisons. See `extended_comparison` for details.
    N_atoms : int, optional
        Number of atoms in the system. Defaults to 1.
    
    Returns
    -------
    int
        The index of the outlier in the dataset.
    """
    if metric == 'MSD' and N_atoms == 1:
        warnings.warn('N_atoms is being specified as 1. Please change if N_atoms is not 1.')
    N = len(matrix)
    sq_data_total = matrix ** 2
    c_sum_total = np.sum(matrix, axis=0)
    sq_sum_total = np.sum(sq_data_total, axis=0)  
    index = len(matrix) + 1
    min_dissim = np.Inf
    for i, object in enumerate(matrix):
        object_square = object ** 2
        value = extended_comparison([c_sum_total - object, sq_sum_total - object_square],
                                    data_type='condensed', metric=metric, 
                                    N=N - 1, N_atoms=N_atoms)
        if value < min_dissim:
            min_dissim = value
            index = i
        else:
            pass
    return index


def trim_outliers(matrix, n_trimmed, metric, N_atoms, criterion='comp_sim'):
    """Trims a desired percentage of outliers (most dissimilar) from the dataset 
    by calculating largest complement similarity.

    Parameters
    ----------
    matrix : array-like
        Data matrix.
    n_trimmed : float or int
        The desired fraction of outliers to be removed or the number of outliers to be removed.
        float : Fraction of outliers to be removed.
        int : Number of outliers to be removed.
    metric : {'MSD', 'RR', 'JT', 'SM', etc}
        Metric used for extended comparisons. See `extended_comparison` for details.
    N_atoms : int
        Number of atoms in the system.
    criterion : {'comp_sim', 'sim_to_medoid'}, optional
        Criterion to use for data trimming. Defaults to 'comp_sim'.
        'comp_sim' removes the most dissimilar objects based on the complement similarity.
        'sim_to_medoid' removes the most dissimilar objects based on the similarity to the medoid.
        
    Returns
    -------
    numpy.ndarray
        A ndarray with desired fraction of outliers removed.
    
    Notes
    -----
    If the criterion is 'comp_sim', the lowest indices are removed because they are the most outlier.
    However, if the criterion is 'sim_to_medoid', the highest indices are removed because they are farthest from the medoid.
    """
    N = len(matrix)
    if isinstance(n_trimmed, int):
        cutoff = n_trimmed
    elif 0 < n_trimmed < 1:
        cutoff = int(np.floor(N * float(n_trimmed)))
    if criterion == 'comp_sim':
        c_sum = np.sum(matrix, axis = 0)
        sq_sum_total = np.sum(matrix ** 2, axis=0)
        comp_sims = []
        for i, row in enumerate(matrix):
            c = c_sum - row
            sq = sq_sum_total - row ** 2
            value = extended_comparison([c, sq], data_type='condensed', metric=metric, 
                                        N=N - 1, N_atoms=N_atoms)
            comp_sims.append((i, value))
        comp_sims = np.array(comp_sims)
        lowest_indices = np.argsort(comp_sims[:, 1])[:cutoff]
        matrix = np.delete(matrix, lowest_indices, axis=0)
    elif criterion == 'sim_to_medoid':
        medoid_index = calculate_medoid(matrix, metric, N_atoms=N_atoms)
        medoid = matrix[medoid_index]
        np.delete(matrix, medoid_index, axis=0)
        values = []
        for i, frame in enumerate(matrix):
            value = extended_comparison(np.array([frame, medoid]), data_type='full', 
                                        metric=metric, N_atoms=N_atoms)
            values.append((i, value))
        values = np.array(values)
        highest_indices = np.argsort(values[:, 1])[-cutoff:]
        matrix = np.delete(matrix, highest_indices, axis=0)
    return matrix


def diversity_selection(matrix, percentage: int, metric, start='medoid', N_atoms=1):
    """Selects a diverse subset of the data using the complementary similarity.
    
    Parameters
    ----------
    matrix : array-like
        Data matrix.
    percentage : int
        Percentage of the data to select.
    metric : {'MSD', 'RR', 'JT', 'SM', etc}
        Metric used for extended comparisons. See `extended_comparison` for details.
    start : {'medoid', 'outlier', 'random', list}, optional
        Seed of diversity selection. Defaults to 'medoid'.
    N_atoms : int, optional
        Number of atoms in the system. Defaults to 1.
    
    Returns
    -------
    list
        List of indices of the selected data.
    """
    n_total = len(matrix)
    total_indices = np.array(range(n_total))
    if start =='medoid':
        seed = calculate_medoid(matrix, metric=metric, N_atoms=N_atoms)
        selected_n = [seed]
    elif start == 'outlier':
        seed = calculate_outlier(matrix, metric=metric, N_atoms=N_atoms)
        selected_n = [seed]
    elif start == 'random':
        seed = np.random.default_rng().integers(0, n_total - 1)
        selected_n = [seed]
    elif isinstance(start, list):
        selected_n = start
    else:
        raise ValueError('Select a correct starting point: medoid, outlier, random or outlier')

    n = len(selected_n)
    n_max = int(np.floor(n_total * percentage / 100))
    if n_max > n_total:
        raise ValueError('Percentage is too high')
    selection = [matrix[i] for i in selected_n] 
    selection = np.array(selection)
    selected_condensed = np.sum(selection, axis=0)
    if metric == 'MSD':
        sq_selection = selection ** 2
        sq_selected_condensed = np.sum(sq_selection, axis=0)
    
    while len(selected_n) < n_max:
        select_from_n = np.delete(total_indices, selected_n)
        if metric == 'MSD':
            new_index_n = get_new_index_n(matrix, metric=metric, selected_condensed=selected_condensed,
                                          sq_selected_condensed=sq_selected_condensed, n=n, 
                                          select_from_n=select_from_n, N_atoms=N_atoms)
            sq_selected_condensed += matrix[new_index_n] ** 2
        else:
            new_index_n = get_new_index_n(matrix, metric=metric, selected_condensed=selected_condensed, 
                                          n=n, select_from_n=select_from_n)
        selected_condensed += matrix[new_index_n]
        selected_n.append(new_index_n)
        n = len(selected_n)
    return selected_n


def get_new_index_n(matrix, metric, selected_condensed, n, select_from_n, **kwargs):
    """Function to get the new index to add to the selected indices
    
    Parameters
    ----------
    matrix : array-like
        Data matrix.
    metric : {'MSD', 'RR', 'JT', 'SM', etc}
        Metric used for extended comparisons. See `extended_comparison` for details.
    selected_condensed : array-like
        Condensed sum of the selected fingerprints.
    n : int
        Number of selected objects.
    select_from_n : 1D array-like
        Indices of the objects to select from.
    **kwarg
        sq_selected_condensed : array-like, optional
            Condensed sum of the squared selected fingerprints. Defaults to None.
        N_atoms : int, optional
            Number of atoms in the system. Defaults to 1.
    
    Returns
    -------
    int
        index of the new fingerprint to add to the selected indices.
    """
    if 'sq_selected_condensed' in kwargs:
        sq_selected_condensed = kwargs['sq_selected_condensed']
    if 'N_atoms' in kwargs:
        N_atoms = kwargs['N_atoms']
    # Number of fingerprints already selected and the new one to add
    n_total = n + 1
    # Placeholders values
    min_value = -1
    index = len(matrix) + 1
    # Calculate MSD for each unselected object and select the index with the highest value.
    for i in select_from_n:
        if metric == 'MSD':
            sim_index = extended_comparison([selected_condensed + matrix[i], sq_selected_condensed + (matrix[i] ** 2)],
                                            data_type='condensed', metric=metric, N=n_total, N_atoms=N_atoms) 
        else:
            sim_index = extended_comparison([selected_condensed + matrix[i]], data_type='condensed', 
                                            metric=metric, N=n_total)
        if sim_index > min_value:
            min_value = sim_index
            index = i
        else:
            pass
    return index


def align_traj(data, N_atoms, align_method=None):
    """Aligns trajectory using uniform or kronecker alignment.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        matrix of data to be aligned
    N_atoms : int
        number of atoms in the system
    align_method : {'uni', 'kron', 'uniform', 'kronecker'}, optional
        alignment method, by default "uni"
    
    Raises
    ------
    ValueError
        if align_method is not 'uni', 'uniform', 'kron', or 'kronecker' or None
    
    Returns
    -------
    array-like of shape (n_samples, n_features)
        matrix of aligned data
    """
    if not align_method:
        return data

    import torch 

    data = data.reshape(len(data), N_atoms, 3)
    device = torch.device('cpu')
    dtype = torch.float32
    traj_tensor = torch.tensor(data, device=device, dtype=dtype)
    torch_align.torch_remove_center_of_geometry(traj_tensor)
    if align_method == 'uni' or align_method == 'uniform':
        uniform_aligned_traj_tensor, uniform_avg_tensor, uniform_var_tensor = torch_align.torch_iterative_align_uniform(
            traj_tensor, device=device, dtype=dtype, verbose=True)
        aligned_traj_numpy = uniform_aligned_traj_tensor.cpu().numpy()
    elif align_method == 'kron' or align_method == 'kronecker':
        kronecker_aligned_traj_tensor, kronecker_avg_tensor, kronecker_precision_tensor, kronecker_lpdet_tensor = torch_align.torch_iterative_align_kronecker(
            traj_tensor, device=device, dtype=dtype, verbose=True)
        aligned_traj_numpy = kronecker_aligned_traj_tensor.cpu().numpy()
    else:
        raise ValueError('Please select a correct alignment method: uni, kron, or None')
    reshaped = aligned_traj_numpy.reshape(aligned_traj_numpy.shape[0], -1)
    return reshaped


def equil_align(indices, sieve, input_top, input_traj, mdana_atomsel, cpptraj_atomsel, ref_index):
    """ Aligns the frames in the trajectory to the reference frame.
    
    Parameters
    ----------
    indices : list
        List of indices of the data points in the cluster.
    input_top : str
        Path to the input topology file.
    input_traj : str
        Path to the input trajectory file.
    mdana_atomsel : str
        Atom selection string for MDAnalysis.
    cpptraj_atomsel : str
        Atom selection string for cpptraj.
    ref_index : int
        Index of the reference frame.
    
    Returns
    -------
    aligned_traj_numpy : numpy.ndarray
        Numpy array of the aligned trajectory.
    """
    u = mda.Universe(input_top, input_traj)
    with mda.Writer(f'unaligned_traj.pdb', u.atoms.n_atoms) as W:
        for ts in u.trajectory[[i * sieve for i in indices]]:
            W.write(u.atoms)
    with open('cpptraj.in', 'w') as outfile:
        outfile.write(f'parm {input_top}\n')
        outfile.write(f'trajin unaligned_traj.pdb\n')
        outfile.write('autoimage\n')
        outfile.write(f'reference {input_traj} frame {ref_index}\n')
        outfile.write(f'rms ToAvg reference {cpptraj_atomsel}\n')
        outfile.write('trajout aligned_traj.pdb nobox\n')
        outfile.write('run\n')
    subprocess.run(['cpptraj', '-i', 'cpptraj.in'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Read aligned trajectory
    aligned_traj_numpy = gen_traj_numpy(input_top, 'aligned_traj.pdb', atomSel=mdana_atomsel)

    # Clean up
    os.remove('cpptraj.in')
    os.remove('unaligned_traj.pdb')
    os.remove('aligned_traj.pdb')
    return aligned_traj_numpy

if __name__ == "__main__":
    arr = [[0, 1, 0, 0, 1, 0], 
           [1, 0, 1, 1, 0, 1],
           [1, 0, 0, 0, 1, 1],
           [1, 1, 0, 1, 1, 1],
           [0, 1, 1, 0, 1, 1]]
    arr = np.array(arr)


class KmeansNANI:
    """K-means algorithm with the N-Ary Natural Initialization (NANI).
    
    Attributes
    ----------
    data : array-like of shape (n_samples, n_features)
        Input dataset.
    n_clusters : int
        Number of clusters.
    metric : {'MSD', 'RR', 'JT', etc}
        Metric used for extended comparisons. 
        See `tools.bts.extended_comparison` for all available metrics.
    N_atoms : int
        Number of atoms.
    percentage : int
        Percentage of the dataset to be used for the initial selection of the 
        initial centers. Default is 10.
    init_type : {'div_select', 'comp_sim', 'k-means++', 'random'}
        Type of initiator selection. 
    labels : array-like of shape (n_samples,)
        Labels of each point.
    centers : array-like of shape (n_clusters, n_features)
        Cluster centers.
    n_iter : int
        Number of iterations run.
    cluster_dict : dict
        Dictionary of the clusters and their corresponding indices.
    
    Properties
    ----------
    kmeans : tuple
        Tuple of the labels, centers and number of iterations.
    kmeans_info : tuple
        Tuple of the labels, centers, number of iterations, scores and cluster dictionary.

    Methods
    -------
    __init__(data, n_clusters, metric, N_atoms, percentage=10, init_type='comp_sim'):
        Initializes the class.
    _check_init_type(self)
        Checks the 'init_type' attribute.
    _check_percentage(self)
        Checks the 'percentage' attribute.
    initiate_kmeans(self)
        Initializes the k-means algorithm with the selected initiators.
    kmeans_clustering(self, initiators)
        Gets the k-means algorithm.
    get_kmeans_info(self)
        Gets the k-means information.
    create_cluster_dict(self, labels)
        Creates a dictionary of the clusters and their corresponding indices.
    write_centroids(self, centroids, n_iter)
        Writes the centroids of the k-means algorithm to a file.
    """
    def __init__(self, data, n_clusters, metric, N_atoms=1, init_type='comp_sim', **kwargs):
        """Initializes the KmeansNANI class.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input dataset.
        n_clusters : int
            Number of clusters.
        metric : {'MSD', 'RR', 'JT', etc}
            Metric used for extended comparisons. 
            See `tools.bts.extended_comparison` for all available metrics.
        N_atoms : int
            Number of atoms.
        percentage : int
            Percentage of the dataset to be used for the initial selection of the 
            initial centers. Default is 10.
        init_type : {'comp_sim', 'div_select', 'k-means++', 'random', 'vanilla_kmeans++'}
            'comp_sim' selects the inital centers based on the diversity in the densest region of the data.
            'div_select' selects the initial centers based on the highest diversity of all data.
            'k-means++' selects the initial centers based on the greedy k-means++ algorithm.
            'random' selects the initial centers randomly.
            'vanilla_kmeans++' selects the initial centers based on the vanilla k-means++ algorithm
        """
        self.data = data
        self.n_clusters = n_clusters
        self.metric = metric
        self.N_atoms = N_atoms
        self.init_type = init_type
        self._check_init_type()
        if self.init_type == 'comp_sim' or self.init_type == 'div_select':
            self.percentage = kwargs.get('percentage', 10)
            self._check_percentage()
    
    def _check_init_type(self):
        """Checks the 'init_type' attribute.

        Raises
        ------
        ValueError
            If 'init_type' is not one of the following: 
            'comp_sim', 'div_select', 'k-means++', 'random', 'vanilla_kmeans++'.
        """
        if self.init_type not in ['comp_sim', 'div_select', 'k-means++', 'random', 'vanilla_kmeans++']:
            raise ValueError('init_type must be one of the following: comp_sim, div_select, k-means++, random, vanilla_kmeans++.')
        
    def _check_percentage(self):
        """Checks the 'percentage' attribute.
        
        Raises
        ------
        TypeError
            If percentage is not an integer.
        ValueError
            If percentage is not between 0 and 100.
        """
        if not isinstance(self.percentage, int):
            raise TypeError('percentage must be an integer [0, 100].')
        if not 0 <= self.percentage <= 100:
            raise ValueError('percentage must be an integer [0, 100].')
    
    def initiate_kmeans(self):
        """Initializes the k-means algorithm with the selected initiating method
        (comp_sim, div_select, k-means++, random, vanilla_kmeans++).

        Returns
        -------
        numpy.ndarray
            The initial centers for k-means of shape (n_clusters, n_features).
        """
        if self.init_type == 'comp_sim':
            n_total = len(self.data)
            n_max = int(n_total * self.percentage / 100)
            comp_sim = calculate_comp_sim(self.data, self.metric, self.N_atoms)
            sorted_comp_sim = sorted(comp_sim, key=lambda item: item[1], reverse=True)
            top_comp_sim_indices = [int(i[0]) for i in sorted_comp_sim][:n_max]
            top_cc_data = self.data[top_comp_sim_indices]
            initiators_indices = diversity_selection(top_cc_data, 100, self.metric, 
                                                     'medoid', self.N_atoms)
            initiators = top_cc_data[initiators_indices]
        elif self.init_type == 'div_select':
            initiators_indices = diversity_selection(self.data, self.percentage, self.metric, 
                                                     'medoid', self.N_atoms)
            initiators = self.data[initiators_indices]
        elif self.init_type == 'vanilla_kmeans++':
            initiators, indices = kmeans_plusplus(self.data, self.n_clusters, 
                                                  random_state=None, n_local_trials=1)
        return initiators
    
    def kmeans_clustering(self, initiators):
        """Executes the k-means algorithm with the selected initiators.

        Parameters
        ----------
        initiators : {numpy.ndarray, 'k-means++', 'random'}
            Method for selecting initial centers.
            'k-means++' selects initial centers in a smart way to speed up convergence.
            'random' selects initial centers randomly.
            If 'initiators' is a numpy.ndarray, it must be of shape (n_clusters, n_features) 

        Returns
        -------
        tuple
            Labels, centers and number of iterations of the k-means algorithm.
        """
        if self.init_type in ['k-means++', 'random']:
            initiators = self.init_type
            n_clusters = self.n_clusters
        elif isinstance(initiators, np.ndarray):
            initiators = initiators[:self.n_clusters]
            if len(initiators) < self.n_clusters:
                n_clusters = len(initiators)
            else:
                n_clusters = self.n_clusters

        n_init = 1
        kmeans = KMeans(n_clusters, init=initiators, n_init=n_init, random_state=None)
        kmeans.fit(self.data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        n_iter = kmeans.n_iter_
        return labels, centers, n_iter

    def create_cluster_dict(self, labels):
        """Creates a dictionary with the labels as keys and the indices of the data as values.
        
        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            Labels of the k-means algorithm.
        
        Returns
        -------
        dict
            Dictionary with the labels as keys and the indices of the data as values.
        """
        dict_labels = {}
        for i in range(self.n_clusters):
            dict_labels[i] = np.where(labels == i)[0]
        return dict_labels
    
    def compute_scores(self, labels):
        """Computes the Davies-Bouldin and Calinski-Harabasz scores.
        
        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            Labels of the k-means algorithm.
        
        Returns
        -------
        tuple
            Davies-Bouldin and Calinski-Harabasz scores.
        """
        ch_score = calinski_harabasz_score(self.data, labels)
        db_score = davies_bouldin_score(self.data, labels)
        return ch_score, db_score

    def write_centroids(self, centers, n_iter):
        """Writes the centroids of the k-means algorithm to a file.

        Parameters
        ----------
        centers : array-like of shape (n_clusters, n_features)
            Centroids of the k-means algorithm.
        n_iter : int
            Number of iterations of the k-means algorithm.
        """
        header = f'Number of clusters: {self.n_clusters}, Number of iterations: {n_iter}\n\nCentroids\n'
        np.savetxt('centroids.txt', centers, delimiter=',', header=header)
    
    def execute_kmeans_all(self):
        """Function to complete all steps of KMeans for all different 'init_type' options.

        Returns
        -------
        tuple
            Labels, centers and number of iterations of the k-means algorithm.
        """
        if self.init_type in ['comp_sim', 'div_select', 'vanilla_kmeans++']:
            initiators = self.initiate_kmeans()
            # print(f'{initiators=}')
            # print(f'{initiators[0][0] != initiators[1][0]} {initiators[0][1] != initiators[1][1]} {initiators[0][2] != initiators[1][2]}')
            labels, centers, n_iter = self.kmeans_clustering(initiators)
        elif self.init_type == 'k-means++' or self.init_type == 'random':
            labels, centers, n_iter = self.kmeans_clustering(initiators=self.init_type)
        return labels, centers, n_iter


def compute_scores(data, labels):
    """Computes the Davies-Bouldin and Calinski-Harabasz scores.
    
    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Labels of the k-means algorithm.
    
    Returns
    -------
    tuple
        Davies-Bouldin and Calinski-Harabasz scores.
    """
    ch_score = calinski_harabasz_score(data, labels)
    db_score = davies_bouldin_score(data, labels)
    return ch_score, db_score
