import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Optional, List, Dict, Any, Generator, Union
import time
import numpy as np
import numpy.typing as npt
import pandas as pd
from natsort import natsorted
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from westpa.core.segment import Segment
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
import mdtraj
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from os.path import expandvars

import os
from westpa_ddmd.config import BaseSettings
import operator
from copy import deepcopy

import westpa
from westpa.core.binning import Bin
from westpa.core.we_driver import WEDriver
from westpa.core.h5io import tostr

log = logging.getLogger(__name__)

# TODO: This is a temporary solution until we can pass
# arguments through the westpa config. Requires a
# deepdrivemd.yaml file in the same directory as this script
CONFIG_PATH = Path(__file__).parent / "deepdrivemd.yaml"
SIM_ROOT_PATH = Path(__file__).parent

class DeepDriveMDDriver(WEDriver, ABC):

    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)
        
        self.weight_split_threshold = 1.0
        self.weight_merge_cutoff = 1.0

        self.niter: int = 0
        self.nsegs: int = 0
        self.nframes: int = 0
        self.cur_pcoords: npt.ArrayLike = []
        self.rng = np.random.default_rng()
        self.segments = None
        self.bin = None
        # Note: Several of the getter methods that return npt.ArrayLike
        # objects use a [:, 1:] index trick to avoid the initial frame
        # in westpa segments which corresponds to the last frame of the
        # previous iterations segment which is helpful for rate-constant
        # calculations, but not helpful for DeepDriveMD.

    @abstractmethod
    def run(self, segments: Sequence[Segment]) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Implement the DeepDriveMD routine.

        Returns
        -------
        npt.ArrayLike
            Array of integer indices for which segments to split.
        npt.ArrayLike
            Array of integer indices for which segments to merge.
        """
        ...

    def _split_by_data(self, bin: Bin, segment : Segment, num_splits: int) -> None:
        """
        Split the segment num_splits times into the bin

        Parameters
        ----------
        bin : array_like
            the bin to split into
        segment : Segment
            the segment to be split
        num_splits : int
            number of times to split

        Returns
        -------
        None
        """
        bin.remove(segment)
        new_segments_list = self._split_walker(segment, num_splits, bin)
        bin.update(new_segments_list)

    def _merge_by_data(self, bin: Bin, to_merge: Sequence[Segment]) -> None:
        """
        Merge a group of walkers together

        Parameters
        ----------
        bin : array_like
            the bin to split into
        to_merge : Sequence[Segment]
            a list of segments to be merged together

        Returns
        -------
        None
        """
        bin.difference_update(to_merge)
        new_segment, _ = self._merge_walkers(to_merge, None, bin)
        bin.add(new_segment)

    def _split_by_weight(self, cluster_df: pd.DataFrame, ideal_segs_per_cluster: int) -> dict:
        """
        Split segs over the ideal weight

        Parameters
        ----------
        cluster_df : pd.DataFrame
            dataframe containing info for all the segs in the cluster
        ideal_segs_per_cluster : int
            number of segments that should be in each cluster for even sampling
        
        Returns
        -------
        None
        """
        ideal_weight = cluster_df['weight'].sum() / ideal_segs_per_cluster
        print(f"{ideal_weight=}")
        # Get all of the segments over the ideal weight
        to_split = cluster_df[cluster_df['weight'] > self.weight_split_threshold*ideal_weight]
        
        for _, row in to_split.iterrows():
            # Find the ind for the row to be split
            split_ind = int(row['inds'])
            # Determine the number of necessary splits to add
            m = int(row['weight'] / ideal_weight)
            # Split the segment
            self._split_by_data(self.bin, self.segments[split_ind], m)
    
    def _merge_by_weight(self, cluster_df: pd.DataFrame, ideal_segs_per_cluster: int):
        '''Merge underweight particles'''

        # Ideal weight for this cluster
        ideal_weight = cluster_df['weight'].sum() / ideal_segs_per_cluster
        print(f"{ideal_weight=}")
        while True:
            # Sort the df by weight
            cluster_df.sort_values('weight', inplace=True)
            # Add up all of the weights
            cumul_weight = np.add.accumulate(cluster_df['weight'])
            # Get the walkers that add up to be under the ideal weight
            to_merge = cluster_df[cumul_weight <= ideal_weight*self.weight_merge_cutoff]
            # If there's not enough for a merge then return
            if len(to_merge) < 2:
                break
            # Merge the segments
            self._merge_by_data(self.bin, self.segments[to_merge.inds.values])
            # Remove the merged walkers
            cluster_df.drop(to_merge.index.values, inplace=True)
            # Reset the index
            cluster_df.reset_index(drop=True, inplace=True)            
  
    def get_prev_dcoords(self, iterations: int, upperbound: Optional[int] = None) -> npt.ArrayLike:
        """Collect coordinates from previous iterations.

        Parameters
        ----------
        iterations : int
            Number of previous iterations to collect.

        upperbound : Optional[int]
            The upper bound range (exclusive) of which to get the past dcoords

        Returns
        -------
        npt.ArrayLike
            Coordinates with shape (N, Nsegments, Nframes, Natoms, Natoms)
        """
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

        # TODO: If this function is slow, we can try direct h5 reads

        if upperbound is None:
            upperbound = self.niter

        back_coords = []
        with data_manager.lock:
            for i in range(upperbound - iterations, upperbound):
                iter_group = data_manager.get_iter_group(i)
                coords_raw = iter_group["auxdata/dmatrix"][:]
                for seg in coords_raw[:, 1:]:
                    back_coords.append(seg)

        return back_coords

    def get_restart_dcoords(self) -> npt.ArrayLike:
        """Collect coordinates for restart from previous iteration.
        Returns
        -------
        npt.ArrayLike
            Coordinates with shape (N, Nsegments, Nframes, Natoms, 3)
        """
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

        # TODO: If this function is slow, we can try direct h5 reads

        back_coords = []
        with data_manager.lock:
            iter_group = data_manager.get_iter_group(self.niter)
            coords_raw = iter_group["auxdata/dmatrix"][:]
            for seg in coords_raw[:, 1:]:
                back_coords.append(seg)

        return back_coords

    def get_prev_rcoords(self, iterations: int) -> npt.ArrayLike:
        """Collect coordinates from previous iterations.

        Parameters
        ----------
        iterations : int
            Number of previous iterations to collect.

        Returns
        -------
        npt.ArrayLike
            Coordinates with shape (N, Nsegments, Nframes, Natoms, 3)
        """
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

        # TODO: If this function is slow, we can try direct h5 reads

        back_coords = []
        with data_manager.lock:
            for i in range(self.niter - iterations, self.niter):
                iter_group = data_manager.get_iter_group(i)
                coords_raw = iter_group["auxdata/coord"][:]
                coords_raw = coords_raw.reshape((self.nsegs, self.nframes + 1, -1, 3))
                for seg in coords_raw[:, 1:]:
                    back_coords.append(seg)

        return back_coords

    def get_restart_rcoords(self) -> npt.ArrayLike:
        """Collect coordinates for restart from previous iteration.
        Returns
        -------
        npt.ArrayLike
            Coordinates with shape (N, Nsegments, Nframes, Natoms, 3)
        """
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

        # TODO: If this function is slow, we can try direct h5 reads

        back_coords = []
        with data_manager.lock:
            iter_group = data_manager.get_iter_group(self.niter)
            coords_raw = iter_group["auxdata/coord"][:]
            coords_raw = coords_raw.reshape((self.nsegs, self.nframes + 1, -1, 3))
            for seg in coords_raw[:, 1:]:
                back_coords.append(seg)

        return back_coords

    def get_restart_auxdata(self, field: Optional[str] = None) -> npt.ArrayLike:
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

        # TODO: If this function is slow, we can try direct h5 reads

        back_data = []
        with data_manager.lock:
            iter_group = data_manager.get_iter_group(self.niter)

            if field:
                data_raw = iter_group["auxdata/" + field][:]
            else:
                data_raw = iter_group["auxdata"][:]

            for seg in data_raw[:, 1:]:
                back_data.append(seg)

        return np.array(back_data)

    def load_synd_model(self):
        from synd.core import load_model
        import pickle

        subgroup_args = westpa.rc.config.get(['west', 'drivers'])
        synd_model_path = expandvars(subgroup_args['synd_model'])
        backmap_path = expandvars(subgroup_args['dmatrix_map'])

        synd_model = load_model(synd_model_path)

        with open(backmap_path, 'rb') as infile:
            dmatrix_map = pickle.load(infile)

        synd_model.add_backmapper(dmatrix_map.get, name='dmatrix')

        return synd_model

    def get_prev_dcoords_training(self, segments: Sequence[Segment], curr_dcoords: np.ndarray, iters_back: int) -> np.ndarray:
        # Grab all the d-matrices from the last iteration (n_iter-1), will be in seg_id order by default.
        # TODO: Hope we have some way to simplify this... WESTPA 3 function...
        ibstate_group = westpa.rc.get_data_manager().we_h5file['ibstates/0']

        if self.niter > 1:
            past_dcoords = []
            if iters_back == 0:
                past_dcoords = []
            elif self.niter <= iters_back:
                iters_back = self.niter -1
                past_dcoords = np.concatenate(self.get_prev_dcoords(iters_back, upperbound=self.niter))
            else:
                past_dcoords = np.concatenate(self.get_prev_dcoords(iters_back, upperbound=self.niter))

            # Get current iter dcoords
            # curr_dcoords = np.concatenate(self.get_prev_dcoords(1))
            # print(curr_dcoords)
        else:  # building the list from scratch, during first iter
            past_dcoords, curr_dcoords = [], []
            for segment in segments:
                istate_id = ibstate_group['istate_index']['basis_state_id', int(segment.parent_id)]
                #print(istate_id)
                auxref = int(tostr(ibstate_group['bstate_index']['auxref', istate_id]))
                #print(auxref)

                dmatrix = self.synd_model.backmap([auxref], mapper='dmatrix')
                curr_dcoords.append(dmatrix[0])
                
        chosen_dcoords = []
        to_pop = []
        for idx, segment in enumerate(segments):
            #print(segment)
            #print(seg.wtg_parent_ids)
            if segment.parent_id < 0:
                istate_id = ibstate_group['istate_index']['basis_state_id', -int(segment.parent_id + 1)]
                #print(istate_id)
                auxref = int(tostr(ibstate_group['bstate_index']['auxref', istate_id]))
                #print(auxref)

                dmatrix = self.synd_model.backmap([auxref], mapper='dmatrix')
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

        # print(len(chosen_dcoords))
        # print(f'curr_dcoords shape: {curr_dcoords.shape}')
        # print(len(past_dcoords))
        # # print(final)
        # print(len(final))

        # print('yay')
        # if len(final) > 1:
        #    print(f'{final[0].shape}, {final[1].shape}')

        return np.vstack((final))

    def get_rcoords(self, segments: Sequence[Segment]) -> npt.ArrayLike:
        """Concatenate the coordinates frames from each segment."""
        rcoords = np.array(list(seg.data["rcoord"] for seg in segments))
        return rcoords.reshape(self.nsegs, self.nframes + 1, -1, 3)[:, 1:]
        # return rcoords.reshape(self.nsegs, self.nframes, -1, 3)[:, 1:]

    def get_dcoords(self, segments: Sequence[Segment]) -> npt.ArrayLike:
        """
        Calculate the dcoords (displacement coordinates) for the given segments.

        Parameters:
        - segments: A sequence of Segment objects.

        Returns:
        - dcoords: An array-like object containing the dcoords for each segment.
        """
        dcoords = np.array(list(seg.data["dmatrix"] for seg in segments))
        return dcoords[:,1:]
        #return rcoords.reshape(self.nsegs, self.nframes + 1, -1, 3)[:, 1:]
        # return rcoords.reshape(self.nsegs, self.nframes, -1, 3)[:, 1:]

    def get_pcoords(self, segments: Sequence[Segment]) -> npt.ArrayLike:
        """
        Get the pcoords (progress coordinates) from a list of segments.

        Parameters:
            segments (Sequence[Segment]): A sequence of Segment objects.

        Returns:
            npt.ArrayLike: An array-like object containing the pcoords.

        """
        pcoords = np.array(list(seg.pcoord for seg in segments))[:, 1:]
        return pcoords.reshape(pcoords.shape[0], pcoords.shape[1], -1)

    def get_weights(self, segments: Sequence[Segment]) -> npt.ArrayLike:
        return np.array(list(seg.weight for seg in segments))

    def get_auxdata(
        self, segments: Sequence[Segment], field: Optional[str] = None
    ) -> npt.ArrayLike:
        if field:
            return np.array(list(seg.data[field] for seg in segments))[:, 1:]
        return np.array(list(seg.data for seg in segments))[:, 1:]

    def _get_segments_by_weight(
        self, bin_: Union[Bin, Generator[Segment, None, None]]  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.weight), np.object_)
        #return np.array([seg for seg in bin_])
    
    def _get_segments_by_parent_id(
        self, bin_: Union[Bin, Generator[Segment, None, None]]  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.parent_id), np.object_)
    
    def _get_segments_by_seg_id(
        self, bin_: Union[Bin, Generator[Segment, None, None]]  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.seg_id), np.object_)
    
    def _adjust_count(self, ibin):
        bin = self.next_iter_binning[ibin]
        target_count = self.bin_target_counts[ibin]
        weight_getter = operator.attrgetter('weight')

        # split        
        while len(bin) < target_count:
            log.debug('adjusting counts by splitting')
            # always split the highest probability walker into two
            segments = sorted(bin, key=weight_getter)[-1]
            bin.remove(segments)
            new_segments_list = self._split_walker(segments, 2, bin)
            bin.update(new_segments_list)

        # merge
        while len(bin) > target_count:
            log.debug('adjusting counts by merging')
            # always merge the two lowest-probability walkers
            segments = sorted(bin, key=weight_getter)[:2]
            bin.difference_update(segments)
            new_segment, _ = self._merge_walkers(segments, None, bin)
            bin.add(new_segment)

    def _run_we(self) -> None:
        """Run recycle/split/merge. Do not call this function directly; instead, use
        populate_initial(), rebin_current(), or construct_next()."""

        # Recycle walkers that have reached the target state
        self._recycle_walkers()

        # Sanity check
        self._check_pre()

        # Resample
        for ibin, bin_ in enumerate(self.next_iter_binning):
            if len(bin_) == 0:
                continue
            else:
                self.niter = np.array([seg for seg in bin_])[0].n_iter -1 
            
            if not self.niter:
                # Randomly merge til we have the target count
                self._adjust_count(ibin)

            # This checks for initializing; if niter is 0 then skip resampling
            if self.niter:
                # This is an attempt to sort all of the segments consistently
                # If there is any recycling, we sort by the weight (less consistent)
                # otherwise we are sorting by parent_id and seg_id (more consistent)
                # The hope is that by the time there is recycling, the weights will be unique
                recycle_check = np.array([seg.parent_id for seg in bin_])
                if np.any(recycle_check < 0):
                    segments = self._get_segments_by_weight(bin_)
                    cur_segments = self._get_segments_by_weight(self.current_iter_segments)
                else:
                    segments = self._get_segments_by_parent_id(bin_)
                    cur_segments = self._get_segments_by_seg_id(self.current_iter_segments)
                
                # print(segments)
                # print(cur_segments)
                self.cur_pcoords = self.get_pcoords(cur_segments)

                # TODO: Is there a way to get nsegs and nframes without pcoords?
                # If so, we could save on a lookup. Can we retrive it from cur_segments?
                self.niter = cur_segments[0].n_iter
                self.nsegs = self.cur_pcoords.shape[0]
                self.nframes = self.cur_pcoords.shape[1]          

                self.run(bin_, cur_segments, segments)

                print("Num walkers going to the final adjust_counts:", len([x for x in bin_]))

                self._adjust_count(ibin)
        # another sanity check
        self._check_post()

        # TODO: What does this line do?
        self.new_weights = self.new_weights or []

        log.debug("used initial states: {!r}".format(self.used_initial_states))
        log.debug("available initial states: {!r}".format(self.avail_initial_states))

def plot_scatter(
    data: np.ndarray,
    color: np.ndarray,
    output_path,
    title: Optional[str] = None,
    cb_label: Optional[str] = None,
    num_segs: Optional[int] = None,
    target_point: Optional[np.ndarray] = None,
    min_max_color: Optional[Tuple[float, float]] = None,
    log_scale: bool = False,
):
    if min_max_color is not None:
        min_color, max_color = min_max_color
    else:
        min_color = np.min(color)
        max_color = np.max(color)
    # if num_segs is None:
    #     x = data[:, 0]
    #     y = data[:, 1]
    #     z = data[:, 2]
    # else:
    #     x = data[num_segs:, 0]
    #     y = data[num_segs:, 1]
    #     z = data[num_segs:, 2]
    #     x2 = data[:num_segs, 0]
    #     y2 = data[:num_segs, 1]
    #     z2 = data[:num_segs, 2]
    #     if color.shape[0] == num_segs:
    #         c = "gray"
    #     else:
    #         c = color[num_segs:]
    #         c2 = color[:num_segs]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    set_color = False
    if num_segs is not None:
        if color.shape[0] == num_segs:
            hist = ax.scatter(data[num_segs:, 0], 
                            data[num_segs:, 1], 
                            data[num_segs:, 2], 
                            # Color the points gray
                            c="gray",
                            label="Past iterations",
            )
        else:
            if log_scale:
                hist = ax.scatter(data[num_segs:, 0], 
                            data[num_segs:, 1], 
                            data[num_segs:, 2], 
                            c=color[num_segs:],
                            norm=mpl.colors.LogNorm(vmin=min_color, vmax=max_color), 
                            label="Past iterations",
                )
            else:   
                hist = ax.scatter(data[num_segs:, 0], 
                                data[num_segs:, 1], 
                                data[num_segs:, 2], 
                                c=color[num_segs:],
                                vmin=min_color,
                                vmax=max_color,
                                label="Past iterations",
                )
            
            set_color = True
        if log_scale:
            cur = ax.scatter(
                data[:num_segs, 0],
                data[:num_segs, 1],
                data[:num_segs, 2],
                c=color[:num_segs],
                # plot them with squares
                marker="s",
                # Make the points larger
                s=150,
                norm=mpl.colors.LogNorm(vmin=min_color, vmax=max_color),
                label="Current iteration", 
            )
        else:
            cur = ax.scatter(
                data[:num_segs, 0],
                data[:num_segs, 1],
                data[:num_segs, 2],
                c=color[:num_segs],
                # plot them with squares
                marker="s",
                # Make the points larger
                s=150,
                vmin=min_color,
                vmax=max_color, 
                label="Current iteration",
            )
    else:   
        cur = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)

    if target_point is not None:
        ax.scatter(target_point[0], target_point[1], target_point[2], c="red", marker="x", s=200, label="Target point")

    if set_color:
        colorby = hist
    else:
        colorby = cur
    if cb_label is not None:
        plt.colorbar(colorby).set_label(cb_label)
    else:
        plt.colorbar(colorby)

    if title is not None:
        plt.title(title, loc="left")
    ax.legend(loc='upper left')
    plt.savefig(output_path)

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the cosine distance between two vectors.

    Parameters:
    v1 (np.ndarray): The first vector.
    v2 (np.ndarray): The second vector.

    Returns:
    float: The cosine distance between the two vectors.
    """
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - similarity

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Parameters:
    v1 (np.ndarray): The first vector.
    v2 (np.ndarray): The second vector.

    Returns:
    float: The Euclidean distance between the two vectors.
    """
    return np.linalg.norm(v1 - v2)

def find_combinations(target, current_combination, start, combo_list):
    """
    Finds all combinations of numbers that add up to the given target.

    Parameters:
    - target (int): The target sum.
    - current_combination (list): The current combination of numbers.
    - start (int): The starting index for the loop.
    - combo_list (list): The list to store all the combinations.

    Returns:
    - None

    This method recursively finds all combinations of numbers from a given list that add up to the target sum.
    It starts from the specified start index and builds the combinations by adding numbers to the current combination.
    The combinations are stored in the combo_list.

    Example usage:
    ```
    target = 10
    current_combination = []
    start = 0
    combo_list = []
    find_combinations(target, current_combination, start, combo_list)
    print(combo_list)  # Output: [[2, 3, 5], [1, 4, 5]]
    ```
    """
    # Base case
    if target == 0:
        # Add the current combination to the list
        combo_list.append(np.array(current_combination))
        return
    # Recursive case
    if target < 0:
        return
    # Iterate over the numbers from the start index
    for i in range(start, target + 1):
        # Add the number to the current combination
        find_combinations(target - i, current_combination + [i], i, combo_list)

class CVAESettings(BaseSettings):
    """Settings for mdlearn SymmetricConv2dVAETrainer object."""

    input_shape: Tuple[int, int, int] = (1, 40, 40)
    filters: List[int] = [16, 16, 16, 16]
    kernels: List[int] = [3, 3, 3, 3]
    strides: List[int] = [1, 1, 1, 2]
    affine_widths: List[int] = [128]
    affine_dropouts: List[float] = [0.5]
    latent_dim: int = 3
    lambda_rec: float = 1.0
    num_data_workers: int = 0
    prefetch_factor: Optional[int] = None
    batch_size: int = 64
    device: str = "cuda"
    optimizer_name: str = "RMSprop"
    optimizer_hparams: Dict[str, Any] = {"lr": 0.001, "weight_decay": 0.00001}
    epochs: int = 100
    checkpoint_log_every: int = 25
    plot_log_every: int = 25
    plot_n_samples: int = 5000
    plot_method: Optional[str] = "raw"

class MLSettings(BaseSettings):
    # static, train
    ml_mode: str
    # Contact map distance cutoff
    contact_cutoff: float
    # How often to update the model. Set to 0 for static.
    update_interval: Optional[int] = 0
    # Number of lagging iterations to use for training data.
    lag_iterations: Optional[int] = 0
    # Path to the base training data (Optional).
    base_training_data_path: Optional[Path] = None
    # Checkpoint file for static mode (Optional).
    static_chk_path: Optional[Path] = None

    @classmethod
    def from_westpa_config(cls) -> "MLSettings":
        westpa_config = westpa.rc.config.get(['west', 'ddmd', 'machine_learning'], {})
        return MLSettings(**westpa_config)

class MachineLearningMethod:
    def __init__(self, niter, log_path):
        """Initialize the machine learning method.

        Parameters
        ----------
        train_path : Path
            The path to save the model and training data.
        base_training_data_path : Path
            The path to a set of pre-exisiting training data to help the model properly converge.
        """
        self.cfg = MLSettings.from_westpa_config()
        self.niter = niter
        self.log_path = log_path
        if self.cfg.ml_mode == 'train':
            # Determine the location for the training data/model
            if self.niter < self.cfg.update_interval:
                self.train_path = self.log_path / "ml-iter-1"
            else:
                self.train_path = (
                    self.log_path
                    / f"ml-iter-{self.niter - self.niter % self.cfg.update_interval}"
                )
            # Init the ML method
            self.train_path.mkdir(exist_ok=True)
        else:
            self.train_path = None

        # Initialize the model
        self.autoencoder = SymmetricConv2dVAETrainer(**CVAESettings().model_dump())

    def get_target_point_coords(self, target_point_path: Path) -> np.ndarray:
        """Get the xyz coordinates for the CA atoms of the target point."""
        # Load the target point pdb in mdtraj
        target_point = mdtraj.load(str(target_point_path))
        # Select the CA atoms
        ca_atoms = target_point.top.select("name CA")
        # Get the xyz coordinates for the CA atoms
        coords = target_point.xyz[0][ca_atoms]
        # Reshape the coordinates to the shape (frames, atoms, atoms)
        return np.reshape(coords, (1, len(ca_atoms), 3))

    def compute_sparse_contact_map(self, coords: np.ndarray) -> np.ndarray:
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
    
    @property
    def save_path(self) -> Path:
        """Path to save the model."""
        return self.train_path / "saves"

    @property
    def most_recent_checkpoint_path(self) -> Path:
        """Get the most recent model checkpoint."""
        if self.cfg.static_chk_path is not None:
            return self.cfg.static_chk_path
        else:
            checkpoint_dir = self.save_path / "checkpoints"
            model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]
            return model_weight_path

    def train(self, coords: np.ndarray) -> None:
        """Takes aligned data and trains a new model on it. Outputs are saved in fixed postions."""
        if self.niter == 1 or self.niter % self.cfg.update_interval == 0:
            print(f"Training a model on iteration {self.niter}...")
            # Load the base training data with the shape (frames, atoms, atoms)
            base_coords = np.load(self.cfg.base_training_data_path)
            
            # Concatenate the base training data with the new data (frames, atoms, atoms)
            coords = np.concatenate((base_coords, coords))
            # Compute the contact maps
            contact_maps = self.compute_sparse_contact_map(coords)
            start = time.time()
            # Train model
            self.autoencoder.fit(X=contact_maps, output_path=self.save_path)
            print(f"Training for {time.time() - start} seconds")

            # Save the loss curve
            pd.DataFrame(self.autoencoder.loss_curve_).plot().get_figure().savefig(
                str(self.train_path / "model_loss_curve.png")
            )
            
            pd.DataFrame(self.autoencoder.loss_curve_).to_csv(self.train_path / "loss.csv")

            z, *_ = self.autoencoder.predict(
                contact_maps, checkpoint=self.most_recent_checkpoint_path
            )
            np.save(self.train_path / "z.npy", z[len(base_coords):])

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Predict the latent space coordinates for a set of coordinate frames."""
        # Compute the contact maps
        contact_maps = self.compute_sparse_contact_map(coords)
        # Predict the latent space coordinates
        
        z, *_ = self.autoencoder.predict(
            contact_maps, checkpoint=self.most_recent_checkpoint_path
        )
        return z
    
    def get_target_point_rep(self, target_point_path: Path) -> np.ndarray:
        """Get the target point representation."""
        # Load the target point coordinates
        target_point = self.get_target_point_coords(target_point_path)
        # Won't normally need these two lines
        distance_matrices = [distance_matrix(frame, frame) for frame in target_point]
        contact_maps = np.array(distance_matrices) < self.cfg.contact_cutoff
        # Predict the target point representation
        target_point_rep = self.predict(contact_maps)
        return np.concatenate(target_point_rep)

class ObjectiveSettings(BaseSettings):
    # Objective method to use (dbscan, kmeans, gmm, or lof).
    objective_method: str
    # Function to measure distance between latent space points (cosine or euclidean).
    distance_metric: str
    # Interval for plotting the latent space.
    plot_interval: int
    # Maximum total number of past latent space points to save for the lof scheme.
    max_past_points: Optional[int]
    # Number of neighbors for LOF.
    lof_n_neighbors: Optional[int]
    # Maximum number of contact maps to save from each cluster.
    max_save_per_cluster: Optional[int]
    # Number of KMeans clusters.
    kmeans_clusters: Optional[int]
    # Epsilon setting for dbscan.
    dbscan_epsilon: Optional[float]
    # Minimum number of points for dbscan.
    dbscan_min_samples: Optional[int]
    # Max components for GMM.
    gmm_max_components: Optional[int]
    # Number of "clusters" to use for the ablation version.
    ablation_clusters: Optional[int]

    @classmethod
    def from_westpa_config(cls) -> 'ObjectiveSettings':
        westpa_config = westpa.rc.config.get(['west', 'ddmd', 'objective'], {})
        return ObjectiveSettings(**westpa_config)

class Objective:
    def __init__(self, nsegs: int, niter: int, datasets_path: Path, log_path: Path, split_weight_limit: float, merge_weight_limit: float, target_point: Optional[np.ndarray] = None):
        self.cfg = ObjectiveSettings.from_westpa_config()
        self.nsegs = nsegs
        self.niter = niter
        self.datasets_path = datasets_path
        self.log_path = log_path
        # Presently this is here only for plotting purposes
        self.target_point = target_point
        self.split_weight_limit = split_weight_limit
        self.merge_weight_limit = merge_weight_limit
        self.rng = np.random.default_rng()
        dist_functions = {
            "cosine": cosine_distance,
            "euclidean": euclidean_distance,
        }
        self.distance_function = dist_functions[self.cfg.distance_metric]

    def save_latent_context(self, cluster_labels: Optional[np.ndarray] = None) -> None:
        """
        Save the cluster context to files.

        Args:
            cluster_labels (np.ndarray): Array of cluster labels.
            cluster_dcoords (np.ndarray): Array of cluster dcoords.
            cluster_pcoords (np.ndarray): Array of cluster pcoords.
            cluster_weights (np.ndarray): Array of cluster weights.

        Returns:
            None
        """
        dcoords_to_save = []
        pcoords_to_save = []
        weights_to_save = []
        if cluster_labels is None:
            # Select indices at random from the entire dataset
            indices = self.rng.choice(len(self.all_dcoords), self.cfg.max_past_points, replace=False)
            dcoords_to_save.append(self.all_dcoords[indices])
            pcoords_to_save.append(self.all_pcoords[indices])
            weights_to_save.append(self.all_weights[indices])
        else:
            # Loop through each cluster label and randomly select indices to save
            for label in set(cluster_labels):
                indices = np.where(cluster_labels == label)[0]
                # If the number of indices is greater than the max save per cluster, randomly select the max save per cluster
                # But save all of the outliers
                if label != -1 and len(indices) > self.cfg.max_save_per_cluster:
                    indices = self.rng.choice(indices, self.cfg.max_save_per_cluster, replace=False)
                dcoords_to_save.append(self.all_dcoords[indices])
                pcoords_to_save.append(self.all_pcoords[indices])
                weights_to_save.append(self.all_weights[indices])
        dcoords_to_save = [x for y in dcoords_to_save for x in y]
        pcoords_to_save = [x for y in pcoords_to_save for x in y]
        weights_to_save = [x for y in weights_to_save for x in y]
        # Index the dcoord and pcoord arrays to save the selected indices
        np.save(self.datasets_path / f"context-dcoords-{self.niter}.npy", dcoords_to_save)
        np.save(self.datasets_path / f"context-pcoords-{self.niter}.npy", pcoords_to_save)
        np.save(self.datasets_path / f"context-weights-{self.niter}.npy", weights_to_save)

    def load_latent_context(self, dcoords, pcoords, weight) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the cluster context for clustering.

        Args:
            dcoords (np.ndarray): The current dcoords.
            pcoords (np.ndarray): The current pcoords.
            weight (np.ndarray): The current weight.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the concatenated dcoords, pcoords, and weights.

        """
        if self.niter > 1:
            # Load the past data to use as context for clustering
            past_dcoords = np.load(self.datasets_path / f"context-dcoords-{self.niter - 1}.npy")
            past_pcoords = np.load(self.datasets_path / f"context-pcoords-{self.niter - 1}.npy")
            past_weights = np.load(self.datasets_path / f"context-weights-{self.niter - 1}.npy")
            self.all_dcoords = np.concatenate([dcoords, past_dcoords])
            self.all_pcoords = np.concatenate([pcoords, past_pcoords])
            self.all_weights = np.concatenate([weight, past_weights])
        else:
            self.all_dcoords = dcoords
            self.all_pcoords = pcoords
            self.all_weights = weight

    def lof_function(self, cluster_z: np.ndarray) -> np.ndarray:
         # Run LOF on the full history of embeddings to assure coverage over past states
        clf = LocalOutlierFactor(n_neighbors=self.cfg.lof_n_neighbors).fit(cluster_z)
        return clf.negative_outlier_factor_

    def plot_latent_space(self, cluster_z: np.ndarray, cluster_labels: np.ndarray) -> None:
        plot_scatter(
            cluster_z,
            cluster_labels,
            self.log_path / f"embedding-cluster-{self.niter}.png",
            title=f"iter: {self.niter}",
            cb_label="cluster ID",
            num_segs=self.nsegs,
            target_point=self.target_point if self.target_point is not None else None,
        )
        plot_scatter(
            cluster_z,
            self.all_pcoords,
            self.log_path / f"embedding-pcoord-{self.niter}.png",
            title=f"iter: {self.niter}",
            cb_label="rmsd to target",
            num_segs=self.nsegs,
            target_point=self.target_point if self.target_point is not None else None,
            min_max_color=(0, 14),
        )
        plot_scatter(
            cluster_z,
            np.array([cosine_distance(z, self.target_point) for z in cluster_z]),
            self.log_path / f"embedding-distance-{self.niter}.png",
            title=f"iter: {self.niter}",
            cb_label="latent space distance to target",
            num_segs=self.nsegs,
            target_point=self.target_point if self.target_point is not None else None,
            min_max_color=(0, 2),
        )
        plot_scatter(
            cluster_z,
            self.all_weights,
            self.log_path / f"embedding-weight-{self.niter}.png",
            title=f"iter: {self.niter}",
            cb_label="weight",
            num_segs=self.nsegs,
            target_point=self.target_point if self.target_point is not None else None,
            min_max_color=(self.split_weight_limit, self.merge_weight_limit),
            log_scale=True,
        )
    
    def kmeans_cluster_segments(self, cluster_z: np.ndarray) -> np.ndarray:
        # Perform the K-means clustering
        cluster_labels = KMeans(n_clusters=self.cfg.kmeans_clusters).fit(cluster_z).labels_
        return cluster_labels
    
    def dbscan_cluster_segments(self, cluster_z: np.ndarray) -> np.ndarray:
        """
        Cluster the segments using DBSCAN algorithm and assign cluster labels to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.
            cluster_z (np.ndarray): The latent space representation of the segments.

        Returns:
            pd.DataFrame: The DataFrame with cluster labels assigned to the segments.
        """
        # Cluster the segments
        cluster_labels = DBSCAN(min_samples=self.cfg.dbscan_min_samples, 
                                eps=self.cfg.dbscan_epsilon, 
                                metric=self.distance_function).fit(cluster_z).labels_
        print(f"Total number of clusters: {max(set(cluster_labels)) + 1}")
        return cluster_labels

    # TODO: Fix the outliers for gmm
    def gmm_cluster_segments(self, cluster_z: np.ndarray) -> np.ndarray:
        """
        Cluster the segments using Gaussian Mixture Model and assign cluster labels to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.
            cluster_z (np.ndarray): The latent space representation of the segments.

        Returns:
            pd.DataFrame: The DataFrame with cluster labels assigned to the segments.
        """
        # Perform the GMM clustering
        gmm = BayesianGaussianMixture(n_components=self.cfg.gmm_max_components).fit(cluster_z)
        cluster_labels = gmm.predict(cluster_z)
        return cluster_labels

    def assign_density_outliers(self, cluster_z: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """
        Assigns density outliers to the input DataFrame based on clustering results.

        Args:
            df (pd.DataFrame): The input DataFrame.
            cluster_z (np.ndarray): The embedding history of the clusters.
            cluster_labels (np.ndarray): The labels of the clusters.
            seg_z (np.ndarray): The embedding history of the segments.

        Returns:
            np.ndarray: The updated DataFrame with density outliers assigned.

        """
        outliers = np.zeros(self.nsegs).astype(bool)
        seg_z = cluster_z[:self.nsegs]
        seg_labels = cluster_labels[:self.nsegs]
        # if there are any outliers
        print(f"Number of outliers: {np.sum(cluster_labels[:self.nsegs] == -1)}")
        if -1 in seg_labels:
            # Set all the -1s to be outliers
            outliers = np.where(seg_labels == -1, True, False)
            # Remove the outliers from the projections and labels
            inlier_embedding_history = cluster_z[cluster_labels != -1]
            inlier_labels = cluster_labels[cluster_labels != -1]
            # Loop through the outlier indices
            for ind in np.nditer(np.where(outliers)):
                # Find the distance to points in the embedding_history
                dist = np.array([self.distance_function(seg_z[ind], z) for z in inlier_embedding_history])
                # Find the min index
                min_ind = np.argmin(dist)
                # Set the cluster label for the outlier to match the min dist point
                cluster_labels[ind] = inlier_labels[min_ind]
        return outliers, cluster_labels

    def cluster_segments(self, cluster_z: np.ndarray) -> np.ndarray:
        """
        Cluster the segments based on the given method.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.
            cluster_z (np.ndarray): The latent space representation of the segments.

        Returns:
            pd.DataFrame: The DataFrame with cluster labels assigned to the segments.

        """

        # TODO: Fix the other cluster methods
        # TODO: Maybe move a bunch of this junk to it's own function in objectives
        # Cluster the segments
        if self.cfg.objective_method == 'kmeans':
            cluster_labels = self.kmeans_cluster_segments(cluster_z)
        elif self.cfg.objective_method == 'dbscan':
            cluster_labels = self.dbscan_cluster_segments(cluster_z)
        elif self.cfg.objective_method == 'optics':
            cluster_labels = self.optics_cluster_segments(cluster_z)
        elif self.cfg.objective_method == 'gmm':
            cluster_labels = self.gmm_cluster_segments(cluster_z)

        # Save the cluster context data
        self.save_latent_context(cluster_labels)
        # Plot the latent space
        if self.niter % self.cfg.plot_interval == 0:
            self.plot_latent_space(cluster_z, cluster_labels)
        
        # Get the cluster labels for the segments
        seg_labels = cluster_labels[:self.nsegs]
        
        # Assign the outliers
        if self.cfg.objective_method in ['dbscan', 'optics']:
            outliers, cluster_labels = self.assign_density_outliers(cluster_z, cluster_labels)
            return outliers, seg_labels
        elif self.cfg.objective_method == 'gmm':
            outliers = self.assign_gmm_outliers(cluster_z, cluster_labels)
            return outliers, seg_labels
        else:
            return np.zeros(self.nsegs).astype(bool), seg_labels

    def ablation_cluster_segments(self) -> np.ndarray:
        seg_labels = [self.rng.integers(self.cfg.ablation_clusters) for _ in range(self.nsegs)]
        return np.array(seg_labels)

class DDWESettings(BaseSettings):
    # Output directory for the run.
    output_path: Path
    # Run machine learning; set to False for ablation.
    do_machine_learning: bool
    # File containing the point for targeting. Set to None for no target.
    target_point_path: Optional[Path] = None
    # Drive resampling with the target in latent space or the pcoord.
    sort_by: str
    # Pcoord approaches zero as the target is approached.
    pcoord_approaches_zero: bool
    # How many walkers to consider for splitting and merging in lof.
    lof_consider_for_resample: int
    # Maximum number of target seeking resamples. Effectively how many to split for lof. Merge twice as many.
    max_resamples: int
    # Lower limit on walker weight. If all of the walkers in 
    # num_trial_splits are below the limit, split/merge is skipped
    # that iteration
    split_weight_limit: float
    # Upper limit on walker weight. If all of the walkers in 
    # num_trial_splits exceed limit, split/merge is skipped
    # that iteration
    merge_weight_limit: float

    @classmethod
    def from_westpa_config(cls) -> 'DDWESettings':
        westpa_config = deepcopy(westpa.rc.config.get(['west', 'ddmd'], {}))
        # Remove other dictionaries from westpa_config
        for key in ['machine_learning', 'objective']:
            westpa_config.pop(key, None)

        return DDWESettings(**westpa_config)

class CustomDriver(DeepDriveMDDriver):
    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)
    
        self.cfg = DDWESettings.from_westpa_config()

        self.log_path = Path(f'{self.cfg.output_path}/westpa-ddmd-logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.datasets_path = Path(f'{self.log_path}/datasets')
        os.makedirs(self.datasets_path, exist_ok=True)
        self.synd_model = self.load_synd_model()
        self.rng = np.random.default_rng()
    
    def sort_df_lof(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the given DataFrame for the LOF method.

        Parameters:
            df (pd.DataFrame): The DataFrame to be sorted.

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        by_pcoord = df.sort_values(['pcoord'], ascending=[self.cfg.pcoord_approaches_zero])

        # Check if sorting by distance and pcoord is the same
        if self.target_point is not None:
            by_distance = df.sort_values(['distance'], ascending=[True])
            self.test_pcoord_vs_dist_sort(by_pcoord, by_distance)      
        if self.cfg.sort_by == "distance":
            return by_distance
        else:
            return by_pcoord

    def sort_df_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the given DataFrame for the density clustering.

        Parameters:
            df (pd.DataFrame): The DataFrame to be sorted.

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        # If outliers exist and there is a cluster column
        if 'outlier' in df.columns and 'ls_cluster' in df.columns:
            pcoord_keys, pcoord_order = ['outlier', 'pcoord'], [False, self.cfg.pcoord_approaches_zero]
            distance_keys, distance_order = ['outlier', 'distance'], [False, True]
        else:
            pcoord_keys, pcoord_order = ['pcoord'], [self.cfg.pcoord_approaches_zero]
            distance_keys, distance_order = ['distance'], [True]

        by_pcoord = df.sort_values(pcoord_keys, ascending=pcoord_order)
        # Check if sorting by distance and pcoord is the same
        if self.target_point is not None:
            by_distance = df.sort_values(distance_keys, ascending=distance_order)
            self.test_pcoord_vs_dist_sort(by_pcoord, by_distance)      
        if self.cfg.sort_by == "distance":
            return by_distance
        else:
            return by_pcoord

    def split_decider(self, cluster_df: pd.DataFrame, num_segs_for_splitting: int, num_resamples: int) -> None:
        """
        Determines the split motif for a given cluster based on the provided parameters.

        Args:
            cluster_df (pd.DataFrame): The DataFrame containing the cluster data.
            num_segs_for_splitting (int): The number of segments to split.
            num_resamples (int): The number of resamples.

        Returns:
            None

        """
        # All the possible combinations of numbers that sum up to the num_resamples_needed
        combos = []
        find_combinations(num_resamples, [], 1, combos)
        # Need to check there's enough walkers to use that particular scheme
        split_possible = []
        # For each possible split
        for x in combos:
            # If the number of walkers is greater than the number needed for that possible split
            if len(x) <= num_segs_for_splitting:
                # Add to the list of possible splits
                split_possible.append(x)
        # This is the chosen split motif for this cluster
        chosen_splits = sorted(split_possible[self.rng.integers(len(split_possible))], reverse=True)
        print(f'split choice: {chosen_splits}')
        # Get the inds of the segs
        sorted_segs = cluster_df.inds.values
        # For each of the chosen segs, split by the chosen value
        for idx, n_splits in enumerate(chosen_splits):
            print(f'idx: {idx}, n_splits: {n_splits}')
            # Find which segment we are splitting using the ind from the sorted_segs
            segment = self.segments[int(sorted_segs[idx])]
            # Split the segment
            self._split_by_data(self.bin, segment, int(n_splits + 1))

    def merge_decider(self, cluster_df: pd.DataFrame, num_segs_for_merging: int, num_resamples: int) -> None:
        """
        Determines the merge motif for a given cluster based on the number of segments available for merging.

        Args:
            cluster_df (pd.DataFrame): The DataFrame containing information about the cluster.
            num_segs_for_merging (int): The number of segments available for merging.
            num_resamples (int): The number of resamples needed for merging.

        Returns:
            None

        """
        # All the possible combinations of numbers that sum up to the num_resamples_needed
        combos = []
        find_combinations(num_resamples, [], 1, combos)
        merges_possible = []
        # Need to check there's enough walkers to use that particular merging scheme
        for x in combos:
            # If the number of walkers is greater than the number needed for that possible merge
            if np.sum(x + 1) <= num_segs_for_merging:
                # Add to the list of possible merges
                merges_possible.append(x + 1)

        # This is the chosen merge motif for this cluster
        chosen_merge = sorted(list(merges_possible[self.rng.integers(len(merges_possible))]), reverse=True)
        print(f'merge choice: {chosen_merge}')
        
        for n in chosen_merge:
            # Get the last n rows of the cluster_df
            rows = cluster_df.tail(n)
            # Get the inds of the segs
            merge_group = list(rows.inds.values)
            print(f'merge group: {merge_group}')
            # Append the merge to the list of all merges
            self._merge_by_data(self.bin, self.segments[merge_group])
            # Remove the sampled rows
            cluster_df = cluster_df.drop(rows.index)

    def _get_cluster_df(self, df: pd.DataFrame, id: int) -> pd.DataFrame:
        """
        Returns a subset of the input DataFrame containing only the rows where the 'ls_cluster' column matches the given id.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            id (int): The id of the cluster to filter by.

        Returns:
            pd.DataFrame: A DataFrame containing only the rows where 'ls_cluster' matches the given id.
        """
        return df[df['ls_cluster'] == id]
        
    def test_pcoord_vs_dist_sort(self, by_pcoord: pd.DataFrame, by_dist: pd.DataFrame) -> None:
        """
        Tests if the sorting by pcoord and distance is the same.

        Parameters:
            by_pcoord (pd.DataFrame): The DataFrame sorted by pcoord.
            by_dist (pd.DataFrame): The DataFrame sorted by distance.

        Returns:
            None
        """
        # Check if the indices are the same
        if list(by_dist.inds.values) == list(by_pcoord.inds.values):
            # Check if the distances are not the same
            if not np.all(by_dist.distance.values[0] == by_dist.distance.values):
                # Check if the pcoords are not the same
                if not np.all(by_dist.pcoord.values[0] == by_dist.pcoord.values):
                    # Check that the df is not one row
                    if not len(by_dist) == 1:
                        num_uni_distances = len(np.unique(by_dist.distance.values))
                        num_uni_pcoords = len(np.unique(by_dist.pcoord.values))
                        print(f"Sorting by distance and pcoord is the same! {num_uni_distances} unique distances and {num_uni_pcoords} unique pcoords.")

    def remove_overweight_segs(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes segments from the given DataFrame that have a weight greater than or equal to the merge weight limit.

        Parameters:
            cluster_df (pd.DataFrame): The DataFrame containing the segments to be filtered.

        Returns:
            pd.DataFrame: The filtered DataFrame with segments that have a weight less than the merge weight limit.
        """
        return cluster_df[cluster_df['weight'] < self.cfg.merge_weight_limit]
    
    def remove_underweight_segs(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes segments from the given DataFrame that have a weight less than the split weight limit.

        Parameters:
        - cluster_df (pd.DataFrame): The DataFrame containing the segments to be filtered.

        Returns:
        - pd.DataFrame: The filtered DataFrame with segments that have weight greater than the split weight limit.
        """
        return cluster_df[cluster_df['weight'] > self.cfg.split_weight_limit]

    def split_up_to(self, cluster_df: pd.DataFrame, num_resamples: int) -> None:
        """
        Split walkers so that a certain number are added to the cluster.

        Args:
            cluster_df (pd.DataFrame): The DataFrame containing information about the walkers.
            num_resamples (int): The number of resamples to perform during splitting.

        Returns:
            None
        """
        # Display walkers under the weight threshold
        removed_splits = cluster_df[cluster_df['weight'] <= self.cfg.split_weight_limit]
        # Display the walkers that are below the weight threshold
        if len(removed_splits) > 1:
            print("Removed these walkers from splitting")
            print(removed_splits) 
        # Filter out weights under the threshold 
        cluster_df = self.remove_underweight_segs(cluster_df)
        cluster_df = self.sort_df_cluster(cluster_df)
        # The number of walkers that have sufficient weight for splitting
        num_segs_for_splitting = len(cluster_df)
        # Test if there are enough walkers with sufficient weight to split
        if num_segs_for_splitting == 0:
            print(f"Walkers up for splitting have weights that are too small. Skipping split/merge on iteration {self.niter}...")
        else: # Splitting can happen!
            self.split_decider(cluster_df, num_segs_for_splitting, num_resamples)

    def merge_down_to(self, cluster_df: pd.DataFrame, num_resamples: int, min_segs_for_merging: int) -> None:
        """
        Merges segments in the cluster dataframe down to a specified number of segments.

        Args:
            cluster_df (pd.DataFrame): The cluster dataframe containing the segments.
            num_resamples (int): The number of resamples to perform during merging.
            min_segs_for_merging (int): The minimum number of segments required for merging.

        Returns:
            None
        """
        # Find the walkers with too much weight
        removed_merges = cluster_df[cluster_df['weight'] >= self.cfg.merge_weight_limit]
        # Display the walkers that are above the weight threshold
        if len(removed_merges) > 1:
            print("Removed these walkers from merging")
            print(removed_merges)

        # Filter out the walkers with too much weight
        cluster_df = self.remove_overweight_segs(cluster_df)
        cluster_df = self.sort_df_cluster(cluster_df)
        num_segs_for_merging = len(cluster_df)
        # Need a minimum number of walkers for merging
        if num_segs_for_merging < min_segs_for_merging:
            print(f"Walkers up for merging have weights that are too large. Skipping split/merge on iteration {self.niter}...")
        else: # Merging gets to happen!
            self.merge_decider(cluster_df, num_segs_for_merging, num_resamples)

    def _recreate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recreates a new DataFrame based on the provided DataFrame and segments currently 
        in the bin. Adjusts the weights and indices of the segments to match the bin.

        Parameters:
            df (pd.DataFrame): The original DataFrame.

        Returns:
            pd.DataFrame: The recreated DataFrame with adjusted weights and indices.
        """
        # Make a copy of the column names from the original df
        new_df = df.iloc[:0,:].copy()
        # Get the segments that were in the bin originally
        og_segments = self.segments
        # Get the segments that are in the bin now
        self.segments = self._get_segments_by_parent_id(self.bin)
        # Loop through the updated bins
        for seg_ind, segment in enumerate(self.segments):
            # Loop through the segments being used as index
            for next_seg_ind, next_seg in enumerate(og_segments):
                # If the parent ids match
                if segment.parent_id == next_seg.parent_id:
                    # Add the row that corresponds to the next segment
                    new_df = new_df.append(df.iloc[next_seg_ind])
                    # Reset index
                    new_df.reset_index(drop=True, inplace=True)
                    # Adjust the weight to the segment from the bin
                    new_df.at[seg_ind, 'weight'] = segment.weight 
                    break
        # Reset the inds 
        new_df['inds'] = np.arange(len(new_df))
        return new_df
    
    def resample_by_weight_in_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        for id in set(df['ls_cluster']):
            # Get just the walkers in this cluster
            cluster_df = self._get_cluster_df(df, id)
            cluster_df = self.sort_df_cluster(cluster_df)
            print(f"cluster_id: {id}")
            print(cluster_df)

            # Ideal weight splits + merges
            self._split_by_weight(cluster_df, self.ideal_segs_per_cluster)
            self._merge_by_weight(cluster_df, self.ideal_segs_per_cluster)

        # Regenerate the df
        df = self._recreate_df(df)
        print("After ideal weight split/merge")
        print(df)
        return df
    
    def adjust_counts_in_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        for id in set(df['ls_cluster']):
            # Get just the walkers in this cluster
            cluster_df = self._get_cluster_df(df, id)
            cluster_df = self.sort_df_cluster(cluster_df)
            print(f"cluster_id: {id}")
            print(cluster_df)
            # Total number of walkers in the cluster
            num_segs_in_cluster = len(cluster_df)
            # Correct number of walkers per cluster
            if num_segs_in_cluster == self.ideal_segs_per_cluster: 
                print("Already the correct number of walkers per cluster")
            else:
                # Number of resamples needed to bring the cluster to the set number of walkers per cluster
                num_resamples = abs(num_segs_in_cluster - self.ideal_segs_per_cluster)
                print(f"Number of resamples needed to hit the ideal number of segs in this cluster: {num_resamples}")
                # Need to split some walkers
                if num_segs_in_cluster < self.ideal_segs_per_cluster: 
                    self.split_up_to(cluster_df, num_resamples)
                # Need to merge some walkers
                else: 
                    # Minimum number of walkers needed for merging
                    min_segs_for_merging = num_segs_in_cluster - self.ideal_segs_per_cluster + 1
                    self.merge_down_to(cluster_df,  num_resamples, min_segs_for_merging)

        # Regenerate the df
        df = self._recreate_df(df)
        print("After adjusting count split/merge")
        print(df)
        return df

    def resample_for_target_in_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples the data for each cluster in the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.

        Returns:
            pd.DataFrame: The resampled DataFrame.

        """
        for id in set(df['ls_cluster']):
            # Get just the walkers in this cluster
            cluster_df = self._get_cluster_df(df, id)
            cluster_df = self.sort_df_cluster(cluster_df)
            print(f"cluster_id: {id}")
            print(cluster_df)
            og_len = len(cluster_df)
            # Remove walkers outside the thresholds
            cluster_df = self.remove_underweight_segs(cluster_df)
            cluster_df = self.remove_overweight_segs(cluster_df)
            cluster_df = self.sort_df_cluster(cluster_df)
            num_segs_in_cluster = len(cluster_df)
            if num_segs_in_cluster != og_len:
                print("After removing walkers outside thresholds")
                print(cluster_df)

            # Decide the number of resamples based on the number of segments in the cluster   
            if num_segs_in_cluster < 3: # Too few walkers in weight threshold to resample here
                print('Not enough walkers within the thresholds to split/merge in this cluster')
                num_resamples = 0
            elif num_segs_in_cluster >= self.cfg.max_resamples * 3:
                num_resamples = self.cfg.max_resamples
            else: # Determine resamples dynamically
                num_resamples = int(num_segs_in_cluster / 3)   

            print(f"Number of resamples based on the cluster length: {num_resamples}")
            if num_resamples != 0:
                # Run resampling for the cluster
                self.split_decider(cluster_df, num_resamples, num_resamples)
                self.merge_decider(cluster_df, 2 * num_resamples, num_resamples)

        # Regenerate the df
        df = self._recreate_df(df)
        print("After target seeking split/merge")
        print(df)
        return df
    
    def resample_with_clusters(self, df: pd.DataFrame, seg_labels: np.ndarray) -> pd.DataFrame:
        df["ls_cluster"] = seg_labels
        # Set of all the cluster ids
        cluster_ids = sorted(set(df.ls_cluster.values))
        print(f"Number of clusters that have segments in them currently: {len(cluster_ids)}")
        # Ideal number of walkers per cluster
        self.ideal_segs_per_cluster = int((len(df)) / len(cluster_ids))
        print(f"Ideal number of walkers per cluster: {self.ideal_segs_per_cluster}")
        print(df)

        print("Starting ideal weight split/merges")
        df = self.resample_by_weight_in_clusters(df)
        print("Starting adjust counts split/merges")
        df = self.adjust_counts_in_clusters(df)
        print("Starting target seeking split/merges")
        df = self.resample_for_target_in_clusters(df)

    def lof_resampler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples the data in the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.

        Returns:
            pd.DataFrame: The resampled DataFrame.

        """
        if 'outlier' in df.columns:
            # Sort the DataFrame by outliers
            df = df.sort_values('outlier', ascending=False)
        else:
            # Randomly shuffle the DataFrame
            df = df.sample(frac=1)

        # Top outliers are up for splitting
        split_df = df.head(self.cfg.lof_consider_for_resample)
        # Bottom outliers are up for merging
        merge_df = df.tail(self.cfg.lof_consider_for_resample)

        # Remove out of weight walkers
        print("Walkers up for splitting")
        print(split_df)
        # The number of walkers up for splitting
        og_len = len(split_df)
        # Remove walkers outside the threshold
        split_df = self.remove_underweight_segs(split_df)
        # The number of walkers up for splitting after removing walkers outside the threshold
        num_segs_to_split = len(split_df)
        # Check if the number of walkers has changed
        if num_segs_to_split != og_len:
            print("After removing walkers outside threshold")
            print(split_df)
        
        # Remove out of weight walkers
        print("Walkers up for merging")
        print(merge_df)
        # The number of walkers up for splitting
        og_len = len(merge_df)
        # Remove out of weight walkers
        merge_df = self.remove_overweight_segs(merge_df)
        # The number of walkers up for splitting after removing walkers outside the threshold
        num_segs_to_merge = len(merge_df)
        # Check if the number of walkers has changed
        if num_segs_to_merge != og_len:
            print("After removing walkers outside threshold")
            print(merge_df)

        split_df = self.sort_df_lof(split_df)
        merge_df = self.sort_df_lof(merge_df)

        if num_segs_to_merge > 2 * num_segs_to_split:
            num_possible_resamples = num_segs_to_split
        else:
            num_possible_resamples = int(num_segs_to_merge / 2)

        # Decide the number of resamples based on the number of segments in the cluster
        if num_possible_resamples == 0: # Too few walkers in weight threshold to resample here
            print('Not enough walkers within the thresholds to split/merge in this cluster')
            num_resamples = 0
        elif num_possible_resamples >= self.cfg.max_resamples:
            num_resamples = self.cfg.max_resamples
        else: # Determine resamples dynamically
            num_resamples = num_possible_resamples

        print(f"Number of resamples based on the cluster length: {num_resamples}")
        if num_resamples != 0:
            # Run resampling for the cluster
            self.split_decider(df, num_resamples, num_resamples)
            self.merge_decider(df, 2 * num_resamples, num_resamples)

        return df


    def run(self, bin: Bin, cur_segments: Sequence[Segment], next_segments: Sequence[Segment]) -> None:
        self.bin = bin
        self.segments = next_segments

        # Get data for sorting
        pcoords = np.concatenate(self.get_pcoords(cur_segments)[:, -1])
        try:
            final_state = self.get_auxdata(cur_segments, "state_indices")[:, -1]
        except KeyError:
            final_state = self.get_restart_auxdata("state_indices")[:, -1]
        
        weight = self.get_weights(cur_segments)[:]
        df = pd.DataFrame(
            {
                "inds": np.arange(self.nsegs),
                "pcoord": pcoords,
                "cluster_id": final_state,
                "weight": weight,
            }
        )
        if self.cfg.do_machine_learning:
            self.machine_learning_method = MachineLearningMethod(self.niter, log_path=self.log_path)
            
            # extract and format current iteration's data
            try:
                 dcoords = self.get_dcoords(cur_segments)
            except KeyError:
                dcoords = self.get_restart_dcoords()
            dcoords = np.concatenate(dcoords)
            # Use Jeremy's fancy function to get the dcoords for the previous iterations
            all_dcoords = self.get_prev_dcoords_training(next_segments, dcoords, self.machine_learning_method.cfg.lag_iterations)
            # Get the dcoords for the current iteration
            dcoords = all_dcoords[:self.nsegs]

            # If training on the fly, check if it's time to train the model
            if self.machine_learning_method.cfg.ml_mode == 'train':
                self.machine_learning_method.train(all_dcoords)
            
            if self.cfg.target_point_path is not None:
                # Get the target point representation
                self.target_point = self.machine_learning_method.get_target_point_rep(self.cfg.target_point_path)
            
            # Initialize the objective object with the target point
            self.objective = Objective(self.nsegs, self.niter, self.datasets_path, self.log_path, self.cfg.split_weight_limit, self.cfg.merge_weight_limit, self.target_point)
            # Load the cluster context
            self.objective.load_latent_context(dcoords, pcoords, weight)
            
            # Predict the latent space coordinates
            cluster_z = self.machine_learning_method.predict(self.objective.all_dcoords)
            print(f"Total number of points in the latent space (including past iterations): {len(cluster_z)}")
            seg_z = cluster_z[:self.nsegs]
            if self.cfg.target_point_path is not None:
                # Add the distance column
                df["distance"] = [self.objective.distance_function(z, self.target_point) for z in seg_z]
            if self.objective.cfg.objective_method != 'lof':
                # Cluster the segments
                outliers, seg_labels = self.objective.cluster_segments(cluster_z)
            else:
                # Run LOF on the full history of embeddings to assure coverage over past states
                outliers = self.objective.lof_function(cluster_z)

                # Add the cluster outliers to the DataFrame
            df["outlier"] = outliers
        else:
            # Initialize the objective objects without the target point
            self.objective = Objective(self.nsegs, self.niter, self.datasets_path, self.log_path)
            seg_labels = self.objective.ablation_cluster_segments()
        

        # Resample the segments
        if self.objective.cfg.objective_method == 'lof':
            self.lof_resampler(df)
        else:
            self.resample_with_clusters(df, seg_labels)


