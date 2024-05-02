import logging
import os
import operator
import time
import mdtraj
import westpa
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from os.path import expandvars
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Optional, List, Dict, Any, Generator, Union
from natsort import natsorted

from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import BayesianGaussianMixture

from westpa.core.binning import Bin
from westpa.core.we_driver import WEDriver
from westpa.core.h5io import tostr
from westpa.core.segment import Segment

from westpa_ddmd.config import BaseSettings
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from nani import KmeansNANI, compute_scores, extended_comparison

mpl.use('Agg')
log = logging.getLogger(__name__)


class DeepDriveMDDriver(WEDriver, ABC):
    def _process_args(self):
        float_class = ['split_weight_limit', 'merge_weight_limit', 
                       'dbscan_epsilon', 'contact_cutoff']
        int_class = ['update_interval', 'lag_iterations', 'kmeans_clusters',
                     'max_save_per_cluster', 'dbscan_min_samples',
                     'gmm_max_components']
                 
        self.cfg = westpa.rc.config.get(['west', 'ddmd'], {})
        self.cfg.update({'train_path': None, 'machine_learning_method': None})
        for key in self.cfg:
            if key in int_class:
                setattr(self, key, int(self.cfg[key]))
            elif key in float_class:
                setattr(self, key, float(self.cfg[key]))
            else:
                setattr(self, key, self.cfg[key])

    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)
        
        self.weight_split_threshold = 1.0
        self.weight_merge_cutoff = 1.0
        self._process_args()

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

    def _split_by_data(
        self, bin: Bin, segment : Segment, num_splits: int
    ) -> None:
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
        """Collect auxdata for restart from the previous iteration.

        Parameters
        ----------
       
        Returns
        -------
        npt.ArrayLike
        """
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
        self, bin_: Union[Bin, Generator["Walker", None, None]]  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.weight), np.object_)
        #return np.array([seg for seg in bin_])
    
    def _get_segments_by_parent_id(
        self, bin_: Union[Bin, Generator["Walker", None, None]]  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.parent_id), np.object_)
    
    def _get_segments_by_seg_id(
        self, bin_: Union[Bin, Generator["Walker", None, None]]  # noqa
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

                print("Num walkers going to the final adjust_counts", len([x for x in bin_]))

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

class MachineLearningMethod:
    def __init__(self, train_path: Path, base_training_data_path: Path, static_chk_path: Path, contact_cutoff: float) -> None:
        """Initialize the machine learning method.

        Parameters
        ----------
        train_path : Path
            The path to save the model and training data.
        base_training_data_path : Path
            The path to a set of pre-exisiting training data to help the model properly converge.
        """
        self.train_path = train_path
        self.base_training_data_path = base_training_data_path
        self.static_chk_path = static_chk_path
        self.contact_cutoff = contact_cutoff

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
        checkpoint_dir = self.save_path / "checkpoints"
        model_weight_path = natsorted(list(checkpoint_dir.glob("*.pt")))[-1]
        return model_weight_path

    def train(self, coords: np.ndarray) -> None:
        """Takes aligned data and trains a new model on it. Outputs are saved in fixed postions."""
        # Load the base training data with the shape (frames, atoms, atoms)
        base_coords = np.load(self.base_training_data_path)
        
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
        # Concatenate the coords from all the frames (frames, atoms, atoms)
        # coords = np.concatenate(coords)
        # Compute the contact maps
        contact_maps = self.compute_sparse_contact_map(coords)
        # Predict the latent space coordinates
        if self.static_chk_path is not None:
            z, *_ = self.autoencoder.predict(
                contact_maps, checkpoint=self.static_chk_path
            )
        else:
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
        contact_maps = np.array(distance_matrices) < self.contact_cutoff
        # Predict the target point representation
        target_point_rep = self.predict(contact_maps)
        return np.concatenate(target_point_rep)

class CustomDriver(DeepDriveMDDriver):
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

    def _process_mdance_args(self):
        float_class = []
        int_class = ['sieve', 'n_structures', 'percentage']
                 
        self.cfg = westpa.rc.config.get(['west', 'mdance'], {})
        self.cfg.update({})
        
        if 'n_clusters' in self.cfg:
            temp = self.cfg['n_clusters']
            if isinstance(temp, str):
                from ast import literal_eval

                self.cfg['n_clusters'] = literal_eval(temp)
                del temp

        if 'percentage' not in self.cfg:
            self.cfg.update({'percentage': 10})

        for key in self.cfg:
            if key in int_class:
                setattr(self, key, int(self.cfg[key]))
            elif key in float_class:
                setattr(self, key, float(self.cfg[key]))
            else:
                setattr(self, key, self.cfg[key])

    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)

        self._process_args()
        self._process_mdance_args()

        self.base_training_data_path = expandvars('$WEST_SIM_ROOT/common_files/train.npy')

        self.log_path = Path(f'{self.output_path}/westpa-ddmd-logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.datasets_path = Path(f'{self.log_path}/datasets')
        os.makedirs(self.datasets_path, exist_ok=True)
        self.synd_model = self.load_synd_model()
        self.rng = np.random.default_rng()
        self.sort_df = self.sort_df_for_density
            
    def lof_function(self, z: np.ndarray) -> np.ndarray:
        # Load up to the last 50 of all the latent coordinates here
        if self.niter > self.lof_iteration_history:
            embedding_history = [
                np.load(self.datasets_path / f"z-{p}.npy")
                for p in range(self.niter - self.lof_iteration_history, self.niter)
            ]
        else:
            embedding_history = [np.load(p) for p in self.log_path.glob("z-*.npy")]

        # Append the most recent frames to the end of the history
        embedding_history.append(z)
        embedding_history = np.concatenate(embedding_history)

        # Run LOF on the full history of embeddings to assure coverage over past states
        clf = LocalOutlierFactor(n_neighbors=self.lof_n_neighbors).fit(
            embedding_history
        )
        # For selection of outliers, only consider the current segments
        current_nof = clf.negative_outlier_factor_[-len(z) :]
        # Reshape the outlier scores to correspond to their given segments
        nof = np.reshape(current_nof, (self.nsegs, self.nframes))

        # Take the final time point of each segment
        nof_per_segment = nof[:, -1]

        return nof_per_segment

    def plot_latent_space(self, cluster_z: np.ndarray, cluster_labels: np.ndarray, cluster_pcoords: np.ndarray, cluster_weights: np.ndarray) -> None:
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
            cluster_pcoords,
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
            cluster_weights,
            self.log_path / f"embedding-weight-{self.niter}.png",
            title=f"iter: {self.niter}",
            cb_label="weight",
            num_segs=self.nsegs,
            target_point=self.target_point if self.target_point is not None else None,
            min_max_color=(self.split_weight_limit, self.merge_weight_limit),
            log_scale=True,
        )
    
    def kmeans_cluster_segments(self, df: pd.DataFrame, cluster_z: np.ndarray) -> pd.DataFrame:
        # Perform the K-means clustering
        cluster_labels = KMeans(n_clusters=self.kmeans_clusters).fit(cluster_z).labels_
        # Save the cluster context data
        self.save_cluster_context(cluster_labels, self.cluster_dcoords, self.cluster_pcoords, self.cluster_weights)
        # Get the cluster labels for the segments
        seg_labels = cluster_labels[:self.nsegs]
        # Plot the latent space
        if self.niter % 10 == 0:
            self.plot_latent_space(cluster_z, cluster_labels, self.cluster_pcoords, self.cluster_weights)
        # Add the cluster labels to the DataFrame
        df["ls_cluster"] = seg_labels
        return df

    def knani_scan_cluster_segments(self, embedding_history, pcoord_history: np.ndarray) -> np.ndarray:
        # Perform the K-means clustering
        all_scores, all_models = [], []

        if self.init_type in ['comp_sim', 'div_select']:
            model = KmeansNANI(data=embedding_history, n_clusters=self.n_clusters[1], metric=self.metric, init_type=self.init_type, percentage=self.percentage)
            initiators = model.initiate_kmeans() 

        for i_cluster in range(self.n_clusters[0], self.n_clusters[1]+1):
            total = 0

            model = KmeansNANI(data=embedding_history, n_clusters=i_cluster, metric=self.metric, init_type=self.init_type, percentage=self.percentage)
            if self.init_type == 'vanilla_kmeans++':
                initiators = model.initiate_kmeans()
            elif self.init_type in ['comp_sim', 'div_select']:
                pass
            else:  # For k-means++, random
                initiators = self.init_type
            labels, centers, n_iter = model.kmeans_clustering(initiators=initiators)

            all_models.append([labels, centers])
            ch_score, db_score = compute_scores(embedding_history, labels=labels)

            dictionary = {}
            for j in range(i_cluster):
                dictionary[j] = embedding_history[np.where(labels == j)[0]]
            for val in dictionary.values():
                total += extended_comparison(np.asarray(val), traj_numpy_type='full', metric=self.metric)

            all_scores.append((i_cluster, n_iter, ch_score, db_score, total/i_cluster))


        all_scores = np.array(all_scores)

        if self.db_second:
            print(f'using second derivative of DBI to find optimal N')
            all_db = all_scores[:, 3]
            result = np.zeros((len(all_scores)-2, 2))

            for idx, i_cluster in enumerate(all_scores[1:-1, 0]):
                # Calculate the second derivative
                result[idx] =  [i_cluster, all_db[idx] + all_db[idx+2] - (2*all_db[idx+1])]

            chosen_idx = np.argmax(result[:,1])+1
            chosen_k = result[chosen_idx-1, 0]
        else:  # Pick only by using the lowest DBI
            chosen_idx = np.argmin(all_scores[:, 3])
            chosen_k = all_scores[chosen_idx, 0]

        header = f'init_type: {self.init_type}, percentage: {self.percentage}, metric: {self.metric}'
        header += 'Number of Clusters, Number of Iterations, Calinski-Harabasz score, Davies-Bouldin score, Average MSD'
        np.savetxt(self.log_path / f"{self.niter}-{self.percentage}{self.init_type}_summary.csv", all_scores, delimiter=',', header=header, fmt='%s')

        if self.niter % 10 == 0:
            plot_scatter(
                embedding_history,
                all_models[chosen_idx][0],
                self.log_path / f"embedding-cluster-{self.niter}.png",
            )
            plot_scatter(
                embedding_history,
                pcoord_history,
                self.log_path / f"embedding-pcoord-{self.niter}.png",
            )
        return all_models[chosen_idx][0], int(chosen_k)

    def knani_cluster_segments(self, embedding_history, pcoord_history: np.ndarray) -> np.ndarray:
        # Perform the K-means clustering
        NANI_labels, NANI_centers, NANI_n_iter = KmeansNANI(data=embedding_history, n_clusters=self.n_clusters, metric=self.metric, init_type=self.init_type, percentage=self.percentage).execute_kmeans_all()
        
        if self.niter % 10 == 0:
            plot_scatter(
                embedding_history,
                NANI_labels,
                self.log_path / f"embedding-cluster-{self.niter}.png",
            )
            plot_scatter(
                embedding_history,
                pcoord_history,
                self.log_path / f"embedding-pcoord-{self.niter}.png",
            )
        return NANI_labels


    def dbscan_cluster_segments(self, df: pd.DataFrame, cluster_z: np.ndarray) -> pd.DataFrame:
        """
        Cluster the segments using DBSCAN algorithm and assign cluster labels to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.
            cluster_z (np.ndarray): The latent space representation of the segments.

        Returns:
            pd.DataFrame: The DataFrame with cluster labels assigned to the segments.
        """
        # Cluster the segments
        cluster_labels = DBSCAN(min_samples=self.dbscan_min_samples, 
                                eps=self.dbscan_epsilon, 
                                metric=cosine_distance).fit(cluster_z).labels_
        print(f"Total number of clusters: {max(set(cluster_labels)) + 1}")
        # Save the cluster context data
        self.save_cluster_context(cluster_labels, self.cluster_dcoords, self.cluster_pcoords, self.cluster_weights)
        # Add the cluster labels to the DataFrame
        df["ls_cluster"] = cluster_labels[:self.nsegs]
        # Plot the latent space
        if self.niter % 10 == 0:
            self.plot_latent_space(cluster_z, cluster_labels, self.cluster_pcoords, self.cluster_weights)
        # Assign the outliers
        df = self.assign_density_outliers(df, cluster_z, cluster_labels)
        return df

    # TODO: Fix the outliers for gmm
    def gmm_cluster_segments(self, df: pd.DataFrame, cluster_z: np.ndarray) -> pd.DataFrame:
        """
        Cluster the segments using Gaussian Mixture Model and assign cluster labels to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.
            cluster_z (np.ndarray): The latent space representation of the segments.

        Returns:
            pd.DataFrame: The DataFrame with cluster labels assigned to the segments.
        """
        # Perform the GMM clustering
        gmm = BayesianGaussianMixture(n_components=self.gmm_clusters).fit(cluster_z)
        cluster_labels = gmm.predict(cluster_z)
        # Save the cluster context data
        self.save_cluster_context(cluster_labels, self.cluster_dcoords, self.cluster_pcoords, self.cluster_weights)
        # Get the cluster labels for the segments
        seg_labels = cluster_labels[:self.nsegs]
        # Plot the latent space
        if self.niter % 10 == 0:
            self.plot_latent_space(cluster_z, cluster_labels, self.cluster_pcoords, self.cluster_weights)
        # Add the cluster labels to the DataFrame
        df["ls_cluster"] = seg_labels
        # Assign the outliers
        df = self.assign_density_outliers(df, cluster_z, seg_labels, cluster_z[:self.nsegs])
        return df
    
    def sort_df_for_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the given DataFrame for the density clustering.

        Parameters:
            df (pd.DataFrame): The DataFrame to be sorted.

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        by_pcoord = df.sort_values(["outlier", 'pcoord'], ascending=[False, True])
        # Check if sorting by distance and pcoord is the same
        if self.target_point is not None:
            by_distance = df.sort_values(["outlier", 'distance'], ascending=[False, True])
            self.test_pcoord_vs_dist_sort(by_pcoord, by_distance)        
        if self.sort_by == "pcoord":
            return by_pcoord
        elif self.sort_by == "distance":
            return by_distance
        else:
            return by_pcoord

    def sort_df_for_gmm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the given DataFrame for the GMM clustering.

        Parameters:
            df (pd.DataFrame): The DataFrame to be sorted.

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        by_pcoord = df.sort_values(["outlier", 'pcoord'], ascending=[False, True])
        # Check if sorting by distance and pcoord is the same
        if self.target_point is not None:
            by_distance = df.sort_values(["outlier", 'distance'], ascending=[False, True])
            self.test_pcoord_vs_dist_sort(by_pcoord, by_distance)
        if self.sort_by == "pcoord":
            return by_pcoord
        elif self.sort_by == "distance":
            return by_distance
        else:
            return by_pcoord

    def assign_density_outliers(self, df: pd.DataFrame, cluster_z: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
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
        df["outlier"] = np.zeros(self.nsegs).astype(bool)
        seg_z = cluster_z[:self.nsegs]
        # if there are any outliers
        print(f"Number of outliers: {np.sum(cluster_labels[:self.nsegs] == -1)}")
        if -1 in df['ls_cluster'].values:
            # Set all the -1s to be outliers
            df['outlier'] = np.where(df['ls_cluster'] == -1, True, False)
            # Remove the outliers from the projections and labels
            inlier_embedding_history = cluster_z[cluster_labels != -1]
            inlier_labels = cluster_labels[cluster_labels != -1]
            # Loop through the outlier indices
            for ind in df[df['outlier']].inds.values:
                # Find the distance to points in the embedding_history
                dist = np.array([cosine_distance(seg_z[ind], z) for z in inlier_embedding_history])
                # Set zero dists to the max; should only happen if measuring the point to itself
                dist[dist == 0] = np.max(dist)
                # Find the min index
                min_ind = np.argmin(dist)
                # Set the ls_cluster for the outlier to match the min dist point
                df.loc[ind, 'ls_cluster'] = inlier_labels[min_ind]
        return df

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
        for ind, x in enumerate(combos):
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
        for ind, x in enumerate(combos):
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
            # Append the merge to the list of all merges
            self._merge_by_data(self.bin, self.segments[merge_group])
            # Remove the sampled rows
            cluster_df = cluster_df.drop(rows.index)

    def get_cluster_df(self, df: pd.DataFrame, id: int) -> pd.DataFrame:
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
        return cluster_df[cluster_df['weight'] < self.merge_weight_limit]
    
    def remove_underweight_segs(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes segments from the given DataFrame that have a weight less than the split weight limit.

        Parameters:
        - cluster_df (pd.DataFrame): The DataFrame containing the segments to be filtered.

        Returns:
        - pd.DataFrame: The filtered DataFrame with segments that have weight greater than the split weight limit.
        """
        return cluster_df[cluster_df['weight'] > self.split_weight_limit]

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
        removed_splits = cluster_df[cluster_df['weight'] <= self.split_weight_limit]
        # Display the walkers that are below the weight threshold
        if len(removed_splits) > 1:
            print("Removed these walkers from splitting")
            print(removed_splits) 
        # Filter out weights under the threshold 
        cluster_df = self.remove_underweight_segs(cluster_df).sort_values(["outlier", 'pcoord'], ascending=[False, True])
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
        removed_merges = cluster_df[cluster_df['weight'] >= self.merge_weight_limit]
        # Display the walkers that are above the weight threshold
        if len(removed_merges) > 1:
            print("Removed these walkers from merging")
            print(removed_merges)

        # Filter out the walkers with too much weight
        cluster_df = self.remove_overweight_segs(cluster_df)
        cluster_df = self.sort_df(cluster_df)
        num_segs_for_merging = len(cluster_df)
        # Need a minimum number of walkers for merging
        if num_segs_for_merging < min_segs_for_merging:
            print(f"Walkers up for merging have weights that are too large. Skipping split/merge on iteration {self.niter}...")
        else: # Merging gets to happen!
            self.merge_decider(cluster_df, num_segs_for_merging, num_resamples)

    def recreate_df(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def decide_num_resamples(self, num_segs_in_cluster: int) -> int:
        """
        Determines the number of resamples based on the number of segments in a cluster.

        Args:
            num_segs_in_cluster (int): The number of segments in the cluster.

        Returns:
            int: The number of resamples to be performed.
        """
        if num_segs_in_cluster < 3: # Too few walkers in weight threshold to resample here
            print('Not enough walkers within the thresholds to split/merge in this cluster')
            num_resamples = 0
        elif num_segs_in_cluster >= 12: # Would have more than 4 resamples
            # Split top 4 merge bottom 8
            num_resamples = 4
        else: # Determine resamples dynamically
            num_resamples = int(num_segs_in_cluster / 3)
        return num_resamples
    
    def save_cluster_context(self, cluster_labels: np.ndarray, cluster_dcoords: np.ndarray, cluster_pcoords: np.ndarray, cluster_weights: np.ndarray) -> None:
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
        # Loop through each cluster label and randomly select up to 50 indices to save
        for label in set(cluster_labels):
            indices = np.where(cluster_labels == label)[0]
            if label != -1 and len(indices) > self.max_save_per_cluster:
                indices = self.rng.choice(indices, self.max_save_per_cluster, replace=False)
            dcoords_to_save.append(cluster_dcoords[indices])
            pcoords_to_save.append(cluster_pcoords[indices])
            weights_to_save.append(cluster_weights[indices])
        dcoords_to_save = [x for y in dcoords_to_save for x in y]
        pcoords_to_save = [x for y in pcoords_to_save for x in y]
        weights_to_save = [x for y in weights_to_save for x in y]
        # Index the dcoord and pcoord arrays to save the selected indices
        np.save(self.datasets_path / f"cluster-context-dcoords-{self.niter}.npy", dcoords_to_save)
        np.save(self.datasets_path / f"cluster-context-pcoords-{self.niter}.npy", pcoords_to_save)
        np.save(self.datasets_path / f"cluster-context-weights-{self.niter}.npy", weights_to_save)

    def load_cluster_context(self, dcoords, pcoords, weight) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            past_dcoords = np.load(self.datasets_path / f"cluster-context-dcoords-{self.niter - 1}.npy")
            past_pcoords = np.load(self.datasets_path / f"cluster-context-pcoords-{self.niter - 1}.npy")
            past_weights = np.load(self.datasets_path / f"cluster-context-weights-{self.niter - 1}.npy")
            self.cluster_dcoords = np.concatenate([dcoords, past_dcoords])
            self.cluster_pcoords = np.concatenate([pcoords, past_pcoords])
            self.cluster_weights = np.concatenate([weight, past_weights])
        else:
            self.cluster_dcoords = dcoords
            self.cluster_pcoords = pcoords
            self.cluster_weights = weight
    
    def resample_by_weight_in_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        for id in set(df['ls_cluster']):
            # Get just the walkers in this cluster
            cluster_df = self.get_cluster_df(df, id)
            cluster_df = self.sort_df(cluster_df)
            print(f"cluster_id: {id}")
            print(cluster_df)

            # Ideal weight splits + merges
            self._split_by_weight(cluster_df, self.ideal_segs_per_cluster)
            self._merge_by_weight(cluster_df, self.ideal_segs_per_cluster)

        # Regenerate the df
        df = self.recreate_df(df)
        print("After ideal weight split/merge")
        print(df)
        return df
    
    def adjust_counts_in_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        for id in set(df['ls_cluster']):
            # Get just the walkers in this cluster
            cluster_df = self.get_cluster_df(df, id)
            cluster_df = self.sort_df(cluster_df)
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
        df = self.recreate_df(df)
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
            cluster_df = self.get_cluster_df(df, id)
            cluster_df = self.sort_df(cluster_df)
            print(f"cluster_id: {id}")
            print(cluster_df)
            og_len = len(cluster_df)
            # Remove walkers outside the thresholds
            cluster_df = self.remove_underweight_segs(cluster_df)
            cluster_df = self.remove_overweight_segs(cluster_df)
            cluster_df = self.sort_df(cluster_df)
            num_segs_in_cluster = len(cluster_df)
            if num_segs_in_cluster != og_len:
                print("After removing walkers outside thresholds")
                print(cluster_df)
            # Decide the number of resamples based on the number of segments in the cluster
            num_resamples = self.decide_num_resamples(num_segs_in_cluster)            
            print(f"Number of resamples based on the cluster length: {num_resamples}")
            if num_resamples != 0:
                # Run resampling for the cluster
                self.split_decider(cluster_df, num_resamples, num_resamples)
                self.merge_decider(cluster_df, 2 * num_resamples, num_resamples)

        # Regenerate the df
        df = self.recreate_df(df)
        print("After target seeking split/merge")
        print(df)
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
        if self.ml_mode != 'ablation':
            if self.ml_mode == 'train':
                # Determine the location for the training data/model
                if self.niter < self.update_interval:
                    self.train_path = self.log_path / "ml-iter-1"
                else:
                    self.train_path = (
                        self.log_path
                        / f"ml-iter-{self.niter - self.niter % self.update_interval}"
                    )
                # Init the ML method
                self.train_path.mkdir(exist_ok=True)
            self.machine_learning_method = MachineLearningMethod(
                self.train_path, self.base_training_data_path, self.static_chk_path,
                self.contact_cutoff,
            )
            # extract and format current iteration's data
            try:
                 dcoords = self.get_dcoords(cur_segments)
            except KeyError:
                dcoords = self.get_restart_dcoords()
            dcoords = np.concatenate(dcoords)
            # Use Jeremy's fancy function to get the dcoords for the previous iterations
            all_dcoords = self.get_prev_dcoords_training(next_segments, dcoords, self.lag_iterations)
            # Get the dcoords for the current iteration
            dcoords = all_dcoords[:self.nsegs]

            # If training on the fly, check if it's time to train the model
            if self.ml_mode == 'train':
                if self.niter == 1 or self.niter % self.update_interval == 0:
                    print(f"Training a model on iteration {self.niter}...")
                    self.machine_learning_method.train(all_dcoords)
            
            # Load the cluster context
            self.load_cluster_context(dcoords, pcoords, weight)
            
            # Predict the latent space coordinates
            cluster_z = self.machine_learning_method.predict(self.cluster_dcoords)
            print(f"Total number of points in the latent space (including past iterations): {len(cluster_z)}")
            seg_z = cluster_z[:self.nsegs]
            if self.target_point_path is not None:
                # Get the target point representation
                self.target_point = self.machine_learning_method.get_target_point_rep(self.target_point_path)
                # Add the distance column
                df["distance"] = [cosine_distance(z, self.target_point) for z in seg_z]
            
        # Cluster the segments
        if self.cluster_method == 'kmeans':
            df = self.kmeans_cluster_segments(df, cluster_z)
        elif self.cluster_method == 'dbscan':
            df = self.dbscan_cluster_segments(df, cluster_z)
        elif self.cluster_method == 'optics':
            df = self.optics_cluster_segments(df, cluster_z)
        elif self.cluster_method == 'gmm':
            df = self.gmm_cluster_segments(df, cluster_z)
        elif self.ml_mode == 'ablation':
            seg_labels = [self.rng.integers(8) for _ in range(self.nsegs)]
            df["ls_cluster"] = seg_labels

        # TODO: THIS IS OLD.
        embedding_history, pcoord_history = self.get_data_for_objective(z, pcoords)
        if self.knani:
            if isinstance(self.n_clusters, tuple):
                all_labels, output_n_clusters = self.knani_scan_cluster_segments(embedding_history, pcoord_history)
                print(f'scanned {self.n_clusters} and used {output_n_clusters}')
            else:
                all_labels = self.knani_cluster_segments(embedding_history, pcoord_history)
                print(f"running k-nani with {self.n_clusters}")
        else:
            all_labels = self.optics_cluster_segments(embedding_history, pcoord_history)
            print("running optics")

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
  

