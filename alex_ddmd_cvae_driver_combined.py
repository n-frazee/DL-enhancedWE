import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Optional, List, Dict, Any, Generator, Union
import time
import numpy as np
import numpy.typing as npt
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
from westpa.core.segment import Segment
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from itertools import product
from os.path import expandvars

import os
from westpa_ddmd.driver import DeepDriveMDDriver
from westpa_ddmd.config import BaseSettings, mkdir_validator

import westpa
from westpa.core.binning import Bin
from westpa.core.segment import Segment
from westpa.core.we_driver import WEDriver
import operator

log = logging.getLogger(__name__)


class LS_Cluster_Driver(WEDriver):
    def _process_args(self):
        float_class = ['split_weight_limit', 'merge_weight_limit']
        int_class = ['update_interval', 'lag_iterations', 'lof_n_neighbors', 
                     'lof_iteration_history', 'num_we_splits', 'num_trial_splits']
                 
        self.cfg = westpa.rc.config.get(['west', 'ddmd'], {})
        self.cfg.update({'train_path': None, 'machine_learning_method': None})
        for key in self.cfg:
            if key in int_class:
                setattr(self, key, int(self.cfg[key]))
            elif key in float_class:
                setattr(self, key, float(self.cfg[key]))
            else:
                setattr(self, key, self.cfg[key])

        if 'map_location' not in self.cfg.keys():
            self.map_location = 'cpu'

        self.CVAESettings = westpa.rc.config.get(['mdlearn', 'cvae'], {})

        if 'device' not in self.CVAESettings.keys():
            self.CVAESettings.update({'device': 'cpu'})

        if 'prefetch_factor' not in self.CVAESettings.keys():
            self.CVAESettings.update({'prefetch_factor': None})


    def load_synd_model(self):
        from synd.core import load_model
        import pickle

        subgroup_args = westpa.rc.config.get(['west', 'drivers', 'subgroup_arguments'])
        synd_model_path = expandvars(subgroup_args['synd_model'])
        backmap_path = expandvars(subgroup_args['dmatrix_map'])

        synd_model = load_model(synd_model_path)

        with open(backmap_path, 'rb') as infile:
            dmatrix_map = pickle.load(infile)

        synd_model.add_backmapper(dmatrix_map.get, name='dmatrix')

        return synd_model


    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)
        
        self._process_args()

        westpa.rc.pstatus(self.CVAESettings)

        self.niter: int = 0
        self.nsegs: int = 0
        self.nframes: int = 0
        self.cur_pcoords: npt.ArrayLike = []
        self.rng = np.random.default_rng()
        temp = np.asarray(list(product(range(self.num_we_splits+1), repeat=self.num_we_splits)), dtype=int)
        self.split_possible = temp[np.sum(temp, axis=1) == self.num_we_splits]
        self.split_total = sum(self.split_possible)
        self.autoencoder = SymmetricConv2dVAETrainer(**self.CVAESettings)
        self.autoencoder._load_checkpoint(expandvars(rc.config.get(['west','ddmd', 'checkpoint_model'])), map_location=self.map_location)

        self.synd_model = self.load_synd_model()

        del temp

        # Note: Several of the getter methods that return npt.ArrayLike
        # objects use a [:, 1:] index trick to avoid the initial frame
        # in westpa segments which corresponds to the last frame of the
        # previous iterations segment which is helpful for rate-constant
        # calculations, but not helpful for DeepDriveMD.

    def get_ibstates_ref(self) -> npt.ArrayLike:
        """Collect ibstates from information.

        Returns
        -------
        npt.ArrayLike
            Coordinates with shape (N, Nsegments, Nframes, Natoms, Natoms)
        """
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

        # TODO: If this function is slow, we can try direct h5 reads

        back_coords = []
        with data_manager.lock:
            iter_group = data_manager.get_iter_group(i)
            coords_raw = iter_group["auxdata/dmatrix"][:]

        return back_coords

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
        """Collect coordinates for restart from current iteration.
        Returns
        -------
        npt.ArrayLike
            Coordinates with shape (N, Nsegments, Nframes, Natoms, Natoms)
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

    def get_dcoords(self, segments: Sequence[Segment]) -> npt.ArrayLike:
        """Concatenate the coordinates frames from each segment."""
        dcoords = np.array(list(seg.data["dmatrix"] for seg in segments))
        return dcoords[:,1:]
        #return rcoords.reshape(self.nsegs, self.nframes + 1, -1, 3)[:, 1:]
        # return rcoords.reshape(self.nsegs, self.nframes, -1, 3)[:, 1:]

    def _run_we(self):
        '''Run recycle/split/merge. Do not call this function directly; instead, use
        populate_initial(), rebin_current(), or construct_next().'''
        self._recycle_walkers()

        # sanity check
        self._check_pre()

        # Regardless of current particle count, always split overweight particles and merge underweight particles
        # Then and only then adjust for correct particle count
        total_number_of_subgroups = 0
        total_number_of_particles = 0

        for ibin, bin in enumerate(self.next_iter_binning):
            if len(bin) == 0:
                continue

            for e in bin:
                self.niter = e.n_iter
                break

            # Splits the bin into subgroups as defined by the called function
            target_count = self.bin_target_counts[ibin]
            subgroups = self.subgroup_function(self, ibin, n_iter=self.niter, **self.subgroup_function_kwargs)
            total_number_of_subgroups += len(subgroups)
            # Clear the bin
            segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
            weights = np.array(list(map(operator.attrgetter('weight'), segments)))
            ideal_weight = weights.sum() / target_count
            bin.clear()
            # Determines to see whether we have more sub bins than we have target walkers in a bin (or equal to), and then uses
            # different logic to deal with those cases.  Should devolve to the Huber/Kim algorithm in the case of few subgroups.
            if len(subgroups) >= target_count:
                for i in subgroups:
                    # Merges all members of set i.  Checks to see whether there are any to merge.
                    if len(i) > 1:
                        (segment, parent) = self._merge_walkers(
                            list(i),
                            np.add.accumulate(np.array(list(map(operator.attrgetter('weight'), i)))),
                            i,
                        )
                        i.clear()
                        i.add(segment)
                    # Add all members of the set i to the bin.  This keeps the bins in sync for the adjustment step.
                    bin.update(i)

                if len(subgroups) > target_count:
                    self._adjust_count(bin, subgroups, target_count)

            if len(subgroups) < target_count:
                for i in subgroups:
                    self._split_by_weight(i, target_count, ideal_weight)
                    self._merge_by_weight(i, target_count, ideal_weight)
                    # Same logic here.
                    bin.update(i)
                if self.do_adjust_counts:
                    # A modified adjustment routine is necessary to ensure we don't unnecessarily destroy trajectory pathways.
                    self._adjust_count(bin, subgroups, target_count)
            if self.do_thresholds:
                for i in subgroups:
                    self._split_by_threshold(bin, i)
                    self._merge_by_threshold(bin, i)
                for iseg in bin:
                    if iseg.weight > self.largest_allowed_weight or iseg.weight < self.smallest_allowed_weight:
                        log.warning(
                            f'Unable to fulfill threshold conditions for {iseg}. The given threshold range is likely too small.'
                        )
            total_number_of_particles += len(bin)
        log.debug('Total number of subgroups: {!r}'.format(total_number_of_subgroups))

        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))


class DeepDriveMDDriver(WEDriver, ABC):
    def _process_args(self):
        float_class = ['split_weight_limit', 'merge_weight_limit']
        int_class = ['update_interval', 'lag_iterations', 'lof_n_neighbors', 
                     'lof_iteration_history', 'num_we_splits', 'num_trial_splits']
                 
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
        
        self._process_args()

        self.niter: int = 0
        self.nsegs: int = 0
        self.nframes: int = 0
        self.cur_pcoords: npt.ArrayLike = []
        self.rng = np.random.default_rng()
        temp = np.asarray(list(product(range(self.num_we_splits+1), repeat=self.num_we_splits)), dtype=int)
        self.split_possible = temp[np.sum(temp, axis=1) == self.num_we_splits]
        self.split_total = sum(self.split_possible)
        del temp

        # Note: Several of the getter methods that return npt.ArrayLike
        # objects use a [:, 1:] index trick to avoid the initial frame
        # in westpa segments which corresponds to the last frame of the
        # previous iterations segment which is helpful for rate-constant
        # calculations, but not helpful for DeepDriveMD.

    @abstractmethod
    def run(self, segments: Sequence[Segment]) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Implement the DeepDriveMD routine.

        Returns
        -------
        npt.ArrayLike
            Array of integer indices for which segments to split.
        npt.ArrayLike
            Array of integer indices for which segments to merge.
        """
        ...

    def _split_by_data(
        self, bin: Bin, to_split: Sequence[Segment], split_into: int
    ) -> None:
        
        chosen_pick = self.split_possible[self.rng.integers(self.split_total)][0]
        print(f'split choice: {chosen_pick}')
        
        for segments, split_into_custom  in zip(to_split, chosen_pick):
            bin.remove(segments)
            new_segments_list = self._split_walker(segments, split_into_custom+1, bin)
            bin.update(new_segments_list)         

    def _merge_by_data(self, bin: Bin, to_merge: Sequence[Segment]) -> None:
        bin.difference_update(to_merge)
        new_segment, parent = self._merge_walkers(to_merge, None, bin)
        bin.add(new_segment)

    def get_prev_dcoords(self, iterations: int) -> npt.ArrayLike:
        """Collect coordinates from previous iterations.

        Parameters
        ----------
        iterations : int
            Number of previous iterations to collect.

        Returns
        -------
        npt.ArrayLike
            Coordinates with shape (N, Nsegments, Nframes, Natoms, Natoms)
        """
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

        # TODO: If this function is slow, we can try direct h5 reads

        back_coords = []
        with data_manager.lock:
            for i in range(self.niter - iterations, self.niter):
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
            Coordinates with shape (N, Nsegments, Nframes, Natoms, Natoms)
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
        """Concatenate the coordinates frames from each segment."""
        dcoords = np.array(list(seg.data["dmatrix"] for seg in segments))
        return dcoords[:,1:]
        #return rcoords.reshape(self.nsegs, self.nframes + 1, -1, 3)[:, 1:]
        # return rcoords.reshape(self.nsegs, self.nframes, -1, 3)[:, 1:]

    def get_pcoords(self, segments: Sequence[Segment]) -> npt.ArrayLike:
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

    def _get_segments(
        self, bin_: Union[Bin, Generator["Walker", None, None]]  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.weight), np.object_)

    def _run_we(self) -> None:
        """Run recycle/split/merge. Do not call this function directly; instead, use
        populate_initial(), rebin_current(), or construct_next()."""

        # Recycle walkers that have reached the target state
        self._recycle_walkers()

        # Sanity check
        self._check_pre()

        # Resample
        for bin_ in self.next_iter_binning:
            if len(bin_) == 0:
                continue

            # This will allow you to get the pcoords for all frames which is necessary for DDMD
            # segments required for splitting and merging logic
            segments = self._get_segments(bin_)
            cur_segments = self._get_segments(self.current_iter_segments)
            # print(cur_segments)
            # print(cur_segments[0].data)
            # print(self.get_rcoords(cur_segments))
            # print(self.get_weights(cur_segments))
            self.cur_pcoords = self.get_pcoords(cur_segments)

            # TODO: Is there a way to get nsegs and nframes without pcoords?
            # If so, we could save on a lookup. Can we retrive it from cur_segments?
            self.niter = cur_segments[0].n_iter
            self.nsegs = self.cur_pcoords.shape[0]
            self.nframes = self.cur_pcoords.shape[1]

            # This checks for initializing; if niter is 0 then skip resampling
            if self.niter:
                to_split_inds, merge_groups_inds = self.run(cur_segments)
                #print(merge_groups_inds)
                
                if merge_groups_inds is not None:
                    check = [len(i) for i in merge_groups_inds]
                    
                if to_split_inds is not None and merge_groups_inds is not None and np.max(check) <= self.num_we_splits +1 and np.min(check) > 0:
                    to_split = np.array([segments[to_split_inds]])[0]

                    # Split walker with largest scaled diff
                    
                    self._split_by_data(bin_, to_split, split_into=2)

                    skip_merged = []
                    for k_id, to_merge_inds in enumerate(merge_groups_inds):
                        if len(to_merge_inds) > 1:
                            to_merge = segments[to_merge_inds]
                            self._merge_by_data(bin_, to_merge)
                        else:
                            skip_merged.append(k_id)
                    print(f'skipped merging for kmeans clusters {skip_merged}')
                else:
                    print(f'skipped due to kmeans')

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
    outlier_inds: Optional[np.ndarray] = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ff = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color)
    plt.colorbar(ff)
    if outlier_inds is not None:
        ax.scatter(
            data[outlier_inds, 0],
            data[outlier_inds, 1],
            data[outlier_inds, 2],
            color="k",
        )
    plt.savefig(output_path)


class ExperimentSettings(BaseSettings):
    output_path: Path
    """Output directory for the run."""
    update_interval: int
    """How often to update the model. Set to 0 for static."""
    lag_iterations: int
    """Number of lagging iterations to use for training data."""
    lof_n_neighbors: int
    """Number of neigbors to use for local outlier factor."""
    lof_iteration_history: int
    """Number of iterations to look back at for local outlier factor."""
    num_we_splits: int
    """Number of westpa splits to prioritize outliers with.
    num_we_merges gets implicitly set to num_we_splits + 1."""
    num_trial_splits: int
    """The top number of outlier segments that will be further
    filtered by some biophysical observable. Must satisify
    num_trial_splits >= num_we_splits + 1 """
    split_weight_limit: float
    """Lower limit on walker weight. If all of the walkers in 
    num_trial_splits are below the limit, split/merge is skipped
    that iteration"""
    merge_weight_limit: float
    """Upper limit on walker weight. If all of the walkers in 
    num_trial_splits exceed limit, split/merge is skipped
    that iteration"""
    # TODO: Add validator for this num_trial_splits condition
    contact_cutoff: float = 8.0
    """The Angstrom cutoff for contact map generation."""

    # validators
    _mkdir_output_path = mkdir_validator("output_path")


class MachineLearningMethod:
    def _process_args(self):
        float_class = ['lambda_rec']
        int_class = ['latent_dim', 'num_data_workers','batch_size', 'epochs', 
                     'checkpoint_log_every', 'plot_log_every', 'plot_n_samples', 'prefetch_factor']
                 
        self.cfg = westpa.rc.config.get(['mdlearn', 'cvae'], {})
        self.cfg.update({'train_path': None, 'machine_learning_method': None})
        for key in self.cfg:
            if key in int_class:
                setattr(self, key, int(self.cfg[key]))
            elif key in float_class:
                setattr(self, key, float(self.cfg[key]))
            else:
                setattr(self, key, self.cfg[key])

        if device not in self.cfg:
            self.device = 'cpu'

    def __init__(self, train_path: Path, base_training_data_path: Path) -> None:
        """Initialize the machine learning method.

        Parameters
        ----------
        train_path : Path
            The path to save the model and training data.
        base_training_data_path : Path
            The path to a set of pre-exisiting training data to help the model properly converge.
        """
        self._process_args()

        self.train_path = train_path
        self.base_training_data_path = base_training_data_path

        # Initialize the model
        self.autoencoder = SymmetricConv2dVAETrainer(**self.cfg())

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
        return f'{self.train_path}/saves'

    @property
    def most_recent_checkpoint_path(self) -> Path:
        """Get the most recent model checkpoint."""
        checkpoint_dir = f'{self.save_path}/checkpoints'
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
        # print(contact_maps)
        start = time.time()
        # Train model
        self.autoencoder.fit(X=contact_maps, output_path=self.save_path)
        print(f"Training for {time.time() - start} seconds")

        # Save the loss curve
        pd.DataFrame(self.autoencoder.loss_curve_).plot().get_figure().savefig(
            f'{self.train_path}/model_loss_curve.png'
        )
        
        pd.DataFrame(self.autoencoder.loss_curve_).to_csv(f'{self.train_path}/loss.csv')

        z, *_ = self.autoencoder.predict(
            contact_maps, checkpoint=self.most_recent_checkpoint_path
        )
        np.save(f'{self.train_path}/z.npy', z[len(base_coords):])

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Predict the latent space coordinates for a set of coordinate frames."""
        # Concatenate the coords from all the frames (frames, atoms, atoms)
        coords = np.concatenate(coords)
        #print(self.autoencoder.model)
        # Compute the contact maps
        contact_maps = self.compute_sparse_contact_map(coords)
        # Predict the latent space coordinates
        z, *_ = self.autoencoder.predict(
            contact_maps, checkpoint=self.most_recent_checkpoint_path
        )

        return z


class CustomDriver(DeepDriveMDDriver):
    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)

        self._process_args()

        self.base_training_data_path = expandvars(f'$WEST_SIM_ROOT/common_files/train.npy')

        self.log_path = Path(f'{self.output_path}/westpa-ddmd-logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.datasets_path = Path(f'{self.log_path}/datasets')
        os.makedirs(self.datasets_path, exist_ok=True)

        #self.base_training_data_path = expand_vars(f'$WEST_SIM_ROOT/common_files/train.npy')
        # self.cfg = ExperimentSettings.from_yaml(CONFIG_PATH)
        #self.log_path = f'{self.cfg.output_path}/westpa-ddmd-logs'
        #self.log_path.mkdir(exist_ok=True)
        #self.machine_learning_method = None
        #self.train_path = None
        #self.datasets_path = f'{self.log_path}/datasets'
        #self.datasets_path.mkdir(exist_ok=True)

    def lof_function(self, z: np.ndarray) -> np.ndarray:
        # Load up to the last 50 of all the latent coordinates here
        if self.niter > self.lof_iteration_history:
            embedding_history = [
                np.load(f'{self.datasets_path}/z-{p}.npy')
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
    
    def plot_prev_data(self):
        # Plot the old latent space projections of the training data and iteration data colored by phi
        old_model_path = (f'{self.log_path}/ml-iter-{self.niter - self.update_interval}')
            
        training_z = np.load(f'{old_model_path}/z.npy')
                
        training_z = np.reshape(
            training_z, (-1, self.nframes, training_z.shape[1])
        )[:, -1]

        live_z = np.concatenate(
            [
                np.load(f'{self.datasets_path}/last-z-{iter}.npy')
                for iter in range(
                    self.niter - self.update_interval, self.niter
                )
            ]
        )
        
        z_data = np.concatenate((training_z, live_z))
        pcoord_data = np.concatenate(
            [
                np.load(f'{self.datasets_path}/pcoord-{iter}.npy')
                for iter in range(self.niter - int(len(z_data)/self.nsegs), self.niter)
            ]
        )

        state_data = np.concatenate(
            (0 * np.ones(len(training_z)), 1 * np.ones(len(live_z)))
        )
        plot_scatter(
            z_data,
            pcoord_data,
            f'{self.log_path}/embedding-pcoord-{self.niter}.png',
        )
        plot_scatter(
            z_data,
            state_data,
            f'{self.log_path}/embedding-state-{self.niter}.png',
        )
        return None

    def train_decider(self, all_coords: np.ndarray) -> None:
        if self.niter == 1 or self.niter % self.update_interval == 0:
            # Time to train a model
            if self.niter == 1:  # Training on first iteration data
                print("Training an initial model...")
                train_coords = np.concatenate(all_coords)

            else:  # Retraining time
                print("Training a model on iteration " + str(self.niter) + "...")
                if self.niter > self.update_interval:
                    self.plot_prev_data()
                if self.lag_iterations >= self.niter:
                    train_coords = np.concatenate(self.get_prev_dcoords(self.niter - 1))
                else:
                    train_coords = np.concatenate(
                        self.get_prev_dcoords(self.lag_iterations)
                    )

            self.machine_learning_method.train(train_coords)

    def run(self, segments: Sequence[Segment]) -> None:
        # Determine the location for the training data/model
        if self.niter < self.update_interval:
            self.train_path = f'{self.log_path}/ml-iter-1'
        else:
            self.train_path = (
                f'{self.log_path}/ml-iter-{self.niter - self.niter % self.update_interval}')

        # Init the ML method
        self.train_path.mkdir(exist_ok=True)
        self.machine_learning_method = MachineLearningMethod(
            self.train_path, self.base_training_data_path
        )

        # extract and format current iteration's data
        # additional audxata can be extracted in a similar manner
        try:
            all_coords = self.get_dcoords(segments)
        except KeyError:
            all_coords = self.get_restart_dcoords()

        # all_coords.shape=(segments, frames, atoms, xyz)
        np.save(f'{self.datasets_path}/coords-{self.niter}.npy', all_coords)

        # Train a new model if it's time
        self.train_decider(all_coords)

        # Regardless of training, predict
        z = self.machine_learning_method.predict(all_coords)
        nof_per_segment = self.lof_function(z)
        # Get data for sorting
        
        pcoord = np.concatenate(self.get_pcoords(segments)[:, -1])
        try:
            final_state = self.get_auxdata(segments, "state_indices")[:, -1]
        except KeyError:
            final_state = self.get_restart_auxdata("state_indices")[:, -1]
        
        weight = self.get_weights(segments)[:]
        df = pd.DataFrame(
            {
                "nof": nof_per_segment,
                "inds": np.arange(self.nsegs),
                "pcoord": pcoord,
                "cluster_id": final_state,
                "weight": weight,
            }
        ).sort_values("nof")    
        print(df)

        # Finally, sort the smallest lof scores by biophysical values
        split_df = (  # Outliers
            df.head(self.num_trial_splits)
        )
        removed_splits = split_df[split_df['weight'] <= self.split_weight_limit]
        if len(removed_splits) > 1:
            print("Removed these walkers from splitting")
            print(removed_splits)
                
        # Filter out weights above the threshold 
        split_df = split_df[split_df['weight'] > self.split_weight_limit]
        if len(split_df) < self.num_we_splits:
            print("Walkers up for splitting have weights that are too small. Skipping split/merge this iteration...")
            to_split_inds = None
        else:
            split_df = (  # Outliers
                split_df.sort_values("pcoord", ascending=True)
                .head(self.num_we_splits)
            )
            # Collect the outlier segment indices
            to_split_inds = split_df.inds.values
            
        # Take the inliers for merging, sorting them by
        merge_df = (  # Inliers
            df.tail(self.num_trial_splits)
        )

        removed_merges = merge_df[merge_df['weight'] >= self.merge_weight_limit]
        if len(removed_merges) > 1:
            print("Removed these walkers from merging")
            print(removed_merges)

        merge_df = merge_df[merge_df['weight'] < self.merge_weight_limit]
        if len(merge_df) < 2 * self.num_we_splits:
            print("Walkers up for merging have weights that are too large. Skipping split/merge this iteration...")
            merge_list = None
        else:
            merge_df = (
                merge_df.sort_values("pcoord", ascending=True)
                .tail(2 * self.num_we_splits)
            )

            kmeans = KMeans(n_clusters=self.num_we_splits)
            kmeans.fit(np.array(merge_df['pcoord']).reshape(-1, 1))
            merge_df['cluster'] = kmeans.labels_

            merge_list = []
            for n in range(self.num_we_splits):
                cluster_df = merge_df[merge_df['cluster'] == n]
                #if len(cluster_df) > 1:
                merge_list.append(cluster_df.inds.values)


        # Log dataframes
        print(f"\n{split_df}\n{merge_df}")
        df.to_csv(f'{self.datasets_path}/full-niter-{self.niter}.csv')
        split_df.to_csv(f'{self.datasets_path}/split-niter-{self.niter}.csv')
        merge_df.to_csv(f'{self.datasets_path}/merge-niter-{self.niter}.csv')

        # Log the machine learning outputs
        np.save(f'{self.datasets_path}/z-{self.niter}.npy', z)

        # Save data for plotting
        np.save(
            f'{self.datasets_path}/last-z-{self.niter}.npy',
            np.reshape(z, (self.nsegs, self.nframes, -1))[:, -1, :],
        )
        np.save(f'{self.datasets_path}/pcoord-{self.niter}.npy', pcoord)

        return to_split_inds, merge_list

