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

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, OPTICS, DBSCAN
from os.path import expandvars
import math

import os
from westpa_ddmd.config import BaseSettings, mkdir_validator
import operator

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
    def _process_args(self):
        float_class = ['split_weight_limit', 'merge_weight_limit']
        int_class = ['update_interval', 'lag_iterations', 'kmeans_clusters',
                     'kmeans_iteration_history']
                 
        self.cfg = westpa.rc.config.get(['west', 'ddmd'], {})
        self.cfg.update({'train_path': None, 'machine_learning_method': None})
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
        self, bin: Bin, segments: Sequence[Segment], split_dict: dict
    ) -> None:
        for ind in split_dict:
            segment = segments[ind]
            bin.remove(segment)
            new_segments_list = self._split_walker(segment, split_dict[ind], bin)
            bin.update(new_segments_list)

    def _merge_by_data(self, bin: Bin, to_merge: Sequence[Segment]) -> None:
        bin.difference_update(to_merge)
        new_segment, parent = self._merge_walkers(to_merge, None, bin)
        bin.add(new_segment)

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
                #print(iter_group)
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
                # print(f"{len(bin_)=}")
                self._adjust_count(ibin)
                # print(f"{len(bin_)=}")

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

                to_split_dict, merge_groups_inds = self.run(cur_segments, segments)
                
                if to_split_dict is not None and merge_groups_inds is not None:
                    
                    self._split_by_data(bin_, segments, to_split_dict)

                    for to_merge_inds in merge_groups_inds:
                        to_merge = segments[to_merge_inds]
                        self._merge_by_data(bin_, to_merge)
                else:
                    print('Too many walkers outside the weight thresholds!')
                
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
    ml_mode: str
    """static, ablation, train"""
    static_chk_path: Path
    """checkpoint file for static"""
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
    def __init__(self, train_path: Path, base_training_data_path: Path) -> None:
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

        # Load the configuration
        #self.cfg = ExperimentSettings.from_yaml(CONFIG_PATH)

        # Initialize the model
        self.autoencoder = SymmetricConv2dVAETrainer(**CVAESettings().dict())

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
        # print(f"{base_coords.shape=}")
        
        # Concatenate the base training data with the new data (frames, atoms, atoms)
        coords = np.concatenate((base_coords, coords))
        # print(f"{coords.shape=}")
        # Compute the contact maps
        contact_maps = self.compute_sparse_contact_map(coords)
        # print(contact_maps)
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

    def predict(self, coords: np.ndarray, static_chk_path=None) -> np.ndarray:
        """Predict the latent space coordinates for a set of coordinate frames."""
        # Concatenate the coords from all the frames (frames, atoms, atoms)
        # coords = np.concatenate(coords)
        #print(self.autoencoder.model)
        # Compute the contact maps
        contact_maps = self.compute_sparse_contact_map(coords)
        # with open("tmp.pkl", 'wb') as f:
        #     pickle.dump(contact_maps, f)
        #print(f"{contact_maps=}")
        # Predict the latent space coordinates
        if static_chk_path is not None:
            z, *_ = self.autoencoder.predict(
                contact_maps, checkpoint=static_chk_path
            )
        else:
            z, *_ = self.autoencoder.predict(
                contact_maps, checkpoint=self.most_recent_checkpoint_path
            )  

        return z

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

    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)

        self._process_args()

        self.base_training_data_path = expandvars('$WEST_SIM_ROOT/common_files/train.npy')

        self.log_path = Path(f'{self.output_path}/westpa-ddmd-logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.datasets_path = Path(f'{self.log_path}/datasets')
        os.makedirs(self.datasets_path, exist_ok=True)
        self.synd_model = self.load_synd_model()
        self.rng = np.random.default_rng()

    def get_data_for_objective(self, z: np.ndarray, pcoords: np.ndarray):
        if self.niter == 1:
            embedding_history = []
            pcoord_history = []
        elif self.niter > self.kmeans_iteration_history:
            embedding_history = [
                np.load(self.datasets_path / f"z-{p}.npy")
                for p in range(self.niter - self.kmeans_iteration_history, self.niter)
            ]
            pcoord_history = [
                np.load(self.datasets_path / f"pcoord-{p}.npy")
                for p in range(self.niter - self.kmeans_iteration_history, self.niter)
            ]
        else:
            embedding_history = [np.load(p) for p in self.datasets_path.glob("z-*.npy")]
            pcoord_history = [np.load(p) for p in self.datasets_path.glob("pcoord-*.npy")]
        # Append the most recent frames to the end of the history
        embedding_history.append(z)
        embedding_history = np.concatenate(embedding_history)

        pcoord_history.append(pcoords)
        pcoord_history = np.concatenate(pcoord_history)

        return embedding_history, pcoord_history
    
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

    def find_combinations(self, target, current_combination, start, combo_list):
        if target == 0:
            combo_list.append(np.array(current_combination))
            return
        if target < 0:
            return

        for i in range(start, target + 1):
            self.find_combinations(target - i, current_combination + [i], i, combo_list)

    def cluster_segments(self, z: np.ndarray, pcoords: np.ndarray) -> np.ndarray:
        # Load up to the old latent coordinates here
        embedding_history, pcoord_history = self.get_data_for_objective(z, pcoords)

        # Perform the K-means clustering
        kmeans = KMeans(n_clusters=self.kmeans_clusters).fit(embedding_history)
        seg_labels = kmeans.labels_[-len(z):]
        if self.niter % 10 == 0:
            plot_scatter(
                embedding_history,
                kmeans.labels_,
                self.log_path / f"embedding-cluster-{self.niter}.png",
            )
            plot_scatter(
                embedding_history,
                pcoord_history,
                self.log_path / f"embedding-pcoord-{self.niter}.png",
            )
        return seg_labels
    
    def dbscan_cluster_segments(self, embedding_history: np.ndarray, pcoord_history: np.ndarray) -> np.ndarray:
        # if self.niter == 1000 or self.niter == 1100 or self.niter == 1200 or self.niter == 1300 or self.niter == 1400:
        #     for x in range(1, 41):
        #         for y in range(1, 41):
        #             clustering = DBSCAN(eps=y*.025, min_samples=x*5).fit(embedding_history)
        #             plot_scatter(
        #                 embedding_history,
        #                 clustering.labels_,
        #                 self.log_path / f"embedding-cluster-{self.niter}-min-{x*5}-eps-{y*.05}.png",
        #             )
        #             print(x*5, y*.05, len(set(clustering.labels_)), list(clustering.labels_).count(-1)/len(clustering.labels_))
        
        if self.niter > self.kmeans_iteration_history:
            min_samples = 50
        else:
            min_samples = int(math.ceil(self.niter * 50 / self.kmeans_iteration_history))

        # Perform the K-means clustering
        clustering = DBSCAN(min_samples=min_samples, eps=1.5).fit(embedding_history)
        
        if self.niter % 10 == 0:
            plot_scatter(
                embedding_history,
                clustering.labels_,
                self.log_path / f"embedding-cluster-{self.niter}.png",
            )
            plot_scatter(
                embedding_history,
                pcoord_history,
                self.log_path / f"embedding-pcoord-{self.niter}.png",
            )
        return clustering.labels_

    def optics_cluster_segments(self, embedding_history: np.ndarray, pcoord_history: np.ndarray) -> np.ndarray:
        # Load up to the old latent coordinates here
        if self.niter > self.kmeans_iteration_history:
            min_samples = 135
        else:
            min_samples = int(math.ceil(self.niter * 1.35))

        # Perform the K-means clustering
        clustering = OPTICS(min_samples=min_samples).fit(embedding_history)
        
        if self.niter % 10 == 0:
            plot_scatter(
                embedding_history,
                clustering.labels_,
                self.log_path / f"embedding-cluster-{self.niter}.png",
            )
            plot_scatter(
                embedding_history,
                pcoord_history,
                self.log_path / f"embedding-pcoord-{self.niter}.png",
            )
            
            # exit()
        return clustering.labels_
    
    def plot_prev_data(self):
        # Plot the old latent space projections of the training data and iteration data colored by phi
        old_model_path = (
            self.log_path
            / f"ml-iter-{self.niter - self.update_interval}"
        )
        training_z = np.load(old_model_path / "z.npy")
                
        training_z = np.reshape(
            training_z, (-1, self.nframes, training_z.shape[1])
        )[:, -1]

        live_z = np.concatenate(
            [
                np.load(self.datasets_path / f"last-z-{iter}.npy")
                for iter in range(
                    self.niter - self.update_interval, self.niter
                )
            ]
        )
        
        z_data = np.concatenate((training_z, live_z))
        pcoord_data = np.concatenate(
            [
                np.load(self.datasets_path / f"pcoord-{iter}.npy")
                for iter in range(self.niter - int(len(z_data)/self.nsegs), self.niter)
            ]
        )

        state_data = np.concatenate(
            (0 * np.ones(len(training_z)), 1 * np.ones(len(live_z)))
        )
        plot_scatter(
            z_data,
            pcoord_data,
            self.log_path / f"embedding-pcoord-{self.niter}.png",
        )
        plot_scatter(
            z_data,
            state_data,
            self.log_path / f"embedding-state-{self.niter}.png",
        )
        return None

    def get_prev_dcoords_training(self, segments, curr_dcoords, iters_back):

        
        # print('func')
        # for seg in segments:
        #     print(seg)
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

    def train_decider(self, all_coords: np.ndarray) -> None:
        if self.niter == 1 or self.niter % self.update_interval == 0:
            # Time to train a model
            if self.niter == 1:  # Training on first iteration data
                print("Training an initial model...")

            else:  # Retraining time
                print(f"Training a model on iteration {self.niter}...")
                if self.niter > 2 * self.update_interval:
                    self.plot_prev_data()

            self.machine_learning_method.train(all_coords)
    
    def run(self, cur_segments: Sequence[Segment], next_segments) -> None:
        # Get data for sorting
        pcoords = np.concatenate(self.get_pcoords(cur_segments)[:, -1])
        try:
            final_state = self.get_auxdata(cur_segments, "state_indices")[:, -1]
        except KeyError:
            final_state = self.get_restart_auxdata("state_indices")[:, -1]
        
        weight = self.get_weights(cur_segments)[:]

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
                self.train_path, self.base_training_data_path
            )

            # extract and format current iteration's data
            # additional audxata can be extracted in a similar manner
            try:
                cur_dcoords = self.get_dcoords(cur_segments)
            except KeyError:
                cur_dcoords = self.get_restart_dcoords()

            # Concatenate all frames together
            cur_dcoords = np.concatenate(cur_dcoords)
            # print(f"{cur_dcoords.shape=}")

            # for x in range(len(cur_dcoords)-1):
            #     print(np.all(cur_dcoords[x] == cur_dcoords[x+1]))
            all_dcoords = self.get_prev_dcoords_training(next_segments, cur_dcoords, 10)

            # print(f"{all_dcoords.shape=}")
            cur_dcoords = all_dcoords[:self.nsegs]
            # print(f"{cur_dcoords.shape=}")
            # for x in range(len(cur_dcoords)-1):
            #     print(np.all(cur_dcoords[x] == cur_dcoords[x+1]))

            np.save(self.datasets_path / f"dcoords-{self.niter}.npy", cur_dcoords)

            if self.ml_mode == 'train':
                print("Train")
                # Train a new model if it's time
                self.train_decider(all_dcoords)
                # Regardless of training, predict
                z = self.machine_learning_method.predict(cur_dcoords)
            else:
                print("Static")
                # Regardless of training, predict
                z = self.machine_learning_method.predict(cur_dcoords, self.static_chk_path)
            
            embedding_history, pcoord_history = self.get_data_for_objective(z, pcoords)
            start = time.time()
        
            all_labels = self.dbscan_cluster_segments(embedding_history, pcoord_history)
            print(f"Clustered for {time.time() - start} seconds")
            seg_labels = all_labels[-self.nsegs:]


        else: # no ML of any kind
            print("Ablation")
            seg_labels = [self.rng.integers(self.kmeans_clusters) for _ in range(self.nsegs)]

        df = pd.DataFrame(
            {
                "ls_cluster": seg_labels,
                "outlier": np.zeros(self.nsegs).astype(bool),
                "inds": np.arange(self.nsegs),
                "pcoord": pcoords,
                "cluster_id": final_state,
                "weight": weight,
            }
        )   
        # print(df)
        # print(f"{z=}")
        # Dictionary that maps the index of segments to the number of splits they need
        split_dict = {}
        # List that contains lists of indices of segments that will be merged together
        merge_list =[]
        # Set of all the cluster ids from OPTICS
        cluster_ids = sorted(set(seg_labels))
        print(f"{cluster_ids=}")
        # if there are any outliers
        if -1 in cluster_ids:
            # Set all the -1s to be outliers
            df['outlier'] = np.where(df['ls_cluster'] == -1, True, False)
            # Remove the outliers from the projections and labels
            inlier_embedding_history = embedding_history[all_labels != -1]
            inlier_labels = all_labels[all_labels != -1]
            # print(f"{inlier_embedding_history.shape=}")
            # print(f"{inlier_embedding_history=}")
            # print(f"{inlier_labels.shape=}")
            # print(f"{inlier_labels=}")
            
            # Loop through the outlier indices
            for ind in df[df['outlier']].inds.values:
                # print(f"{ind=}")
                # Find the distance to points in the embedding_history
                dist = cdist(z[ind].reshape(-1, z.shape[1]), inlier_embedding_history)
                # print(f"{dist=}")
                # Set zero dists to the max; should only happen if measuring the point to itself
                dist[dist == 0] = np.max(dist)
                # print(f"{dist=}")
                # Find the min index
                min_ind = np.argmin(dist)
                # print(f"{min_ind=}")
                # Set the ls_cluster for the outlier to match the min dist point
                df.loc[ind, 'ls_cluster'] = inlier_labels[min_ind]
                # print(df)
        
        # Set of all the cluster ids from OPTICS
        cluster_ids = sorted(set(df.ls_cluster.values))
        print(f"{cluster_ids=}")
        
        # print("After adjusting outliers")
        print(df)
        # Ideal number of walkers per cluster
        segs_per_cluster = int(math.floor((len(df)) / len(cluster_ids)))

        # if self.nsegs > 32:
        #     segs_per_cluster = int(math.floor((len(df)) / len(cluster_ids)))
        # else:
        #     segs_per_cluster = int(math.ceil((len(df)) / len(cluster_ids)))

        print(f"{segs_per_cluster=}")
        
        for id in cluster_ids:
            # Get just the walkers in this cluster
            cluster_df = df[df['ls_cluster'] == id].sort_values(["outlier", 'pcoord'], ascending=[False, True])
            print(f"cluster_id: {id}")
            print(cluster_df)
            # Total number of walkers in the cluster
            num_segs_in_cluster = len(cluster_df)
            #print(f"{num_segs_in_cluster=}")
            if len(cluster_df) == segs_per_cluster: # correct number of walkers
                print("Already the correct number of walkers!")
                continue
            else:
                # Number of resamples needed to bring the cluster to the set number of walkers per cluster
                num_resamples_needed = abs(num_segs_in_cluster - segs_per_cluster)
                print(f"{num_resamples_needed=}")

                # All the possible combinations of numbers that sum up to the num_resamples_needed
                combos = []
                self.find_combinations(num_resamples_needed, [], 1, combos)

                if len(cluster_df) < segs_per_cluster: # need to split some walkers
                    # Display walkers under the weight threshold
                    removed_splits = cluster_df[cluster_df['weight'] <= self.split_weight_limit]
                    if len(removed_splits) > 1:
                        print("Removed these walkers from splitting")
                        print(removed_splits)
                            
                    # Filter out weights above the threshold 
                    cluster_df = cluster_df[cluster_df['weight'] > self.split_weight_limit].sort_values(["outlier", 'pcoord'], ascending=[False, True])

                    # The number of walkers that have sufficient weight for splitting
                    num_segs_for_splitting = len(cluster_df)
                    # Test if there are enough walkers with sufficient weight to split
                    if num_segs_for_splitting == 0:
                        print(f"Walkers up for splitting have weights that are too small. Skipping split/merge on iteration {self.niter}...")
                        split_dict = None
                        break
                    else: # Splitting can happen!
                        # Need to check there's enough walkers to use that particular merging scheme
                        split_possible = []
                        for ind, x in enumerate(combos):
                            if len(x) <= num_segs_for_splitting:
                                split_possible.append(x)
                        # This is the chosen split motif for this cluster
                        chosen_splits = sorted(split_possible[self.rng.integers(len(split_possible))], reverse=True)
                        print(f'split choice: {chosen_splits}')
                        # Check if there are any outliers associated with this cluster

                        sorted_segs = cluster_df.inds.values
                        print(f"{sorted_segs=}")

                        # Add to the split_dict with each key corresponding to the index of the walker 
                        # and the value being the number of splits
                        for idx, n_splits in enumerate(chosen_splits):
                            split_dict[int(sorted_segs[idx])] = int(n_splits + 1)

                        print(f"{split_dict=}")

                else: # need to merge some walkers
                    # Find the walkers with too much weight
                    removed_merges = cluster_df[cluster_df['weight'] >= self.merge_weight_limit]
                    if len(removed_merges) > 1:
                        print("Removed these walkers from merging")
                        print(removed_merges)

                    # Filter out the walkers with too much weight
                    cluster_df = cluster_df[cluster_df['weight'] < self.merge_weight_limit].sort_values(["outlier", 'pcoord'], ascending=[False, True])
                    num_segs_for_merging = len(cluster_df)
                    # Need a minimum number of walkers for merging
                    if num_segs_for_merging < num_segs_in_cluster - segs_per_cluster + 1:
                        print(f"Walkers up for merging have weights that are too large. Skipping split/merge on iteration {self.niter}...")
                        merge_list = None
                        break
                    else: # Merging gets to happen!
                        merges_possible = []
                        # Need to check there's enough walkers to use that particular merging scheme
                        for ind, x in enumerate(combos):
                            if np.sum(x + 1) <= num_segs_for_merging:
                                merges_possible.append(x + 1)

                        # print(f"{merges_possible=}")

                        # This is the chosen merge motif for this cluster
                        chosen_merge = sorted(list(merges_possible[self.rng.integers(len(merges_possible))]), reverse=True)
                        print(f'merge choice: {chosen_merge}')
                        #print(f"{cluster_df=}")
                        for n in chosen_merge:
                            rows = cluster_df.tail(n)
                            merge_group = list(rows.inds.values)
                            # Append the merge to the list of all merges
                            merge_list.append(merge_group)
                            print(f"{merge_group=}")
                            # Remove the sampled rows
                            cluster_df = cluster_df.drop(rows.index)
                        print(f"{merge_list=}")

        if self.ml_mode != "ablation":

            # Log dataframes
            df.to_csv(self.datasets_path / f"full-niter-{self.niter}.csv")

            # Log the machine learning outputs
            np.save(self.datasets_path / f"z-{self.niter}.npy", z)

            # Save data for plotting
            np.save(
                self.datasets_path / f"last-z-{self.niter}.npy",
                np.reshape(z, (self.nsegs, self.nframes, -1))[:, -1, :],
            )
            np.save(self.datasets_path / f"pcoord-{self.niter}.npy", pcoords)

        return split_dict, merge_list

