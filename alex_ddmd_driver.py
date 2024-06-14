import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
from natsort import natsorted

mpl.use("Agg")
import operator
import os
from copy import deepcopy
from os.path import expandvars

import matplotlib.pyplot as plt
import mdtraj
import westpa
from itertools import combinations_with_replacement
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from scipy.sparse import coo_matrix
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from westpa.core.binning import Bin
from westpa.core.h5io import tostr
from westpa.core.segment import Segment
from westpa.core.we_driver import WEDriver

from nani import KmeansNANI, compute_scores, extended_comparison
from westpa_ddmd.config import BaseSettings

log = logging.getLogger(__name__)


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - similarity


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.linalg.norm(v1 - v2)


def find_combinations(num_resamples: int, num_segs) -> List[np.ndarray]:
    combos = []
    for x in range(1, num_segs + 1):
        com = combinations_with_replacement(range(1, num_resamples + 1), x)
        for c in com:
            if sum(c) == num_resamples:
                combos.append(np.array(c))
    return combos


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
    def run(
        self, segments: Sequence[Segment]
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]: ...

    def _split_by_data(self, bin: Bin, segment: Segment, num_splits: int) -> None:
        bin.remove(segment)
        new_segments_list = self._split_walker(segment, num_splits, bin)
        bin.update(new_segments_list)

    def _merge_by_data(self, bin: Bin, to_merge: Sequence[Segment]) -> None:
        bin.difference_update(to_merge)
        new_segment, _ = self._merge_walkers(to_merge, None, bin)
        bin.add(new_segment)

    def _split_by_weight(self, df: pd.DataFrame, ideal_segs_per_cluster: int):
        ideal_weight = df["weight"].sum() / ideal_segs_per_cluster
        print(f"{ideal_weight=}")
        # Get all of the segments over the ideal weight
        to_split = df[df["weight"] > self.weight_split_threshold * ideal_weight]

        for _, row in to_split.iterrows():
            # Find the ind for the row to be split
            split_ind = int(row["inds"])
            # Determine the number of necessary splits to add
            m = int(row["weight"] / ideal_weight)
            # Split the segment
            self._split_by_data(self.bin, self.segments[split_ind], m)

    def _merge_by_weight(self, df: pd.DataFrame, ideal_segs_per_cluster: int):
        # Ideal weight for this cluster
        ideal_weight = df["weight"].sum() / ideal_segs_per_cluster
        print(f"{ideal_weight=}")
        while True:
            # Sort the df by weight
            df.sort_values("weight", inplace=True)
            # Add up all of the weights
            cumul_weight = np.add.accumulate(df["weight"])
            # Get the walkers that add up to be under the ideal weight
            to_merge = df[cumul_weight <= ideal_weight * self.weight_merge_cutoff]
            # If there's not enough for a merge then return
            if len(to_merge) < 2:
                break
            # Merge the segments
            self._merge_by_data(self.bin, self.segments[to_merge.inds.values])
            # Remove the merged walkers
            df.drop(to_merge.index.values, inplace=True)
            # Reset the index
            df.reset_index(drop=True, inplace=True)

    def split_with_combinations(
        self, df: pd.DataFrame, num_segs_for_splitting: int, num_resamples: int
    ) -> None:
        """
        Splits the segments based on the Dataframe using combinations of numbers.

        Args:
            df (pd.DataFrame): The DataFrame to be split.
            num_segs_for_splitting (int): The number of segments up for splitting.
            num_resamples (int): The number of resamples needed for splitting.

        Returns:
            None
        """
        # All the possible combinations of numbers that sum up to the num_resamples_needed
        combos = find_combinations(num_resamples, num_segs_for_splitting)
        # Need to check there's enough walkers to use that particular scheme
        split_possible = []
        # For each possible split
        for x in combos:
            # If the number of walkers is greater than the number needed for that possible split
            if len(x) <= num_segs_for_splitting:
                # Add to the list of possible splits
                split_possible.append(x)
        # This is the chosen split motif for this cluster
        chosen_splits = sorted(
            split_possible[self.rng.integers(len(split_possible))], reverse=True
        )
        print(f"split choice: {chosen_splits}")
        # Get the inds of the se gs
        sorted_segs = df.inds.values
        # For each of the chosen segs, split by the chosen value
        for idx, n_splits in enumerate(chosen_splits):
            print(f"idx: {int(sorted_segs[idx])}, n_splits: {n_splits}")
            # Find which segment we are splitting using the ind from the sorted_segs
            segment = self.segments[int(sorted_segs[idx])]
            # Split the segment
            self._split_by_data(self.bin, segment, int(n_splits + 1))

    def merge_with_combinations(
        self, df: pd.DataFrame, num_segs_for_merging: int, num_resamples: int
    ) -> None:
        """
        Merge segments based on combinations.

        Args:
            df (pd.DataFrame): The DataFrame containing segment data.
            num_segs_for_merging (int): The number of segments available for merging.
            num_resamples (int): The number of resamples needed for merging.

        Returns:
            None
        """
        # All the possible combinations of numbers that sum up to the num_resamples_needed
        combos = find_combinations(num_resamples, num_segs_for_merging)
        merges_possible = []
        # Need to check there's enough walkers to use that particular merging scheme
        for x in combos:
            # If the number of walkers is greater than the number needed for that possible merge
            if np.sum(x + 1) <= num_segs_for_merging:
                # Add to the list of possible merges
                merges_possible.append(x + 1)

        # This is the chosen merge motif for this cluster
        chosen_merge = sorted(
            list(merges_possible[self.rng.integers(len(merges_possible))]), reverse=True
        )
        print(f"merge choice: {chosen_merge}")

        for n in chosen_merge:
            # Get the last n rows of the df
            rows = df.tail(n)
            # Get the inds of the segs
            merge_group = list(rows.inds.values)
            print(f"merge group: {merge_group}")
            # Append the merge to the list of all merges
            self._merge_by_data(self.bin, self.segments[merge_group])
            # Remove the sampled rows
            df = df.drop(rows.index)

    def remove_overweight_segs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes overweight segments from the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.

        Returns:
            pd.DataFrame: The DataFrame with overweight segments removed.
        """
        og_len = len(df)
        # Remove out of weight walkers
        df = df[df["weight"] < self.cfg.merge_weight_limit]
        # The number of walkers up for splitting after removing walkers outside the threshold
        num_segs_to_merge = len(df)
        # Check if the number of walkers has changed
        if num_segs_to_merge != og_len:
            print("After removing walkers outside threshold")
            print(df)
        return df

    def remove_underweight_segs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes underweight segments from the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.

        Returns:
            pd.DataFrame: The DataFrame with underweight segments removed.
        """
        og_len = len(df)
        # Remove walkers outside the threshold
        df = df[df["weight"] > self.cfg.split_weight_limit]
        # The number of walkers up for splitting after removing walkers outside the threshold
        num_segs_to_split = len(df)
        # Check if the number of walkers has changed
        if num_segs_to_split != og_len:
            print("After removing walkers outside threshold")
            print(df)
        return df

    def get_cluster_df(self, df: pd.DataFrame, id: int) -> pd.DataFrame:
        return df[df["ls_cluster"] == id]

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
        new_df = df.iloc[:0, :].copy()
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
                    new_df.at[seg_ind, "weight"] = segment.weight
                    break
        # Reset the inds
        new_df["inds"] = np.arange(len(new_df))
        return new_df

    def get_prev_dcoords(
        self, iterations: int, upperbound: Optional[int] = None
    ) -> npt.ArrayLike:
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

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
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

        back_coords = []
        with data_manager.lock:
            iter_group = data_manager.get_iter_group(self.niter)
            coords_raw = iter_group["auxdata/dmatrix"][:]
            for seg in coords_raw[:, 1:]:
                back_coords.append(seg)

        return back_coords

    def get_prev_rcoords(self, iterations: int) -> npt.ArrayLike:
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

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
        # extract previous iteration data and add to curr_coords
        data_manager = westpa.rc.get_sim_manager().data_manager

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
        import pickle

        from synd.core import load_model

        subgroup_args = westpa.rc.config.get(["west", "drivers"])
        synd_model_path = expandvars(subgroup_args["synd_model"])
        backmap_path = expandvars(subgroup_args["dmatrix_map"])

        synd_model = load_model(synd_model_path)

        with open(backmap_path, "rb") as infile:
            dmatrix_map = pickle.load(infile)

        synd_model.add_backmapper(dmatrix_map.get, name="dmatrix")

        return synd_model

    def get_prev_dcoords_training(
        self, segments: Sequence[Segment], curr_dcoords: np.ndarray, iters_back: int
    ) -> np.ndarray:
        # Grab all the d-matrices from the last iteration (n_iter-1), will be in seg_id order by default.
        # TODO: Hope we have some way to simplify this... WESTPA 3 function...
        ibstate_group = westpa.rc.get_data_manager().we_h5file["ibstates/0"]

        if self.niter > 1:
            past_dcoords = []
            if iters_back == 0:
                past_dcoords = []
            elif self.niter <= iters_back:
                iters_back = self.niter - 1
                past_dcoords = np.concatenate(
                    self.get_prev_dcoords(iters_back, upperbound=self.niter)
                )
            else:
                past_dcoords = np.concatenate(
                    self.get_prev_dcoords(iters_back, upperbound=self.niter)
                )

            # Get current iter dcoords
            # curr_dcoords = np.concatenate(self.get_prev_dcoords(1))
            # print(curr_dcoords)
        else:  # building the list from scratch, during first iter
            past_dcoords, curr_dcoords = [], []
            for segment in segments:
                istate_id = ibstate_group["istate_index"][
                    "basis_state_id", int(segment.parent_id)
                ]
                # print(istate_id)
                auxref = int(tostr(ibstate_group["bstate_index"]["auxref", istate_id]))
                # print(auxref)

                dmatrix = self.synd_model.backmap([auxref], mapper="dmatrix")
                curr_dcoords.append(dmatrix[0])

        chosen_dcoords = []
        to_pop = []
        for idx, segment in enumerate(segments):
            # print(segment)
            # print(seg.wtg_parent_ids)
            if segment.parent_id < 0:
                istate_id = ibstate_group["istate_index"][
                    "basis_state_id", -int(segment.parent_id + 1)
                ]
                # print(istate_id)
                auxref = int(tostr(ibstate_group["bstate_index"]["auxref", istate_id]))
                # print(auxref)

                dmatrix = self.synd_model.backmap([auxref], mapper="dmatrix")
                chosen_dcoords.append(dmatrix[0])
            else:
                # print(idx)
                # print(segment.parent_id)
                chosen_dcoords.append(curr_dcoords[segment.parent_id])
                to_pop.append(segment.parent_id)

        curr_dcoords = np.asarray(curr_dcoords)
        if len(to_pop) > 0:
            curr_dcoords = np.delete(curr_dcoords, to_pop, axis=0)
        final = [
            np.asarray(i)
            for i in [chosen_dcoords, curr_dcoords, past_dcoords]
            if len(i) > 0
        ]

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
        rcoords = np.array(list(seg.data["rcoord"] for seg in segments))
        return rcoords.reshape(self.nsegs, self.nframes + 1, -1, 3)[:, 1:]
        # return rcoords.reshape(self.nsegs, self.nframes, -1, 3)[:, 1:]

    def get_dcoords(self, segments: Sequence[Segment]) -> npt.ArrayLike:
        dcoords = np.array(list(seg.data["dmatrix"] for seg in segments))
        return dcoords[:, 1:]
        # return rcoords.reshape(self.nsegs, self.nframes + 1, -1, 3)[:, 1:]
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
        self,
        bin_: Union[Bin, Generator[Segment, None, None]],  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.weight), np.object_)

    def _get_segments_by_parent_id(
        self,
        bin_: Union[Bin, Generator[Segment, None, None]],  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.parent_id), np.object_)

    def _get_segments_by_seg_id(
        self,
        bin_: Union[Bin, Generator[Segment, None, None]],  # noqa
    ) -> Sequence[Segment]:
        return np.array(sorted(bin_, key=lambda x: x.seg_id), np.object_)

    def _adjust_count(self, ibin):
        bin = self.next_iter_binning[ibin]
        target_count = self.bin_target_counts[ibin]
        weight_getter = operator.attrgetter("weight")

        # split
        while len(bin) < target_count:
            log.debug("adjusting counts by splitting")
            # always split the highest probability walker into two
            segments = sorted(bin, key=weight_getter)[-1]
            bin.remove(segments)
            new_segments_list = self._split_walker(segments, 2, bin)
            bin.update(new_segments_list)

        # merge
        while len(bin) > target_count:
            log.debug("adjusting counts by merging")
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
                self.niter = np.array([seg for seg in bin_])[0].n_iter - 1

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
                    cur_segments = self._get_segments_by_weight(
                        self.current_iter_segments
                    )
                else:
                    segments = self._get_segments_by_parent_id(bin_)
                    cur_segments = self._get_segments_by_seg_id(
                        self.current_iter_segments
                    )

                # print(segments)
                # print(cur_segments)
                self.cur_pcoords = self.get_pcoords(cur_segments)

                self.niter = cur_segments[0].n_iter
                self.nsegs = self.cur_pcoords.shape[0]
                self.nframes = self.cur_pcoords.shape[1]

                self.run(bin_, cur_segments, segments)

                print(
                    "Num walkers going to the final adjust_counts:",
                    len([x for x in bin_]),
                )

                self._adjust_count(ibin)
        # another sanity check
        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug("used initial states: {!r}".format(self.used_initial_states))
        log.debug("available initial states: {!r}".format(self.avail_initial_states))


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
    ml_mode: Optional[str] = "train"
    # Contact map distance cutoff
    contact_cutoff: Optional[float] = 8.0
    # How often to update the model.
    update_interval: Optional[int] = 10
    # Number of lagging iterations to use for training data.
    lag_iterations: Optional[int] = 50
    # Path to the base training data (Optional).
    base_training_data_path: Optional[Path] = None
    # Checkpoint file for static mode (Optional).
    static_chk_path: Optional[Path] = None

    @classmethod
    def from_westpa_config(cls) -> "MLSettings":
        westpa_config = westpa.rc.config.get(["west", "ddwe", "machine_learning"], {})
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
        if self.cfg.ml_mode == "train":
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
        """
        Load the target point pdb file and return the xyz coordinates for the CA atoms.

        Parameters:
        target_point_path (Path): The path to the target point pdb file.

        Returns:
        np.ndarray: The xyz coordinates for the CA atoms, reshaped to the shape (1, atoms, 3).
        """
        # Load the target point pdb in mdtraj
        target_point = mdtraj.load(str(target_point_path))
        # Select the CA atoms
        ca_atoms = target_point.top.select("name CA")
        # Get the xyz coordinates for the CA atoms
        coords = target_point.xyz[0][ca_atoms] * 10
        # Reshape the coordinates to the shape (1, atoms, 3)
        return np.reshape(coords, (1, len(ca_atoms), 3))

    def compute_sparse_contact_map(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute the sparse contact map from the given coordinates.

        Parameters:
            coords (np.ndarray): The input coordinates.

        Returns:
            np.ndarray: The computed sparse contact map.
        """
        # TODO: Fix this for real data
        # Compute a distance matrix for each frame
        # distance_matrices = [distance_matrix(frame, frame) for frame in coords]
        # Convert the distance matrices to contact maps (binary matrices of 0s and 1s)
        # contact_maps = np.array(distance_matrices) < self.cfg.contact_cutoff
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
        """
        Trains the model using the provided coordinates.

        Args:
            coords (np.ndarray): The coordinates to train the model on.

        Returns:
            None
        """
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

            pd.DataFrame(self.autoencoder.loss_curve_).to_csv(
                self.train_path / "loss.csv"
            )

            z, *_ = self.autoencoder.predict(
                contact_maps, checkpoint=self.most_recent_checkpoint_path
            )
            np.save(self.train_path / "z.npy", z[len(base_coords) :])

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """
        Predicts the latent space coordinates for a given set of coordinates.

        Parameters:
            coords (np.ndarray): The input coordinates.

        Returns:
            np.ndarray: The predicted latent space coordinates.
        """
        # Compute the contact maps
        contact_maps = self.compute_sparse_contact_map(coords)
        # Predict the latent space coordinates
        z, *_ = self.autoencoder.predict(
            contact_maps, checkpoint=self.most_recent_checkpoint_path
        )
        return z

    def get_target_point_rep(self, target_point_path: Path) -> np.ndarray:
        """
        Get the representation of a target point.

        Args:
            target_point_path (Path): The path to the target point file.

        Returns:
            np.ndarray: The representation of the target point.
        """

        # Load the target point coordinates
        target_point = self.get_target_point_coords(target_point_path)
        # Won't normally need these two lines
        distance_matrices = [distance_matrix(frame, frame) for frame in target_point]
        contact_maps = np.array(distance_matrices) < self.cfg.contact_cutoff
        # Predict the target point representation
        target_point_rep = self.predict(contact_maps)
        return np.concatenate(target_point_rep)


class ObjectiveSettings(BaseSettings):
    # Objective method to use (lof, clustering).
    objective_method: str
    # Function to measure distance between latent space points (cosine or euclidean).
    distance_metric: Optional[str] = "cosine"
    # How many walkers to consider for splitting and merging in lof.
    lof_consider_for_resample: Optional[int] = 12
    # Maximum total number of past latent space points to save for the lof scheme.
    max_past_points: Optional[int] = 2000
    # Number of neighbors for LOF.
    lof_n_neighbors: Optional[int] = 20
    # Maximum number of contact maps to save from each cluster.
    max_save_per_cluster: Optional[int] = 40
    # Clustering method to use (kmeans, dbscan, gmm, knani).
    cluster_method: Optional[str] = "kmeans"
    # Number of KMeans clusters.
    kmeans_clusters: Optional[int] = 8
    # Epsilon setting for dbscan.
    dbscan_epsilon: Optional[float] = 0.1
    # Minimum number of points for dbscan.
    dbscan_min_samples: Optional[int] = 25
    # Max components for GMM.
    gmm_max_components: Optional[int] = 8
    # Threshold for outliers in GMM.
    gmm_threshold: Optional[float] = 0.75
    # Number of "clusters" to use for the ablation version.
    ablation_clusters: Optional[int] = 8
    # Use a tuple to "scan" across multiple numbers, inclusive, e.g. (4,6)
    knani_clusters: Optional[Union[int, Tuple[int, int]]] = 8
    # Whether to use the Second Derivative of Davies–Bouldin index to determine minimum
    db_second: Optional[bool] = "true"
    # How often to load in structures, takes every sieve-th frame from the trajectory for analysis.
    sieve: Optional[int] = 1
    # The number of frames to extract from each cluster.
    n_structures: Optional[int] = 11
    # What metric to use to compare between frames, Mean-Square-Distance vs. whatever
    metric: Optional[str] = "MSD"
    # Different cluster initialization methods. comp_sim is the k-NANI initialization
    init_type: Optional[str] = "comp_sim"
    # Subset of data for Diversity selection. 20% * number of input structures needs to be >= the number of clusters you request (defaults to 10)
    percentage: Optional[int] = 10

    @classmethod
    def from_westpa_config(cls) -> "ObjectiveSettings":
        westpa_config = deepcopy(
            westpa.rc.config.get(["west", "ddwe", "objective"], {})
        )
        temp = {}
        keys_to_del = []
        # Find all keys to dictionaries in the config
        for key, value in westpa_config.items():
            if isinstance(value, dict):
                # Add all of the key value pairs to the config
                temp.update(value)
                # Add the key to the list of keys to delete
                keys_to_del.append(key)
        westpa_config.update(temp)
        for key in keys_to_del:
            # Remove all of the sub dictionaries
            del westpa_config[key]

        if "knani_clusters" in westpa_config:
            # If the clusters are a string, convert it to a tuple of ints
            try:
                westpa_config["knani_clusters"] = int(westpa_config["knani_clusters"])
            except ValueError:
                westpa_config["knani_clusters"] = tuple(
                    map(int, westpa_config["knani_clusters"].split(","))
                )
        return ObjectiveSettings(**westpa_config)


class Objective:
    def __init__(
        self,
        nsegs: int,
        niter: int,
        log_path: Path,
        datasets_path: Path,
    ):
        self.cfg = ObjectiveSettings.from_westpa_config()
        self.nsegs = nsegs
        self.niter = niter
        self.log_path = log_path
        self.datasets_path = datasets_path
        self.rng = np.random.default_rng()
        dist_functions = {
            "cosine": cosine_distance,
            "euclidean": euclidean_distance,
        }
        self.distance_function = dist_functions[self.cfg.distance_metric]

    def save_latent_context(
        self,
        all_labels: Optional[np.ndarray] = None,
        all_outliers: Optional[np.ndarray] = None,
    ) -> None:
        """
        Save the latent context to files.

        Args:
            all_labels (Optional[np.ndarray]): Array of cluster labels. If None, indices will be selected randomly from the entire dataset.
            all_outliers (Optional[np.ndarray]): Array of outlier indices.

        Returns:
            None
        """

        dcoords_to_save = []
        pcoords_to_save = []
        weights_to_save = []
        if all_labels is None:
            # Select indices at random from the entire dataset
            if len(self.all_dcoords) > self.cfg.max_past_points:
                indices = self.rng.choice(
                    len(self.all_dcoords), self.cfg.max_past_points, replace=False
                )
            else:
                indices = np.arange(len(self.all_dcoords))
            dcoords_to_save.append(self.all_dcoords[indices])
            pcoords_to_save.append(self.all_pcoords[indices])
            weights_to_save.append(self.all_weights[indices])
        else:
            # Loop through each cluster label and randomly select indices to save
            for label in set(all_labels):
                # Get the indices of the current label
                indices = np.where(all_labels == label)[0]
                if all_outliers is not None:
                    # Select all indices where the outliers are true
                    outlier_indices = indices[all_outliers[indices]]
                    # Remove the outlier indices from the indices
                    indices = np.setdiff1d(indices, outlier_indices)

                # If the number of indices is greater than the max save per cluster
                if len(indices) > self.cfg.max_save_per_cluster:
                    # Randomly select the max save per cluster
                    indices = self.rng.choice(
                        indices, self.cfg.max_save_per_cluster, replace=False
                    )
                if all_outliers is not None:
                    # Concatenate the selected indices
                    indices = np.concatenate((indices, outlier_indices))
                dcoords_to_save.append(self.all_dcoords[indices])
                pcoords_to_save.append(self.all_pcoords[indices])
                weights_to_save.append(self.all_weights[indices])

        dcoords_to_save = [x for y in dcoords_to_save for x in y]
        pcoords_to_save = [x for y in pcoords_to_save for x in y]
        weights_to_save = [x for y in weights_to_save for x in y]
        # Index the dcoord and pcoord arrays to save the selected indices
        np.save(
            self.datasets_path / f"context-dcoords-{self.niter}.npy", dcoords_to_save
        )
        np.save(
            self.datasets_path / f"context-pcoords-{self.niter}.npy", pcoords_to_save
        )
        np.save(
            self.datasets_path / f"context-weights-{self.niter}.npy", weights_to_save
        )

    def load_latent_context(
        self, dcoords, pcoords, weight
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            past_dcoords = np.load(
                self.datasets_path / f"context-dcoords-{self.niter - 1}.npy"
            )
            past_pcoords = np.load(
                self.datasets_path / f"context-pcoords-{self.niter - 1}.npy"
            )
            past_weights = np.load(
                self.datasets_path / f"context-weights-{self.niter - 1}.npy"
            )
            self.all_dcoords = np.concatenate([dcoords, past_dcoords])
            self.all_pcoords = np.concatenate([pcoords, past_pcoords])
            self.all_weights = np.concatenate([weight, past_weights])
        else:
            self.all_dcoords = dcoords
            self.all_pcoords = pcoords
            self.all_weights = weight

    def lof_function(self, all_z: np.ndarray) -> np.ndarray:
        # Run LOF on the full history of embeddings to assure coverage over past states
        clf = LocalOutlierFactor(
            n_neighbors=self.cfg.lof_n_neighbors, metric=self.distance_function
        ).fit(all_z)
        return clf.negative_outlier_factor_

    def kmeans_cluster_segments(self, all_z: np.ndarray) -> np.ndarray:
        # Perform the K-means clustering
        all_labels = KMeans(n_clusters=self.cfg.kmeans_clusters).fit(all_z).labels_
        return all_labels

    def knani_scan_cluster_segments(self, all_z: np.ndarray) -> np.ndarray:
        # Perform the K-means clustering
        all_scores, all_models = [], []

        if self.cfg.init_type in ["comp_sim", "div_select"]:
            model = KmeansNANI(
                data=all_z,
                n_clusters=self.cfg.knani_clusters[1],
                metric=self.cfg.metric,
                init_type=self.cfg.init_type,
                percentage=self.cfg.percentage,
            )
            initiators = model.initiate_kmeans()

        for i_cluster in range(
            self.cfg.knani_clusters[0], self.cfg.knani_clusters[1] + 1
        ):
            total = 0

            model = KmeansNANI(
                data=all_z,
                n_clusters=i_cluster,
                metric=self.cfg.metric,
                init_type=self.cfg.init_type,
                percentage=self.cfg.percentage,
            )
            if self.cfg.init_type == "vanilla_kmeans++":
                initiators = model.initiate_kmeans()
            elif self.cfg.init_type in ["comp_sim", "div_select"]:
                pass
            else:  # For k-means++, random
                initiators = self.cfg.init_type
            labels, centers, n_iter = model.kmeans_clustering(initiators=initiators)

            all_models.append([labels, centers])
            ch_score, db_score = compute_scores(all_z, labels=labels)

            dictionary = {}
            for j in range(i_cluster):
                dictionary[j] = all_z[np.where(labels == j)[0]]
            for val in dictionary.values():
                total += extended_comparison(
                    np.asarray(val), traj_numpy_type="full", metric=self.cfg.metric
                )

            all_scores.append(
                (i_cluster, n_iter, ch_score, db_score, total / i_cluster)
            )

        all_scores = np.array(all_scores)

        if self.cfg.db_second:
            print("Using second derivative of DBI to find optimal N")
            all_db = all_scores[:, 3]
            result = np.zeros((len(all_scores) - 2, 2))

            for idx, i_cluster in enumerate(all_scores[1:-1, 0]):
                # Calculate the second derivative
                result[idx] = [
                    i_cluster,
                    all_db[idx] + all_db[idx + 2] - (2 * all_db[idx + 1]),
                ]

            chosen_idx = np.argmax(result[:, 1]) + 1
        else:  # Pick only by using the lowest DBI
            chosen_idx = np.argmin(all_scores[:, 3])

        header = f"init_type: {self.cfg.init_type}, percentage: {self.cfg.percentage}, metric: {self.cfg.metric}"
        header += "Number of Clusters, Number of Iterations, Calinski-Harabasz score, Davies-Bouldin score, Average MSD"
        np.savetxt(
            self.log_path
            / f"{self.niter}-{self.cfg.percentage}-{self.cfg.init_type}_summary.csv",
            all_scores,
            delimiter=",",
            header=header,
            fmt="%s",
        )

        return all_models[chosen_idx][0]

    def knani_cluster_segments(self, all_z: np.ndarray) -> np.ndarray:
        # Perform the K-means clustering
        NANI_labels, _, _ = KmeansNANI(
            data=all_z,
            n_clusters=self.cfg.knani_clusters,
            metric=self.cfg.metric,
            init_type=self.cfg.init_type,
            percentage=self.cfg.percentage,
        ).execute_kmeans_all()

        return NANI_labels

    def dbscan_cluster_segments(
        self, all_z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster the segments using DBSCAN algorithm and assign cluster labels to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.
            all_z (np.ndarray): The latent space representation of the segments.

        Returns:
            pd.DataFrame: The DataFrame with cluster labels assigned to the segments.
        """
        # Cluster the segments
        all_labels = (
            DBSCAN(
                min_samples=self.cfg.dbscan_min_samples,
                eps=self.cfg.dbscan_epsilon,
                metric=self.distance_function,
            )
            .fit(all_z)
            .labels_
        )
        print(f"Total number of clusters: {max(set(all_labels)) + 1}")
        all_labels, all_outliers = self.assign_density_outliers(all_z, all_labels)

        return all_labels, all_outliers

    def gmm_cluster_segments(self, all_z: np.ndarray) -> np.ndarray:
        """
        Cluster the segments using Gaussian Mixture Model and assign cluster labels to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.
            all_z (np.ndarray): The latent space representation of the segments.

        Returns:
            pd.DataFrame: The DataFrame with cluster labels assigned to the segments.
        """
        # Perform the GMM clustering
        gmm = BayesianGaussianMixture(n_components=self.cfg.gmm_max_components)
        all_labels = gmm.fit_predict(all_z)
        print(f"Total number of clusters: {len(set(all_labels))}")
        proba = gmm.predict_proba(all_z)
        # Find the outliers based on the cluster labels
        outliers = np.where(proba.max(axis=1) < self.cfg.gmm_threshold, True, False)
        return all_labels, outliers

    def assign_density_outliers(
        self, all_z: np.ndarray, all_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assigns density outliers in the given cluster.

        Args:
            all_z (np.ndarray): The embedding vectors of the cluster.
            all_labels (np.ndarray): The labels of the cluster.

        Returns:
            np.ndarray: A boolean array indicating the outliers in the cluster.
            np.ndarray: The updated cluster labels.
        """

        # Set all the -1s to be outliers
        outliers = np.where(all_labels == -1, True, False)

        # if there are any outliers
        print(f"Number of outliers: {np.sum(all_labels[:self.nsegs] == -1)}")
        if -1 in all_labels:
            # Remove the outliers from the projections and labels
            inlier_z = all_z[all_labels != -1]
            inlier_labels = all_labels[all_labels != -1]

            # Loop through the outlier indices
            for ind in np.nditer(np.where(outliers)):
                # Find the distance to points in the embedding_history
                dist = [self.distance_function(all_z[ind], z) for z in inlier_z]

                # Find the min index
                min_ind = np.argmin(np.array(dist))

                # Set the cluster label for the outlier to match the min dist point
                all_labels[ind] = inlier_labels[min_ind]

        return all_labels, outliers

    def cluster_segments(self, all_z: np.ndarray) -> np.ndarray:
        """
        Cluster the segments based on the given method.

        Args:
            df (pd.DataFrame): The DataFrame containing the segments.
            all_z (np.ndarray): The latent space representation of the segments.

        Returns:
            pd.DataFrame: The DataFrame with cluster labels assigned to the segments.

        """
        # Initialize the outliers to be False
        all_outliers = np.zeros(all_z.shape[0], dtype=bool)
        # Cluster the segments
        if self.cfg.cluster_method == "kmeans":
            all_labels = self.kmeans_cluster_segments(all_z)
        elif self.cfg.cluster_method == "knani":
            if isinstance(self.cfg.knani_clusters, int):
                all_labels = self.knani_cluster_segments(all_z)
            else:
                all_labels = self.knani_scan_cluster_segments(all_z)
        elif self.cfg.cluster_method == "dbscan":
            all_labels, all_outliers = self.dbscan_cluster_segments(all_z)
        elif self.cfg.cluster_method == "gmm":
            all_labels, all_outliers = self.gmm_cluster_segments(all_z)

        return all_labels, all_outliers

    def ablation_cluster_segments(self) -> np.ndarray:
        seg_labels = [
            self.rng.integers(self.cfg.ablation_clusters) for _ in range(self.nsegs)
        ]
        return np.array(seg_labels)


class DDWESettings(BaseSettings):
    # Output directory for the run.
    output_path: Path
    # Run machine learning; set to False for ablation.
    do_machine_learning: bool
    # File containing the point for targeting. Set to None for no target.
    target_point_path: Optional[Path] = None
    # Interval for plotting the latent space.
    plot_interval: int
    # Drive resampling with the target in latent space (lst) or the pcoord (pcoord)
    sort_by: str
    # Pcoord approaches zero as the target is approached.
    pcoord_approaches_zero: bool
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
    def from_westpa_config(cls) -> "DDWESettings":
        westpa_config = deepcopy(westpa.rc.config.get(["west", "ddwe"], {}))
        # Remove other dictionaries from westpa_config
        for key in ["machine_learning", "objective"]:
            westpa_config.pop(key, None)

        return DDWESettings(**westpa_config)


class CustomDriver(DeepDriveMDDriver):
    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)

        self.cfg = DDWESettings.from_westpa_config()

        self.log_path = Path(f"{self.cfg.output_path}/westpa-ddmd-logs")
        os.makedirs(self.log_path, exist_ok=True)
        self.datasets_path = Path(f"{self.log_path}/datasets")
        os.makedirs(self.datasets_path, exist_ok=True)
        self.synd_model = self.load_synd_model()
        self.rng = np.random.default_rng()

    def plot_latent_space(
        self,
        z: np.ndarray,
        pcoords: np.ndarray,
        dist_to_target: np.ndarray,
        weights: np.ndarray,
        outliers: Optional[np.ndarray] = None,
        cluster_ids: Optional[np.ndarray] = None,
    ) -> None:
        # Test if the outliers exist and are not None
        if outliers is not None and outliers.any():
            # Test if outlier_data is an array of type bool
            if outliers.dtype == bool:
                # Switch to binary
                outliers = outliers.astype(int)
            # Plot the outliers
            self.plot_scatter(
                z,
                outliers,
                self.log_path / f"embedding-outlier-{self.niter}.png",
                cb_label="outliers",
            )
        if cluster_ids is not None:
            self.plot_scatter(
                z,
                cluster_ids,
                self.log_path / f"embedding-cluster-{self.niter}.png",
                cb_label="cluster ID",
            )
        self.plot_scatter(
            z,
            pcoords,
            self.log_path / f"embedding-pcoord-{self.niter}.png",
            cb_label="rmsd to target",
            min_max_color=(1, 12),
        )
        self.plot_scatter(
            z,
            dist_to_target,
            self.log_path / f"embedding-distance-{self.niter}.png",
            cb_label="latent space distance to target",
            min_max_color=(0, 2),
        )
        self.plot_scatter(
            z,
            weights,
            self.log_path / f"embedding-weight-{self.niter}.png",
            cb_label="weight",
            log_scale=True,
            min_max_color=(self.cfg.split_weight_limit, self.cfg.merge_weight_limit),
        )

    def plot_scatter(
        self,
        data: np.ndarray,
        color: np.ndarray,
        output_path: Path,
        cb_label: Optional[str] = None,
        min_max_color: Optional[Tuple[float, float]] = None,
        log_scale: bool = False,
    ):
        if min_max_color is not None:
            min_color, max_color = min_max_color
        else:
            min_color = np.min(color)
            max_color = np.max(color)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")
        hx, hy, hz = data[self.nsegs :, 0], data[self.nsegs :, 1], data[self.nsegs :, 2]
        hc = color[self.nsegs :]
        cx, cy, cz = data[: self.nsegs, 0], data[: self.nsegs, 1], data[: self.nsegs, 2]
        cc = color[: self.nsegs]
        if log_scale:
            hist = ax.scatter(
                hx,
                hy,
                hz,
                c=hc,
                norm=mpl.colors.LogNorm(vmin=min_color, vmax=max_color),
                label="Past iterations",
            )
            ax.scatter(
                cx,
                cy,
                cz,
                c=cc,
                marker="s",
                s=150,
                norm=mpl.colors.LogNorm(vmin=min_color, vmax=max_color),
                label="Current iteration",
            )
        else:
            hist = ax.scatter(
                hx,
                hy,
                hz,
                c=hc,
                vmin=min_color,
                vmax=max_color,
                label="Past iterations",
            )
            ax.scatter(
                cx,
                cy,
                cz,
                c=cc,
                marker="s",
                s=150,
                vmin=min_color,
                vmax=max_color,
                label="Current iteration",
            )

        if self.target_point is not None:
            ax.scatter(
                self.target_point[0],
                self.target_point[1],
                self.target_point[2],
                c="red",
                marker="x",
                s=200,
                label="Target point",
            )

        plt.colorbar(hist).set_label(cb_label)

        plt.title(f"iter: {self.niter}", loc="left")
        ax.legend(loc="upper left")
        plt.savefig(output_path)
        plt.close()

    def sort_df_lof(self, df: pd.DataFrame) -> pd.DataFrame:
        by_pcoord = df.sort_values(
            ["pcoord"], ascending=[self.cfg.pcoord_approaches_zero]
        )

        # Check if sorting by distance and pcoord is the same
        if self.target_point is not None:
            by_distance = df.sort_values(["distance"], ascending=[True])
            self.test_pcoord_vs_dist_sort(by_pcoord, by_distance)
        if self.cfg.sort_by == "lst":
            return by_distance
        else:
            return by_pcoord

    def sort_df_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        # If outliers exist and there is a cluster column
        if "outlier" in df.columns and "ls_cluster" in df.columns:
            pcoord_keys, pcoord_order = (
                ["outlier", "pcoord"],
                [False, self.cfg.pcoord_approaches_zero],
            )
            distance_keys, distance_order = ["outlier", "distance"], [False, True]
        else:
            pcoord_keys, pcoord_order = ["pcoord"], [self.cfg.pcoord_approaches_zero]
            distance_keys, distance_order = ["distance"], [True]

        by_pcoord = df.sort_values(pcoord_keys, ascending=pcoord_order)
        # Check if sorting by distance and pcoord is the same
        if self.target_point is not None:
            by_distance = df.sort_values(distance_keys, ascending=distance_order)
            self.test_pcoord_vs_dist_sort(by_pcoord, by_distance)
        if self.cfg.sort_by == "lst":
            return by_distance
        else:
            return by_pcoord

    def test_pcoord_vs_dist_sort(
        self, by_pcoord: pd.DataFrame, by_dist: pd.DataFrame
    ) -> None:
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
                        print(
                            f"Sorting by distance and pcoord is the same! {num_uni_distances} unique distances and {num_uni_pcoords} unique pcoords."
                        )

    def adjust_counts_towards_target(self, df: pd.DataFrame, ideal_count: int):
        """
        Adjusts the counts of walkers in a cluster to the ideal_count.
        This process will be skipped if there are an inproper number of
        walkers within the weight thresholds.

        Args:
            df (pd.DataFrame): The DataFrame containing the walkers in the cluster.
        """

        df = self.sort_df_cluster(df)
        print(df)
        # Total number of walkers
        num_segs_in_cluster = len(df)
        # Correct number of walkers per cluster
        if num_segs_in_cluster == ideal_count:
            print("Already the correct number of walkers")
        else:
            # Number of resamples needed to bring the cluster to the set number of walkers per cluster
            num_resamples = abs(num_segs_in_cluster - ideal_count)
            print(f"Number of resamples needed: {num_resamples}")
            # Need to split some walkers
            if num_segs_in_cluster < ideal_count:
                # Filter out weights under the threshold
                df = self.remove_underweight_segs(df)
                # The number of walkers that have sufficient weight for splitting
                num_segs_for_splitting = len(df)
                # Test if there are enough walkers with sufficient weight to split
                if num_segs_for_splitting == 0:
                    cluster_id = df.ls_cluster.values[0]
                    print(
                        f"Walkers up for splitting have weights that are too small. Skipping split/merge in cluster {cluster_id} on iteration {self.niter}..."
                    )
                else:  # Splitting can happen!
                    df = self.sort_df_cluster(df)
                    self.split_with_combinations(
                        df, num_segs_for_splitting, num_resamples
                    )
            # Need to merge some walkers
            else:
                # Minimum number of walkers needed for merging
                min_segs_for_merging = num_segs_in_cluster - ideal_count + 1
                # Filter out weights over the threshold
                df = self.remove_overweight_segs(df)
                num_segs_for_merging = len(df)
                # Need a minimum number of walkers for merging
                if num_segs_for_merging < min_segs_for_merging:
                    print(
                        f"Walkers up for merging have weights that are too large. Skipping split/merge in cluster {id} on iteration {self.niter}..."
                    )
                else:  # Merging gets to happen!
                    df = self.sort_df_cluster(df)
                    self.merge_with_combinations(
                        df, num_segs_for_merging, num_resamples
                    )

    def resample_for_target(self, df: pd.DataFrame):
        """
        Resamples segments towards the target based on the given
        DataFrame.The number of segments in the cluster determines the
        number of resamples.

        Args:
            df (pd.DataFrame): The DataFrame to be resampled.
        """
        df = self.sort_df_cluster(df)
        print(df)
        # Remove walkers outside the thresholds
        df = self.remove_underweight_segs(df)
        df = self.remove_overweight_segs(df)
        num_segs_in_cluster = len(df)

        # Decide the number of resamples based on the number of segments in the cluster
        # Too few walkers in weight threshold to resample here
        if num_segs_in_cluster < 3:
            print("Not enough walkers within the thresholds")
            num_resamples = 0
        # Maximum number of resamples
        elif num_segs_in_cluster >= self.cfg.max_resamples * 3:
            num_resamples = self.cfg.max_resamples
        # Determine resamples dynamically
        else:
            num_resamples = int(num_segs_in_cluster / 3)

        print(f"Number of resamples based on the number of walkers: {num_resamples}")
        if num_resamples != 0:
            df = self.sort_df_cluster(df)
            # Run resampling for the cluster
            self.split_with_combinations(df, num_resamples, num_resamples)
            self.merge_with_combinations(df, 2 * num_resamples, num_resamples)

    def resample_with_clusters(self, df: pd.DataFrame):
        """
        Resamples segments within each cluster following 3 different strategies:
        1. Ideal weight split/merges to balance weights
        2. Adjust counts in cluster to have same number of walkers
        3. Additional resampling to seek the target

        Args:
            df (pd.DataFrame): The input DataFrame.
            seg_labels (np.ndarray): The array of cluster labels.
        """
        # Set of all the cluster ids
        cluster_ids = sorted(set(df.ls_cluster.values))
        print(
            f"Number of clusters that have segments in them currently: {len(cluster_ids)}"
        )
        # Ideal number of walkers per cluster
        ideal_segs_per_cluster = int((len(df)) / len(cluster_ids))
        print(f"Ideal number of walkers per cluster: {ideal_segs_per_cluster}")
        print(df)

        # Get the cluster ids
        cluster_ids = sorted(set(df.ls_cluster.values))

        print("Starting ideal weight split/merges")
        for id in cluster_ids:
            # Get just the walkers in this cluster
            cluster_df = self.get_cluster_df(df, id)
            cluster_df = self.sort_df_cluster(cluster_df)
            print(f"cluster_id: {id}")
            print(cluster_df)

            # Ideal weight splits + merges
            self._split_by_weight(cluster_df, ideal_segs_per_cluster)
            self._merge_by_weight(cluster_df, ideal_segs_per_cluster)

        # Regenerate the df
        df = self.recreate_df(df)
        print("After ideal weight split/merge")
        print(df)

        print("Starting adjust counts split/merges")
        for id in cluster_ids:
            cluster_df = self.get_cluster_df(df, id)
            print(f"cluster_id: {id}")
            self.adjust_counts_towards_target(cluster_df, ideal_segs_per_cluster)

        # Regenerate the df
        df = self.recreate_df(df)
        print("After adjusting count split/merge")
        print(df)

        print("Starting target seeking split/merges")
        for id in cluster_ids:
            cluster_df = self.get_cluster_df(df, id)
            print(f"cluster_id: {id}")
            self.resample_for_target(cluster_df)

        # Regenerate the df
        df = self.recreate_df(df)
        print("After target seeking split/merge")
        print(df)

    def resample_with_lof(self, df: pd.DataFrame):
        """
        Resampling procedure for using the Local Outlier Factor (LOF) method.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
        """
        if "outlier" in df.columns:
            # Sort the DataFrame by outliers
            df = df.sort_values("outlier", ascending=True)
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
        # Remove walkers outside the threshold
        split_df = self.remove_underweight_segs(split_df)
        # The number of walkers up for splitting after removing walkers outside the threshold
        num_segs_to_split = len(split_df)

        # Remove out of weight walkers
        print("Walkers up for merging")
        print(merge_df)
        # Remove out of weight walkers
        merge_df = self.remove_overweight_segs(merge_df)
        # The number of walkers up for merging after removing walkers outside the threshold
        num_segs_to_merge = len(merge_df)

        # If there are more walkers for merging inside the weight threshold than for splitting
        if num_segs_to_merge > 2 * num_segs_to_split:
            # Set the number of resamples based on the number of segments for splitting
            num_possible_resamples = num_segs_to_split
        else:
            # Set the number of resamples based on the number of segments for merging
            num_possible_resamples = int(num_segs_to_merge / 2)

        # Decide the number of resamples based on the number of segments
        # Too few walkers in weight threshold to resample here
        if num_possible_resamples == 0:
            print(f"Not enough walkers to resample on iteration {self.niter}")
            num_resamples = 0
        # Maximum number of resamples
        elif num_possible_resamples >= self.cfg.max_resamples:
            num_resamples = self.cfg.max_resamples
        # Determine resamples dynamically
        else:
            num_resamples = num_possible_resamples

        print(f"Number of resamples based on the number of walkers: {num_resamples}")
        if num_resamples != 0:
            split_df = self.sort_df_lof(split_df)
            merge_df = self.sort_df_lof(merge_df)
            # Run resampling for the cluster
            self.split_with_combinations(split_df, num_resamples, num_resamples)
            self.merge_with_combinations(merge_df, 2 * num_resamples, num_resamples)

    def run(
        self,
        bin: Bin,
        cur_segments: Sequence[Segment],
        next_segments: Sequence[Segment],
    ) -> None:
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
            self.machine_learning_method = MachineLearningMethod(
                self.niter, log_path=self.log_path
            )

            # extract and format current iteration's data
            try:
                dcoords = self.get_dcoords(cur_segments)
            except KeyError:
                dcoords = self.get_restart_dcoords()
            dcoords = np.concatenate(dcoords)

            # Use Jeremy's fancy function to get the dcoords for the previous iterations
            all_dcoords = self.get_prev_dcoords_training(
                next_segments, dcoords, self.machine_learning_method.cfg.lag_iterations
            )
            # Get the dcoords for the current iteration
            dcoords = all_dcoords[: self.nsegs]

            # If training on the fly, check if it's time to train the model
            if self.machine_learning_method.cfg.ml_mode == "train":
                self.machine_learning_method.train(all_dcoords)

            if self.cfg.target_point_path is not None:
                # TODO: Fix the target point
                # Get the target point representation
                self.target_point = self.machine_learning_method.get_target_point_rep(
                    self.cfg.target_point_path
                )
                # self.target_point = np.load("target_point.npy")

            # Initialize the objective object with the target point
            self.objective = Objective(
                self.nsegs,
                self.niter,
                self.log_path,
                self.datasets_path,
            )
            # Load the cluster context
            self.objective.load_latent_context(dcoords, pcoords, weight)

            # Predict the latent space coordinates
            all_z = self.machine_learning_method.predict(self.objective.all_dcoords)
            print(f"Total number of points in the latent space: {len(all_z)}")
            if self.cfg.target_point_path is not None:
                all_distance = [
                    self.objective.distance_function(z, self.target_point)
                    for z in all_z
                ]
                # Add the distance column
                df["distance"] = all_distance[: self.nsegs]

            if self.objective.cfg.objective_method == "clustering":
                # Cluster the segments
                all_labels, all_outliers = self.objective.cluster_segments(all_z)

                # Get the cluster labels for the segments
                seg_labels = all_labels[: self.nsegs]

                # Plot the latent space
                if self.niter % self.cfg.plot_interval == 0:
                    self.plot_latent_space(
                        all_z,
                        self.objective.all_pcoords,
                        all_distance,
                        self.objective.all_weights,
                        outliers=all_outliers,
                        cluster_ids=all_labels,
                    )
                # Save the cluster context data
                self.objective.save_latent_context(all_labels, all_outliers)
            else:
                # Run LOF on the full history of embeddings to assure coverage over past states
                all_outliers = self.objective.lof_function(all_z)

                # Plot the latent space
                if self.niter % self.cfg.plot_interval == 0:
                    self.plot_latent_space(
                        all_z,
                        self.objective.all_pcoords,
                        all_distance,
                        self.objective.all_weights,
                        outliers=all_outliers,
                    )
                # Save the cluster context data
                self.objective.save_latent_context()

            # Add the outliers to the DataFrame
            df["outlier"] = all_outliers[: self.nsegs]
        else:
            # Initialize the objective object without the target point
            self.objective = Objective(
                self.nsegs, self.niter, self.log_path, self.datasets_path
            )
            self.target_point = None
            seg_labels = self.objective.ablation_cluster_segments()

        # Resample the segments
        if self.objective.cfg.objective_method == "lof":
            self.resample_with_lof(df)
        else:
            df["ls_cluster"] = seg_labels
            self.resample_with_clusters(df)
