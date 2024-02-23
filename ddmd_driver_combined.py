import logging
from pathlib import Path
from typing import Sequence
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans
from itertools import product

import os
from pathlib import Path
from os.path import expandvars

import westpa
from westpa.core.binning import Bin
from westpa.core.segment import Segment
from westpa.core.we_driver import WEDriver

log = logging.getLogger(__name__)

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
        
        for segments, split_into_custom  in zip(to_split, chosen_pick):
            bin.remove(segments)
            new_segments_list = self._split_walker(segments, split_into_custom+1, bin)
            bin.update(new_segments_list)
        
        print(f'split choices: {chosen_pick}')

    def _merge_by_data(self, bin: Bin, to_merge: Sequence[Segment]) -> None:
        bin.difference_update(to_merge)
        new_segment, parent = self._merge_walkers(to_merge, None, bin)
        bin.add(new_segment)

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
                coords_raw = iter_group["auxdata/rcoord"][:]
                coords_raw = coords_raw.reshape((self.nsegs, self.nframes + 1, -1, 3))
                for seg in coords_raw[:, 1:]:
                    back_coords.append(seg)

        return back_coords

    def get_restart_rcoords(self) -> npt.ArrayLike:
        """Collect coordinates for restart from pervious iteration.
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
            coords_raw = iter_group["auxdata/rcoord"][:]
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
                check = [len(i) for i in merge_groups_inds]
                if to_split_inds is not None and merge_groups_inds is not None and np.max(check) <= self.num_we_splits +1 and np.min(check) > 0:
                    to_split = np.array([segments[to_split_inds]])[0]

                    # Split walker with largest scaled diff
                    
                    self._split_by_data(bin_, to_split, split_into=2)

                    for to_merge_inds in merge_groups_inds:
                        if len(to_merge_inds) > 1:
                            to_merge = segments[to_merge_inds]
                            self._merge_by_data(bin_, to_merge)
                else:
                    print(f'skipped due to kmeans')

        # another sanity check
        self._check_post()

        # TODO: What does this line do?
        self.new_weights = self.new_weights or []

        log.debug("used initial states: {!r}".format(self.used_initial_states))
        log.debug("available initial states: {!r}".format(self.avail_initial_states))


class CustomDriver(DeepDriveMDDriver):
    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)
        
        self.base_training_data_path = expandvars(f'$WEST_SIM_ROOT/common_files/train.npy')
         
        self.log_path = Path(f'{self.output_path}/westpa-ddmd-logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.datasets_path = Path(f'{self.log_path}/datasets')
        os.makedirs(self.datasets_path, exist_ok=True)

    def run(self, segments: Sequence[Segment]) -> None:
        pcoord = np.concatenate(self.get_pcoords(segments)[:, -1])
            
        weight = self.get_weights(segments)[:]
        df = pd.DataFrame(
            {
                "inds": np.arange(self.nsegs),
                "pcoord": pcoord,
                "weight": weight,
            }
        )  

        randomized_df = df.sample(frac=1, random_state=np.random.default_rng())
        df = randomized_df.reset_index(drop=True)

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
        df.to_csv(self.datasets_path / f"full-niter-{self.niter}.csv")
        split_df.to_csv(self.datasets_path / f"split-niter-{self.niter}.csv")
        merge_df.to_csv(self.datasets_path / f"merge-niter-{self.niter}.csv")

        return to_split_inds, merge_list
