import logging
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from westpa.core.segment import Segment
from sklearn.cluster import KMeans

import westpa
import os
from pathlib import Path
from os.path import expandvars
from westpa_ddmd.driver import DeepDriveMDDriver
from westpa_ddmd.config import BaseSettings, mkdir_validator

log = logging.getLogger(__name__)

# TODO: This is a temporary solution until we can pass
# arguments through the westpa config. Requires a
# deepdrivemd.yaml file in the same directory as this script
SIM_ROOT_PATH = Path(__file__).parent

float_class = ['split_weight_limit', 'merge_weight_limit']
int_class = ['update_interval', 'lag_iterations', 'lof_n_neighbors', 'lof_iteration_history', 'num_we_splits', 'num_trial_splits']

class CustomDriver(DeepDriveMDDriver):
    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)
        
        self.base_training_data_path = expandvars(f'$WEST_SIM_ROOT/common_files/train.npy')
        self.cfg = westpa.rc.config.get(['west', 'ddmd'])
        for key in self.cfg:
          if key in int_class:
              setattr(self, key, int(self.cfg[key]))
          elif key in float_class:
              setattr(self, key, float(self.cfg[key]))
          else:
              setattr(self, key, self.cfg[key])
        self.log_path = Path(f'{self.output_path}/westpa-ddmd-logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.train_path = None
        self.machine_learning_method = None
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
        print(df)

        randomized_df = df.sample(frac=1, random_state=np.random.randint(0, 999, 1))
        df = randomized_df.reset_index(drop=True)

        # Finally, sort the smallest lof scores by biophysical values
        split_df = (  # Outliers
            df.head(self.num_trial_splits)
        )
        removed_splits = split_df[split_df['weight'] <= float(self.split_weight_limit)]
        if len(removed_splits) > 1:
            print("Removed these walkers from splitting")
            print(removed_splits)
                
        # Filter out weights above the threshold 
        split_df = split_df[split_df['weight'] > float(self.split_weight_limit)]
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
                if len(cluster_df) > 1:
                    merge_list.append(cluster_df.inds.values)


        # Log dataframes
        print(f"\n{split_df}\n{merge_df}")
        df.to_csv(self.datasets_path / f"full-niter-{self.niter}.csv")
        split_df.to_csv(self.datasets_path / f"split-niter-{self.niter}.csv")
        merge_df.to_csv(self.datasets_path / f"merge-niter-{self.niter}.csv")

        return to_split_inds, merge_list
