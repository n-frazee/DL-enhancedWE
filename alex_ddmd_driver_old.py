import logging
from pathlib import Path
from typing import Sequence, Tuple, Optional
import os
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import westpa
from pathlib import Path
from westpa.core.segment import Segment
from sklearn.cluster import KMeans

from westpa_ddmd.driver import DeepDriveMDDriver
from westpa_ddmd.config import BaseSettings, mkdir_validator

log = logging.getLogger(__name__)

# TODO: This is a temporary solution until we can pass
# arguments through the westpa config. Requires a
# deepdrivemd.yaml file in the same directory as this script
SIM_ROOT_PATH = Path(__file__).parent

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

    # validators
    # _mkdir_output_path = mkdir_validator("output_path")
    pass


class MachineLearningMethod:
    def __init__(self, train_path):
        # Now use an autoencoder to model the nonlinear portions
        self.autoencoder = LinearAETrainer(
            input_dim=40,
            latent_dim=5,
            hidden_neurons=[32, 16, 8],
            epochs=500,
            verbose=False,
            checkpoint_log_every=100,
            plot_method=None,
            device="cpu",
            batch_size=64,
            prefetch_factor=None,
        )
        self.cfg = westpa.rc.config.get(['west', 'ddmd'])
        print(westpa.rc.config)
        for key in self.cfg:
          setattr(self, key, self.cfg[key])
          
        self.train_path = train_path
        self.sd4_W_path = Path(f'{self.train_path}/W.npy')
        self.training_data_path = Path(f'{self.train_path}/train.npy')
        self.autoencoder_checkpoint = Path(
            f'{self.train_path}/saves/checkpoints/checkpoint-epoch-500.pt'
        )



    def apply_W(self, coords: np.ndarray, W: np.ndarray) -> np.ndarray:

        # Remove the mean from the coordinates
        coordsAll = np.reshape(
            coords, (len(coords), coords.shape[1] * coords.shape[2])
        ).T
        avgCoordsAll = np.mean(coordsAll, 1)
        tmpAll = np.reshape(
            np.tile(avgCoordsAll, coords.shape[0]),
            (coords.shape[0], coords.shape[1] * coords.shape[2]),
        ).T
        caDevsMDall = coordsAll - tmpAll
        # Apply the transformation with W
        ZPrj4 = W.dot(caDevsMDall).T

        return ZPrj4

    def train(self, coords: np.ndarray, base_training_data_path: Path) -> None:
        """Takes aligned data and trains a new model on it. Outputs are saved in fixed postions."""
        base_coords = np.load(base_training_data_path)
        coords = np.transpose(coords, [0, 2, 1])
        coords = np.concatenate((base_coords, coords))
        # Perform SD2/4 on the aligned data
        start = time.time()
        print("SD2... ", end="")
        Y, _, _, U = SD2(
            coords.reshape(-1, coords.shape[2] * 3),
            m=coords.shape[2] * 3,
            verbose=False,
        )
        print(f"{time.time() - start} seconds")
        start = time.time()
        print("SD4... ", end="")
        W = SD4(Y[0:40, :], m=40, U=U[0:40, :], verbose=False)
        print(f"{time.time() - start} seconds")
        # Save off the W matrix for transforming before prediction
        np.save(self.sd4_W_path, W)

        # Calc the caDevsMDall
        ZPrj4 = self.apply_W(coords, W)

        start = time.time()
        print("Autoencoder... ", end="")
        # Fit the AE with the transformed data
        self.autoencoder.fit(ZPrj4, output_path=self.train_path / "saves")
        print(f"{time.time() - start} seconds")
        pd.DataFrame(self.autoencoder.loss_curve_).plot().get_figure().savefig(self.train_path / "model_loss_curve.png")

        z, _ = self.autoencoder.predict(ZPrj4[len(base_coords):], checkpoint=self.autoencoder_checkpoint)
        np.save(self.train_path / "z.npy", z)

        return None

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Takes aligned data, applies the W matrix from SD2/4 and then predicts with the trained AE"""
        coords = np.concatenate(coords)
        num_frames = len(coords)
        # load in the training data
        train_coords = np.load(self.training_data_path)
        coords = np.transpose(np.concatenate((coords, train_coords)), [0, 2, 1])
        # Apply the transformation from SD4
        W = np.load(self.sd4_W_path)
        ZPrj4 = self.apply_W(coords, W)
        ZPrj4 = ZPrj4[:num_frames]

        # Predict the latent space coordinates
        z, _ = self.autoencoder.predict(ZPrj4, checkpoint=self.autoencoder_checkpoint)

        return z


class CustomDriver(DeepDriveMDDriver):
    def __init__(self, rc=None, system=None):
        super().__init__(rc, system)
        
        self.base_training_data_path = SIM_ROOT_PATH / "common_files/train.npy"
        #self.cfg = ExperimentSettings.from_yaml(CONFIG_PATH)
        self.cfg = westpa.rc.config.get(['west', 'ddmd'])
        for key in self.cfg:
          setattr(self, key, self.cfg[key])
        self.log_path = Path(f'{self.output_path}/westpa-ddmd-logs')
        os.makedirs(self.log_path, exist_ok=True)
        self.train_path = None
        self.machine_learning_method = None
        self.datasets_path = Path(f'{self.log_path}/datasets')
        os.makedirs(self.datasets_path, exist_ok=True)

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
    def plot_old_model(self):
        old_model_path = self.log_path / f"ml-iter-{self.niter - self.update_interval}"
        training_z = np.load(old_model_path / "z.npy")
        training_z = np.reshape(training_z, (-1, self.nframes, training_z.shape[1]))[:, -1]
        live_z = np.concatenate(
            [
                np.load(self.datasets_path / f"last-z-{iter}.npy")
                for iter in range(
                    self.niter - self.update_interval, self.niter
                )
            ]
        )
        z_data = np.concatenate((training_z, live_z))
        num_iter = int(len(z_data) / self.nsegs)

        pcoord_data = np.concatenate(
            [
                np.load(self.datasets_path / f"pcoord-{iter}.npy")
                for iter in range(
                    self.niter - num_iter, self.niter
                )
            ]
        )

        state_data = np.concatenate(
            (0 * np.ones(len(training_z)), 1 * np.ones(len(live_z)))
        )
        plot_scatter(
            z_data, pcoord_data, self.log_path / f"embedding-pcoord-{self.niter}.png"
        )
        plot_scatter(
            z_data, state_data, self.log_path / f"embedding-state-{self.niter}.png"
        )
        return None
    
    def train_decider(self, all_coords: np.ndarray) -> None:
        if self.niter == 1 or self.niter % self.update_interval == 0:
            # Time to train a model
            if self.niter == 1: # Training on first iteration data
                print("Training an initial model...")
                train_coords = np.concatenate(all_coords)

            else: # Retraining time
                print("Training a model on iteration " + str(self.niter) + "...")
                if self.niter > self.update_interval:
                    # Plot the old latent space projections of the training data and iteration data
                    self.plot_old_model()

                if self.lag_iterations >= self.niter:
                    train_coords = np.concatenate(self.get_prev_rcoords(self.niter - 1))
                else:
                    train_coords = np.concatenate(self.get_prev_rcoords(self.lag_iterations))
            # Save the coords used to train for alignment in the future
            # train_coords.shape=(segments * frames, atoms, xyz)
            np.save(self.machine_learning_method.training_data_path, train_coords)

            self.machine_learning_method.train(train_coords, self.base_training_data_path)
        

        return None

    def run(self, segments: Sequence[Segment]) -> None:
        # if self.niter < self.update_interval:
        #     self.train_path = self.log_path / f"ml-iter-1"
        # else:
        #     self.train_path = self.log_path / f"ml-iter-{self.niter - self.niter % self.update_interval}"

        # self.train_path.mkdir(exist_ok=True)
        # self.machine_learning_method = MachineLearningMethod(self.train_path)
        # # extract and format current iteration's data
        # # additional audxata can be extracted in a similar manner
        # try:
        #     all_coords = self.get_rcoords(segments)
        # except KeyError:
        #     all_coords = self.get_restart_rcoords()

        # # all_coords.shape=(segments, frames, atoms, xyz)
        # np.save(self.datasets_path / f"coords-{self.niter}.npy", all_coords)

        # # Train a new model if it's time
        # self.train_decider(all_coords)
        
        # # Regardless of training, predict
        # z = self.machine_learning_method.predict(all_coords)

        # nof_per_segment = self.lof_function(z)
        # Get data for sorting
        
        pcoord = np.concatenate(self.get_pcoords(segments)[:, -1])
        try:
            rmsd = self.get_auxdata(segments, "rmsd")[:, -1]
        except KeyError:
            rmsd = self.get_restart_auxdata("rmsd")[:, -1]
            
        weight = self.get_weights(segments)[:]
        df = pd.DataFrame(
            {
#                "nof": nof_per_segment,
                "inds": np.arange(self.nsegs),
                "pcoord": pcoord,
                "rmsd": rmsd,
                "weight": weight,
            }
        )  
        print(df)


        randomized_df = df.sample(frac=1)#, random_state=42)
        df = randomized_df.reset_index(drop=True)


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
                split_df.sort_values("rmsd", ascending=True)
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
                merge_df.sort_values("rmsd", ascending=True)
                .tail(2 * self.num_we_splits)
            )

            kmeans = KMeans(n_clusters=self.num_we_splits)
            kmeans.fit(np.array(merge_df['rmsd']).reshape(-1, 1))
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

        # # Log the machine learning outputs
        # np.save(self.datasets_path / f"z-{self.niter}.npy", z)

        # # Save data for plotting
        # np.save(
        #     self.datasets_path / f"last-z-{self.niter}.npy",
        #     np.reshape(z, (self.nsegs, self.nframes, -1))[:, -1, :],
        # )
        # np.save(self.datasets_path / f"pcoord-{self.niter}.npy", pcoord)



        return to_split_inds, merge_list
