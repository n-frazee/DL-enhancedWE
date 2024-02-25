import logging
from pathlib import Path
from typing import Sequence, Tuple, Optional, List, Dict, Any
import time
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
from westpa.core.segment import Segment
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

from westpa_ddmd.driver import DeepDriveMDDriver
from westpa_ddmd.config import BaseSettings, mkdir_validator

log = logging.getLogger(__name__)

# TODO: This is a temporary solution until we can pass
# arguments through the westpa config. Requires a
# deepdrivemd.yaml file in the same directory as this script
CONFIG_PATH = Path(__file__).parent / "deepdrivemd.yaml"
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
    contact_cutoff: float = 8.0
    """The Angstrom cutoff for contact map generation."""

    # validators
    _mkdir_output_path = mkdir_validator("output_path")


class CVAESettings(BaseSettings):
    """Settings for mdlearn SymmetricConv2dVAETrainer object."""

    input_shape: Tuple[int, int, int] = (1, 40, 40)
    filters: List[int] = [32, 32, 32, 32]
    kernels: List[int] = [3, 3, 3, 3]
    strides: List[int] = [1, 1, 1, 2]
    affine_widths: List[int] = [128]
    affine_dropouts: List[float] = [0.5]
    latent_dim: int = 3
    lambda_rec: float = 1.0
    num_data_workers: int = 4
    #prefetch_factor: int = None
    batch_size: int = 64
    device: str = "cuda"
    optimizer_name: str = "RMSprop"
    optimizer_hparams: Dict[str, Any] = {"lr": 0.001, "weight_decay": 0.00001}
    epochs: int = 20
    checkpoint_log_every: int = 20
    plot_log_every: int = 20
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
        self.cfg = ExperimentSettings.from_yaml(CONFIG_PATH)

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
            str(self.train_path / "model_loss_curve.png")
        )

        z, *_ = self.autoencoder.predict(
            contact_maps, checkpoint=self.most_recent_checkpoint_path
        )
        np.save(self.train_path / "z.npy", z[len(base_coords):])

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

        self.base_training_data_path = SIM_ROOT_PATH / "common_files/train.npy"
        self.cfg = ExperimentSettings.from_yaml(CONFIG_PATH)
        self.log_path = self.cfg.output_path / "westpa-ddmd-logs"
        self.log_path.mkdir(exist_ok=True)
        self.machine_learning_method = None
        self.train_path = None
        self.datasets_path = self.log_path / "datasets"
        self.datasets_path.mkdir(exist_ok=True)

    def lof_function(self, z: np.ndarray) -> np.ndarray:
        # Load up to the last 50 of all the latent coordinates here
        if self.niter > self.cfg.lof_iteration_history:
            embedding_history = [
                np.load(self.datasets_path / f"z-{p}.npy")
                for p in range(self.niter - self.cfg.lof_iteration_history, self.niter)
            ]
        else:
            embedding_history = [np.load(p) for p in self.log_path.glob("z-*.npy")]

        # Append the most recent frames to the end of the history
        embedding_history.append(z)
        embedding_history = np.concatenate(embedding_history)

        # Run LOF on the full history of embeddings to assure coverage over past states
        clf = LocalOutlierFactor(n_neighbors=self.cfg.lof_n_neighbors).fit(
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
        old_model_path = (
            self.log_path
            / f"ml-iter-{self.niter - self.cfg.update_interval}"
        )
        training_z = np.load(old_model_path / "z.npy")
                
        training_z = np.reshape(
            training_z, (-1, self.nframes, training_z.shape[1])
        )[:, -1]

        live_z = np.concatenate(
            [
                np.load(self.datasets_path / f"last-z-{iter}.npy")
                for iter in range(
                    self.niter - self.cfg.update_interval, self.niter
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

    def train_decider(self, all_coords: np.ndarray) -> None:
        if self.niter == 1 or self.niter % self.cfg.update_interval == 0:
            # Time to train a model
            if self.niter == 1:  # Training on first iteration data
                print("Training an initial model...")
                train_coords = np.concatenate(all_coords)

            else:  # Retraining time
                print("Training a model on iteration " + str(self.niter) + "...")
                if self.niter > self.cfg.update_interval:
                    self.plot_prev_data()
                if self.cfg.lag_iterations >= self.niter:
                    train_coords = np.concatenate(self.get_prev_dcoords(self.niter - 1))
                else:
                    train_coords = np.concatenate(
                        self.get_prev_dcoords(self.cfg.lag_iterations)
                    )

            self.machine_learning_method.train(train_coords)

    def run(self, segments: Sequence[Segment]) -> None:
        # Determine the location for the training data/model
        if self.niter < self.cfg.update_interval:
            self.train_path = self.log_path / f"ml-iter-1"
        else:
            self.train_path = (
                self.log_path
                / f"ml-iter-{self.niter - self.niter % self.cfg.update_interval}"
            )

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
        np.save(self.datasets_path / f"coords-{self.niter}.npy", all_coords)

        # Train a new model if it's time
        self.train_decider(all_coords)

        # Regardless of training, predict
        z = self.machine_learning_method.predict(all_coords)
        nof_per_segment = self.lof_function(z)
        # Get data for sorting
        
        pcoord = np.concatenate(self.get_pcoords(segments)[:, -1])
        #try:
        #    rmsd = self.get_auxdata(segments, "rmsd")[:, -1]
        #except KeyError:
        #    rmsd = self.get_restart_auxdata("rmsd")[:, -1]
        
        weight = self.get_weights(segments)[:]
        df = pd.DataFrame(
            {
                "nof": nof_per_segment,
                "inds": np.arange(self.nsegs),
                "pcoord": pcoord,
                #"rmsd": rmsd,
                "weight": weight,
            }
        ).sort_values("nof")    
        print(df)

        # Finally, sort the smallest lof scores by biophysical values
        split_df = (  # Outliers
            df.head(self.cfg.num_trial_splits)
        )
        removed_splits = split_df[split_df['weight'] <= self.cfg.split_weight_limit]
        if len(removed_splits) > 1:
            print("Removed these walkers from splitting")
            print(removed_splits)
                
        # Filter out weights above the threshold 
        split_df = split_df[split_df['weight'] > self.cfg.split_weight_limit]
        if len(split_df) < self.cfg.num_we_splits:
            print("Walkers up for splitting have weights that are too small. Skipping split/merge this iteration...")
            to_split_inds = None
        else:
            split_df = (  # Outliers
                split_df.sort_values("pcoord", ascending=True)
                .head(self.cfg.num_we_splits)
            )
            # Collect the outlier segment indices
            to_split_inds = split_df.inds.values
            
        # Take the inliers for merging, sorting them by
        merge_df = (  # Inliers
            df.tail(self.cfg.num_trial_splits)
        )

        removed_merges = merge_df[merge_df['weight'] >= self.cfg.merge_weight_limit]
        if len(removed_merges) > 1:
            print("Removed these walkers from merging")
            print(removed_merges)

        merge_df = merge_df[merge_df['weight'] < self.cfg.merge_weight_limit]
        if len(merge_df) < 2 * self.cfg.num_we_splits:
            print("Walkers up for merging have weights that are too large. Skipping split/merge this iteration...")
            merge_list = None
        else:
            merge_df = (
                merge_df.sort_values("pcoord", ascending=True)
                .tail(2 * self.cfg.num_we_splits)
            )

            kmeans = KMeans(n_clusters=self.cfg.num_we_splits)
            kmeans.fit(np.array(merge_df['pcoord']).reshape(-1, 1))
            merge_df['cluster'] = kmeans.labels_

            merge_list = []
            for n in range(self.cfg.num_we_splits):
                cluster_df = merge_df[merge_df['cluster'] == n]
                if len(cluster_df) > 1:
                    merge_list.append(cluster_df.inds.values)


        # Log dataframes
        print(f"\n{split_df}\n{merge_df}")
        df.to_csv(self.datasets_path / f"full-niter-{self.niter}.csv")
        split_df.to_csv(self.datasets_path / f"split-niter-{self.niter}.csv")
        merge_df.to_csv(self.datasets_path / f"merge-niter-{self.niter}.csv")

        # Log the machine learning outputs
        np.save(self.datasets_path / f"z-{self.niter}.npy", z)

        # Save data for plotting
        np.save(
            self.datasets_path / f"last-z-{self.niter}.npy",
            np.reshape(z, (self.nsegs, self.nframes, -1))[:, -1, :],
        )
        np.save(self.datasets_path / f"pcoord-{self.niter}.npy", pcoord)

        return to_split_inds, merge_list

