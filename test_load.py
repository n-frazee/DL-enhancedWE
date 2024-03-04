from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from pydantic_settings import BaseSettings
from typing import Tuple, List, Union, Optional, Dict, Any
import numpy
import numpy as np
from scipy.sparse import coo_matrix


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
    optimizer_name: str = "RMSprop"
    optimizer_hparams: Dict[str, Any] = {"lr": 0.001, "weight_decay": 0.00001}
    epochs: int = 100
    checkpoint_log_every: int = 25
    plot_log_every: int = 25
    plot_n_samples: int = 5000
    plot_method: Optional[str] = "raw"
    

def compute_sparse_contact_map(coords: np.ndarray) -> np.ndarray:
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

autoencoder = SymmetricConv2dVAETrainer(**CVAESettings().dict(), device='mps')
autoencoder._load_checkpoint('common_files/checkpoint-epoch-100.pt', map_location='mps')

X = numpy.load('test.npy')

#X = numpy.expand_dims(X, axis=1)

Y = compute_sparse_contact_map(X)

print(X.shape)
print(len(Y))

autoencoder.predict(Y)
