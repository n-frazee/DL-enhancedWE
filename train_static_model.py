from mdlearn.nn.models.vae.symmetric_conv2d_vae import SymmetricConv2dVAETrainer
from deepdrive_we.alex_ddmd_driver import CVAESettings
from deepdrive_we.alex_ddmd_driver import MachineLearningMethod
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import pickle
import numpy as np
import pandas as pd


def main():
    autoencoder = SymmetricConv2dVAETrainer(**CVAESettings().model_dump())

    base_path = "ntl9_folding_synd/"
    # Load the target point coordinates
    target_point = MachineLearningMethod.get_target_point_coords(
        None, base_path + "ntl9_reference.pdb"
    )

    # Load the near target point coordinates
    near_target_points = np.load(base_path + "near_target_CA_coords.npy")
    # Concatenate the target point with the near target points
    target_points = np.concatenate((target_point, np.concatenate(near_target_points)))

    # Calculate the contact maps
    distance_matrices = [distance_matrix(frame, frame) for frame in target_points]
    contact_maps = np.array(distance_matrices) < 8.0

    # Load the contact maps from the synd
    d_dict = pickle.load(open(base_path + "dmatrix_map.pkl", "rb"))
    d_array = np.array([x for x in d_dict.values()])

    # Add in the contact maps from the target point
    d_array = np.concatenate((d_array, contact_maps))
    print("Total number of frames for training: ", d_array.shape[0])

    # Compute the COO matrices
    coo_maps = MachineLearningMethod.compute_sparse_contact_map(None, d_array)
    save_path = "static_model/"

    # Train the model
    autoencoder.fit(X=coo_maps, output_path=save_path + "model/")

    # Save the loss curve
    pd.DataFrame(autoencoder.loss_curve_).plot().get_figure().savefig(
        str(save_path + "model_loss_curve.png")
    )
    pd.DataFrame(autoencoder.loss_curve_).to_csv(save_path + "loss.csv")

    # Predict the latent space
    z, *_ = autoencoder.predict(coo_maps)
    # Save the latent space
    np.save(save_path + "z.npy", z)


if __name__ == "__main__":
    main()
