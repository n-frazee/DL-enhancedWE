# DeepDriveWESTPA
Implementing binless WESTPA with DeepDriveMD. This repo uses the NTL9 synMD object (created by John Russo & Dan Zuckerman) as the subject and simulation engine for this example.


## Installation
To run these example files, create an environment as follows:
'''bash
mamba create -n deepdrive-westpa -c conda-forge westpa MDAnalysis scikit-learn natsort nbformat
mamba activate deepdrive-westpa 
pip install git+https://github.com/jeremyleung521/SynD.git@rng-fix
pip install git+https://github.com/jeremyleung521/mdlearn.git@pydantic-fix
pip install torch torchvision torchaudio

https://pytorch.org/
'''