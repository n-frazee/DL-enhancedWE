# DeepDriveWESTPA
Implementing binless WESTPA with DeepDriveMD. This repo uses the NTL9 synMD object (created by John Russo & Dan Zuckerman) as the subject and simulation engine for this example.


## Installation
To run these example files, create an environment as follows:

```bash
conda create -n deepdrive-westpa -c conda-forge westpa MDAnalysis scikit-learn natsort nbformat
conda activate deepdrive-westpa 
pip install git+https://github.com/jeremyleung521/SynD.git@rng-fix
pip install git+https://github.com/jeremyleung521/mdlearn.git@pydantic-fix
```

Finally, install pytorch based on the cuda libraies associated with your gpu; visit https://pytorch.org/ to find the correct command for you. Here's an example of a `pip` command:

```bash
pip install torch torchvision torchaudio
```

For more help, check out this post on Stack Overflow: https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with/61034368#61034368.

## Usage
To simply run the example files, execute the following line with the environment active:
```bash
./init.sh && ./run.sh
```

To change the parameters for the run, modify the `west.cfg` file. All of the parameters under the `ddwe` tag are set to configure the behavior of the `ddmd_dirver.py`. The full list of settings (including the config for the CVAE) can be found in `ddmd_driver.py`.

### Additional scripts
In the `scripts` directory are a few helpful python scripts. `prep_synd.ipynb` shows how to generate the `.pkl` files needed to use the `augmentation_driver.py`. `static_model_viewer.ipynb` is a convenient little plotting script for looking at a pretrained CVAE model. `train_static_model.py` handles training a static CVAE model.