# Installation

This document contains detailed instructions for installing dependencies for SANVOT. We recommand using the [install.sh](install.sh). The code is tested on an Ubuntu 20.04 system with Nvidia GPU (We recommand 4 Nvidia 3090).

### Requirments
* Conda with Python 3.8
* Nvidia GPU
* PyTorch 1.10.1
* pyyaml
* yacs
* tqdm
* matplotlib
* OpenCV

## Step-by-step instructions

#### Create environment and activate
```bash
conda create --name sanvot python=3.7
source activate sanvot
```

#### Install numpy/pytorch/opencv
```bash
conda install numpy
conda install pytorch=1.10.1 torchvision cudatoolkit=10.1 -c pytorch
pip install opencv-python
```

#### Install other requirements
```bash
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboard future mpi4py optuna
```

#### Build extensions
```bash
python setup.py build_ext --inplace
```


## Try with scripts
```bash
bash install.sh /path/to/your/conda sanvot
```

