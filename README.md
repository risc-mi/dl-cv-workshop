# Introduction to Deep Learning: Hands-on Workshop in Computer Vision

This repository accompanies the workshop *Introduction to Deep Learning: Hands-on Workshop in Computer Vision*. See below for detailed instructions on setting up the necessary software on your computer.

Code-files (Jupyter notebooks) and data will be added to this repository in due course.

## Software Setup

**Note**: The easiest way is to set up the software locally, on the very laptop you are going to use during the workshop. We will not need lots of computing resources, only approx. 1 GB of memory and a medium-strength CPU with >= 8 GB of RAM.

### 1. Clone this Repository

Clone or download this repository to your computer.

### 2. Download and Install Miniconda

Make sure [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) is installed on your system. It is available for all major operating systems, including Windows, MacOS, and Linux.

Miniconda provides a minimal Python installation with an integrated package manager, and allows to easily create and manage dedicated Python environments.

**Note**: Even if you have Python pre-installed (e.g., on Linux), it is strongly recommended to use Miniconda.

### 3. Create a new Python Environment

Open a terminal window and `cd` to the directory containing the local clone of this repository (Step 1). Then execute the following command:

```
$ conda env create --file environment.yml
```

After a few seconds, you might be asked to confirm creating a new environment and downloading a couple of packages amounting to approx. 1 GB. If so, please confirm and wait until all packages are downloaded and installed (this may take a few minutes). Then, execute

```
$ conda activate dl_cv_workshop
```

to verify that the environment was properly set up. You can then also start a new Python interpreter by executing

```
$ python
```

and importing some example packages, like

```
>>> import torch
>>> import cv2
```

If this does not throw any exception, everything seems to be fine, and you can exit the interpreter again with

```
>>> exit()
```

#### Troubleshooting

If `import cv2` throws an exception that reads something like

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

then

```
apt-get install libgl1
```

might solve the problem (see also [here](https://stackoverflow.com/a/74501248)).

### 4. Install GPU-Version of PyTorch [optional, untested]

If you have an NVIDIA- or AMD GPU, you can install the GPU-version of PyTorch. To check for an NVIDIA GPU, execute

```
$ nvidia-smi
```

If the command succeeds, you should get a table-like output with information about the CUDA version installed on your system, which should be either 11.8 or 12.1. In that case, you can install the GPU-version of PyTorch with

```
$ conda install pytorch pytorch-cuda=<YOUR CUDA VERSION> -c pytorch -c nvidia
```

Check the [official installation instructions](https://pytorch.org/get-started/locally/) if anything goes wrong or you have an AMD GPU.

**Note**: A GPU is not required for the workshop, but can significantly speed up computations, in particular, model training.

## Contact

If you run into problems, do not hesitate to drop Alexander Maletzky (@ RISC-Software) an e-mail, or open an [issue](https://github.com/risc-mi/dl-cv-workshop/issues).