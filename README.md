# unsup-cross-lingual-embedding-transfer
Code for "Unsupervised Cross-lingual Transfer of Word Embedding Spaces" in EMNLP 2018

## Setup

This software runs __python 3.6__ with the following libraries:

- tensorflow r1.6(with cuda 9.0)
- numpy
- tqdm

### Linux setup with Anaconda

```shell

wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
./Anaconda3-5.0.1-Linux-x86_64.sh  # Follow the instructions

conda create -n <name of your environment> python=3.6 anaconda
source activate <name of your environment>

pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp35-cp35m-linux_x86_64.whl
pip install tqdm
```

## Arguments
run ``` python src/runner.py --help ``` to see the usage of arguments

## Example script
``example.sh`` gives a example run of our model. It will run the "bg-en" experiment of "LEX-C". You need to download data before running.

```
cd data
./download.sh
```

Note that this data is a subset of the release from https://github.com/facebookresearch/MUSE

Then run the following command to start training:

```
cd .. # back to root directory
./example.sh
```
