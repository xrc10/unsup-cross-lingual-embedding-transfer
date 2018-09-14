
# unsup-cross-lingual-embedding-transfer
Code for "Unsupervised Cross-lingual Transfer of Word Embedding Spaces" in EMNLP 2018 [[pdf
]](https://arxiv.org/abs/1809.03633)

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

pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl
pip install tqdm
```

## Arguments
run ``` python src/runner.py --help ``` to see the usage of arguments

## Example script
``example.sh`` gives an example run of our model. It will run the "bg-en" experiment of "LEX-C" and then evaluate the accuracy@1. You need to download data before running:

```
cd data
./download.sh
```

Note that this data is a subset of the release from [MUSE](https://github.com/facebookresearch/MUSE) .

Then run the following command to start training:

```
cd .. # back to root repo directory
./example.sh
```

## Cite
Please consider citing our paper if you find this repo useful in your research.
```
@article{xu2018unsupervised,
  title={Unsupervised Cross-lingual Transfer of Word Embedding Spaces},
  author={Xu, Ruochen and Yang, Yiming and Otani, Naoki and Wu, Yuexin},
  booktitle={Conference on Empirical Methods on Natural Language Processing},
  year={2018}
}
```
