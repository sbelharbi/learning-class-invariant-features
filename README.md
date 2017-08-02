### Neural Networks Regularization Through Learning Invariant Features for Small Datasets.

This repository contains the code of the paper `Neural Networks Regularization Through Learning Invariant Features for Small Datasets, S.Belharbi, C.Chatelain, R.Hérault, S.Adam. 2017.`[ArXiv](https://arxiv.org/).

*Please cite this paper if you use the code in this repository as part of a published research project.*

Requirements:
- Python (2.7).
- Theano (0.9).
- Numpy  (1.13).
- Keras (2.0).
- Matplotlib (1.2)
- Yaml (3.10).

To run this code, you need to uncompress the MNIST dataset:
```sh
$ unzip data/mnist.pkl.zip -d data/
$ unzip data/mnist_bin17.pkl.zip -d data/
```

To generate *mnist-noise* and *mnist-img*, please see the file `mnist_manip.py`.

The folder `config_yaml` contains [yaml](http://www.yaml.org/start.html) files to configure an experiment. For instance, this is the content of the yaml file to run an experiment using an mlp with 3 hidden layers:
```yaml
corrupt_input_l: 0.0
debug_code: false
extreme_random: true
h_ind: [false, false, true, false]
h_w: 0.0
hint: true
max_epochs: 400
model: train3_new_dup
nbr_sup: 1000
norm_gh: false
norm_gsup: false
repet: 0
run: 0
start_corrupting: 0
start_hint: 110
use_batch_normalization: [false, false, false, false]
use_sparsity: false
use_sparsity_in_pred: false
use_unsupervised: false
```
To run this experiment on a GPU:
```sh
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32  python train3_new_dup.py train3_new_dup_0_1000_3_0_0_0_0_0_False_False_False_False_False_110.yaml 
```

To use [Slurm](https://slurm.schedmd.com/), see the folder `jobs`.