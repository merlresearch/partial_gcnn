<!--
Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Learning Partial Equivariances From Data

This repository contains the source code accompanying the NeurIPS 2022 paper:

[Learning Partial Equivariances From Data](https://arxiv.org/abs/2110.10211) <br/>**[David W. Romero](https://www.davidromero.ml/), [Suhas Lohit](https://merl.com/people/slohit)**.

## Abstract
Group Convolutional Neural Networks (G-CNNs) constrain learned features to respect the symmetries in the selected group,
and lead to better generalization when these symmetries appear in the data. If this is not the case, however, equivariance
leads to overly constrained models and worse performance. Frequently, transformations occurring in data can be better
represented by a subset of a group than by a group as a whole, e.g., rotations in $[-90^{\circ}, 90^{\circ}]$. In such cases,
a model that respects equivariance *partially* is better suited to represent the data. In addition, relevant transformations
may differ for low and high-level features. For instance, full rotation equivariance is useful to describe edge orientations
in a face, but partial rotation equivariance is better suited to describe face poses relative to the camera. In other words,
the optimal level of equivariance may differ per layer. In this work, we introduce *Partial G-CNNs*: G-CNNs able to learn
layer-wise levels of partial and full equivariance to discrete, continuous groups and combinations thereof as part of training.
Partial G-CNNs retain full equivariance when beneficial, e.g., for rotated MNIST, but adjust it whenever it becomes harmful, e.g.,
for classification of 6/9 digits or natural images. We empirically show that partial G-CNNs pair G-CNNs when full equivariance
is advantageous, and outperform them otherwise.

## Install

The required dependencies can be installed by running:
```
conda env create -f environment.yaml
```
This will create the conda environment `partial_gcnn` with the correct dependencies.

## Repository structure

This repository is organized as follows:

* `partial_equiv` contains the main PyTorch library of partial equivariant models.
  * `ck` contains several continuous parameterizations for the conv. kernels (SIREN, MLP, etc).
  * `general` implements several general layers such as schedulers, utilities, activation functions, etc.
  * `groups` presents a general structure for the implementation of groups, with specific implementations for `SE2` nad `E2`.
  * `partial_gconv` contains lifting, pooling as well as (partial) group convolutional layers.
* `models` and `datasets` contain the models and datasets used throughout our experiments;
* `cfg` contains the default configuration of `main.py` scripts, in YAML. We use Hydra with OmegaConf to manage the configuration of our experiments.
* `EXPERIMENTS.md` contains commands to replicate the experiments from the paper.
* `tests` presents multiple scripts that verify the equivariance of our layers.
* `*_constructor.py` serve as constructors of datasets and models. `optim` creates optimizers and learning rate schedulers.
* `trainer.py` and `tester.py` implement the training and testing routines of the repository.
* `main.py` is the main file of the repository. It is used to execute training and perform testing.
* some jupyter notebooks are provided, which are used to construct the plots in our paper.

## Using the code

### Execution
All our experiments are runned via `main.py` passing the corresponding configurations. The flags (and thus the configuraions) are handled by [Hydra](https://hydra.cc/docs/intro).
See `cfg/config.yaml` for all available flags. Flags can be passed as `xxx.yyy=value`.

- `net.*` describes settings for the networks (GCNN, ResNet, Augerino, etc). (model definitions in `models`).
- `conv.*` describes settings for the convolution, e.g., whether its partially equivariant.
- `kernel.*` describes settings for the construction of the continuous convolutional kernels, e.g., SIREN, MLP.
- `base_group.*` describes the group to be used, as well as the number of samples and the sampling type to be used. It also handles configurations for the Gumbel Softmax if required.
- `train.*` specifies the training arguments, e.g., batch size, learning rates, optimizer and schedulers.
- `dataset.*` specifies the dataset to be used
- `debug=True` specifies whether this is a dry_run.

### Reproducing our experiments

The commands used for the experiments reported in our paper can be found at [experiments/README.md](EXPERIMENTS.md/README.md).

## Cite
If you found this work useful in your research, please consider citing:

```
@article{romero2022learning,
  title={Learning Partial Equivariances from Data},
  author={Romero, David W and Lohit, Suhas},
  journal={NeurIPS},
  year={2022}
}
```

## Contact
David W. Romero: d.w.romeroguzman@vu.nl,
Suhas Lohit: slohit@merl.com

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.


## Copyright and License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:
```
Copyright (c) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

The following files:

* `partial_equiv/general/nn/norm.py`
* `partial_equiv/general/utils/*`

were taken without modification from https://github.com/rjbruin/flexconv (license included in [LICENSES/MIT.txt](LICENSES/MIT.txt)):

```
Copyright (c) 2021 David W. Romero & Robert-Jan Bruintjes
```

The following files:
* `datasets/cifar10.py`, `datasets/cifar100.py`, `datasets/pcam.py`, `datasets/stl10.py`
* `models/ckresnet.py`
* `partial_equiv/ck/*`
* `dataset_constructor.py`, `main.py`, `model_constructor.py`, `optim.py`, `tester.py`, `trainer.py`

were adapted from https://github.com/rjbruin/flexconv (license included in [LICENSES/MIT.txt](LICENSES/MIT.txt)):

```
Copyright (c) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2021 David W. Romero & Robert-Jan Bruintjes
```

The following files
* `partial_equiv/groups/O2.py`
* `partial_equiv/groups/SO2.py`

were adapted from https://github.com/dwromero/g_selfatt (license included in [LICENSES/MIT.txt](LICENSES/MIT.txt)):

```
Copyright (c) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2021 David W. Romero & Jean-Baptiste Cordonnier
```
