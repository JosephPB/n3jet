# n3jet

Neural Network NJet (n3jet) is a package to enable training and
inference of neural networks using the NJet numerical matrix element
calculation package. The package presented here accompanies the paper [arXiv:2002.07516](https://arxiv.org/abs/2002.07516).

Specifically, we aim to use neural networks as a multi-dimensional
interpolation function for complex, high-multiplicity QCD processes at
both leading order (LO) and next-to-leading order (NLO). NLO processes
at these multiplicities are very computationally expensive to
calculate and therefore by training network approximations, we aim to
reduce this computational time by a factor ~ proportional to the ratio
of the number of phase-space points used in inferences / the number of
points required for training.

Here, we provide implementations for the processes e+e- -> <= 5
jets. In the case of NLO approximations, NJet calculates the virtual
matrix correction and therefore we exploring approximating both these
corrections explicitly and the NLO/LO k-factors in which the LO
divergences have been normalised. 

As can be seen in the paper, we test both the ability of a single
network to approximate the entirety of uniformly sampled phase-space,
as well as taking an ensemble approach based on the phase-space
partitioning carried out during FKS subtraction. Implementations of
both the single network and ensemble approach are provided here for
ease of use and comparison.

## Usage

### Requirements

This package was written in Python 2 since the current NJet interface
is not Python 3 compatible, although this should also be Python 3
compatible.

The user is expected to have downloaded and installed NJet.

The `requirements.txt` should contain all remaining packages.

**Note:** Every file in this directory should be internally references
  with the only external reference pointing to you version of the NJet
  home directory. Please change this in:
  `n3jet/utils/njet_run_functions.py`.

### Structure

The directory is structured as follows:
```
|-LO/
  |- single/
  |- fks_ensemble/
|-NLO/
  |- single/
  |- fks_ensemble/
|-models/
  |- model.py
  |- model_dataset.py
|-phase/
  |- ## functions for different phase-space sampling algorithms ##
|-utils/
  |- ## utility files for calling njet and data generation ##
|-tests/
  |- ## test scripts for datasets ##
```

### Interfacing with NJet

n3jet is designed to enable data generation suitable for machine
learning models from NJet through a Python interface. Therefore, this
package requires NJet to be installed, although does not require the
use of additional package linkings.

**Note:** Every file in this directory should be internally references
  with the only external reference pointing to you version of the NJet
  home directory. Please change this in:
  `n3jet/utils/njet_run_functions.py`.

Most of the functions interfacing with NJet have been taken and
adapted from the `NJet/blha/njeyt.py` and  `NJet/examples/runtests.py`
scripts and many of the relevant functions appear in
`n3jets/utils/njet_run_functions.py`.

### Data generation

We offer several methods of data generation:

- explicit 2/3 jet sampling parameterised by jet energy and angles
  with default uniform sampling although this can be altered (see
  `n3jet/phase/phase_space.py`)
- uniform sampling across the whole of phase-space with RAMBO with a
  single global cut using the JADE algorithm (see
  `n3jet/phase/rambo_while.py`)
- uniform sampling over two regions separated by another JADE cut
  allowing for e.g. the creation of divergent and non-divergent region
  datasets (see `n3jet/phase/rambo_piecewise_balance.py`)

**Note:**. To make generting LO data according to the second point
  easier you can use the `n3jet/LO/single/LO_datasets.py` script with
  the appropriate flags, while NLO data can be done using:
  `n3jet/NLO/NLO_datasets.py`.

### Models

The `models` directory contains machine learning models which are all
written in Keras. 

Models are written as classes which include all the necessary
functions for data processing, allowing the user to process data
without training a model as well as just having to call `model.fit()`
if the class was initialised with the desired data.

The default model is a three hidden layer, fully connected model,
however, the `baseline_model()` function can easily be changed and
substituted for training alternatives.

### Single network models

Generating data and training networks on uniformly sampled data can be
done using `n3jet/LO/single/LO_datasets_models.py`, while the
equivalent for NLO can be achieved by using
`n3jet/NLO/NLO_datasets_modles.py`.

### FKS inspired ensemble

## TODO
- Create requirements file
- Write script with flags for piecewise generation
- Add more to model sections
- Write a script for training models without generating data
