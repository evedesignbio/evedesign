# evedesign: unified framework for accessible biosequence design

[![PyPI - Version](https://img.shields.io/pypi/v/evedesign.svg)](https://pypi.org/project/evedesign)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/evedesign.svg)](https://pypi.org/project/evedesign)

-----

<p align="center"><img src=".github/evedesign_logo.png" alt="evedesign logo" style="width:50%;" /></p>

## What is evedesign?

*evedesign* is a unified open-source framework for biosequence design that formalizes conditional design problems in a method-agnostic way. The framework allows users to seamlessly build and execute complex multiobjective design workflows, including supervised and unsupervised models, from a standardized set of specifications and operations. An interactive web-based user interface facilitates end-to-end biomolecular design for a broad scientific audience and is publicly available at https://evedesign.bio. 

This repository implements the core interfaces for standardizing the interaction with biomolecular models, 
generation of nucleotide sequences, and many other utility functions for structure handling, sequence space embeddings, etc.

Please also check [evedesign-server](https://github.com/evedesignbio/evedesign-server) for automated pipeline execution
from declarative design specifications and the REST API, as well as [evedesign-ui](https://github.com/evedesignbio/evedesign-ui)
for the interactive user interface. 

## Publication

[Hopf TA, Gazizov A, Garcia Busto S, Eschbach E, Lee S, Mirdita M, Orenbuch R, Belahsen K, Ross D, Sander C, Steinegger M, d'Oelsnitz S, Marks D. evedesign: accessible biosequence design with a unified
framework. bioRxiv (2026) doi:10.64898/2026.03.17.712115](https://www.biorxiv.org/content/10.64898/2026.03.17.712115v1)

## Installation

Use the following command to install *evedesign* with support for all currently implemented models.
You can remove any of the options if you do not need the respective model.
```
pip install evedesign[evmutation2,esm2,mpnn,umap] 
```

## Getting started

Please refer to some of our [examples](examples) how to use *evedesign*. We are planning to extend these further in the near future.

To implement your own models in the framework, please have a look at our existing reference implementations 
(e.g. [EVmutation2](src/evedesign/models/evmutation2.py), [ESM-2](src/evedesign/models/esm2.py), 
[ProteinMPNN](src/evedesign/models/mpnn.py), [Gibbs sampler](src/evedesign/samplers/gibbs.py)) as well as
the underlying [model interfaces](src/evedesign/model.py) and
[description of molecular systems and instances](src/evedesign/system.py).

We are happy to help if you have any questions!

## Roadmap and contributing

We plan to continuously add more models, restraints, oracles and samplers to the framework, e.g. *de novo* 3D structure generation
with BoltzGen or BindCraft. 

We are actively looking for further contributors to develop our framework jointly with the community. 
If you are interested or feel like an important model is missing from the framework, please get in contact with us!

### Development setup

Contributions should target the `develop` branch. To set up a local environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest build
```

Run the test suite and verify the package builds:

```bash
pytest tests/
python -m build
```

CI runs these checks on pull requests using core dependencies only. Optional model extras (`evmutation2`, `esm2`, `mpnn`, etc.) are not installed in CI because they pull in heavier ML dependencies such as PyTorch.

### Releasing

Tagged releases (`v*`) trigger the release workflow, which builds and publishes to PyPI. Maintainers must configure [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) for this repository before the first automated release.

## License

*evedesign* is released under the MIT license.

## Contact

For general questions or inquiries about *evedesign* please reach out to [hello@evedesign.bio](mailto:hello@evedesign.bio).
