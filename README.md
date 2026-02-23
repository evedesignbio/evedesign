# evedesign: unified framework for accessible biosequence design

[![PyPI - Version](https://img.shields.io/pypi/v/evedesign.svg)](https://pypi.org/project/evedesign)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/evedesign.svg)](https://pypi.org/project/evedesign)

-----

## What is evedesign?

*evedesign* is a unified open-source framework for biosequence design that formalizes conditional design problems in a method-agnostic way. The framework allows users to seamlessly build and execute complex multiobjective design workflows, including supervised and unsupervised models, from a standardized set of specifications and operations. An interactive web-based user interface facilitates end-to-end biomolecular design for a broad scientific audience and is publicly available at https://evedesign.bio. 

This repository implements the core interfaces for standardizing the interaction with biomolecular models, 
generation of nucleotide sequences, and many other utility functions for structure handling, sequence space embeddings, etc.

Please also check [evedesign-server](https://github.com/evedesignbio/evedesign-server) for automated pipeline execution
from declarative design specifications and the REST API, as well as [evedesign-ui](https://github.com/evedesignbio/evedesign-ui)
for the interactive user interface. 

## Reference

Our preprint describing the core concepts behind the evedesign framework will be posted here shortly.

## Installation

Use the following command to install *evedesign* with support for all currently implemented models.
You can remove any of the options if you do not need the respective model.
```
pip install evedesign[evmutation2,esm2,mpnn,umap] 
```

## Getting started

Please refer to some of our [examples](examples) how to use *evedesign*. We are planning to extend these further in the near future.

## Roadmap and contributing

We plan to continuously add more models, restraints, oracles and samplers to the framework, e.g. *de novo* 3D structure generation
with BoltzGen or BindCraft. 

We are actively looking for further contributors to develop our framework jointly with the community. 
If you are interested or feel like an important model is missing from the framework, please get in contact with us!


## License

*evedesign* is released under the MIT license.

## Contact

For general questions or inquiries about *evedesign* please reach out to [hello@evedesign.bio](mailto:hello@evedesign.bio).
