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
You can remove any of the options if you do not need the respective model. Please see specific instructions for Boltz-2 further below.
```
pip install evedesign[evmutation2,esm2,mpnn,umap,gpytorch,eve,promb,sapiens] 
```

### Boltz2 installation 

#### GPU-based (CUDA, recommended)

For structure predictions with Boltz-2 and CUDA, install the `boltz2fold-cuda` extra with [uv](https://docs.astral.sh/uv/):
```bash
uv pip install evedesign[boltz2fold-cuda]
```
PyPI's default torch ships with CUDA support (tested on a CUDA 13
build, which runs on CUDA 12.x+ drivers via forward compatibility).
If you need a specific CUDA build to match an older driver, you might explore adding the
matching PyTorch index, e.g.:
```bash
    uv pip install evedesign[boltz2fold-cuda] \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --index-strategy unsafe-best-match
```
**uv is required** for this install path: boltz and its dependencies ship overly-conservative version pins that
are overridden in `[tool.uv]` in `pyproject.toml`, and pip does not honor those overrides.

#### CPU/MPS-based

For CPU/MPS-only use (no CUDA), install the `boltz2fold` extra instead of `boltz2fold-cuda`.

## Getting started

Please refer to some of our [examples](examples) how to use *evedesign*. We are continuously extending these as new 
models are added to the framework.

To implement your own models in the framework, please have a look at our existing reference implementations 
(e.g. [EVmutation2](src/evedesign/models/evmutation2.py), [ESM-2](src/evedesign/models/esm2.py), 
[ProteinMPNN](src/evedesign/models/mpnn.py), [Gibbs sampler](src/evedesign/samplers/gibbs.py)) as well as
the underlying [model interfaces](src/evedesign/model.py) and
[description of molecular systems and instances](src/evedesign/system.py).

We are happy to help if you have any questions!

## Currently available models and methods

### Biomolecular models, embedders and restraints

Note that most methods listed as `Scorer` also support the `MutationScorer` and `ConditionalMutationScorer` interfaces. 

| Name                        | Class                                                            | Interfaces                           | 
|-----------------------------|------------------------------------------------------------------|--------------------------------------|
| EVmutation2                 | `evedesign.models.evmutation2.EVmutation2`                       | `Generator`, `Scorer`, `Transformer` |
| LigandMPNN/ProteinMPNN      | `evedesign.models.mpnn.LigandMPNN`                               | `Generator`, `Scorer`                |
| ESM-2                       | `evedesign.models.esm2.ESM2`                                     | `Transformer` `Scorer`               |
| Boltz-2                     | `evedesign.models.boltzfold.BoltzFoldTransformer`                | `Transformer`, `Scorer`              |
| EVcouplings                 | `evedesign.models.evcouplings.EVcouplings`                       | `Scorer`                             |
| EVE                         | `evedesign.models.eve.EVE`                                       | `Scorer`                             |
| MixMHC2pred                 | `evedesign.models.immunogenicity.MixMHC2Pred`                    | `Scorer`                             |
| promb/OASis humanness       | `evedesign.models.oasis_humanness.OASisHumanness`                | `Scorer`                             |
| Sapiens antibody humanizer  | `evedesign.models.sapiens_humanizer.SapiensHumanizer`            | `Generator`                          |
| One-hot encoding embedder   | `evedesign.models.embedders.OneHotEmbedder`                      | `Transformer`                        |
| BLOSUM embedder             | `evedesign.models.embedders.BLOSUMEmbedder`                      | `Transformer`                        |
| Sequence distance restraint | `evedesign.restraints.seq_dist.LinearSeqDistRestraint`           | `Scorer`                             |
| Exposed sequence motifs     | `evedesign.restraints.motif.ExposedMotifRestraint`               | `Scorer`                             |
| Isoelectric point restraint | `evedesign.restraints.physicochemical.IsoelectricPointRestraint` | `Scorer`                             |
| Molecular weight restraint  | `evedesign.restraints.physicochemical.MolecularWeightRestraint`  | `Scorer`                             |

### Supervised models

| Name                                | Class                                                            | Interfaces       | 
|-------------------------------------|------------------------------------------------------------------|------------------|
| Scikit-learn regressors/classifiers | `evedesign.models.supervised.SklearnPredictorOnEmbeddingsScores` | `Scorer`         |
| Gaussian Process regression         | `evedesign.models.supervised.GpytorchModel`                      | `Scorer`         |

### Samplers

| Name          | Class                                   | Interfaces  | 
|---------------|-----------------------------------------|-------------|
| Gibbs sampler | `evedesign.samplers.gibbs.GibbsSampler` | `Generator` |

### Analyzers

| Name                                   | Class                                                         | Interfaces   | 
|----------------------------------------|---------------------------------------------------------------|--------------|
| UMAP sequence space projection         | `evedesign.analyzers.sequence_space.SequenceSpaceUMAP`        | `Analyzer`   |
| MDS sequence space projection          | `evedesign.analyzers.sequence_space.SequenceSpaceMDS`         | `Analyzer`   |
| Landmark MDS sequence space projection | `evedesign.analyzers.sequence_space.SequenceSpaceLandmarkMDS` | `Analyzer`   |
| PCA sequence space projection          | `evedesign.analyzers.sequence_space.SequenceSpacePCA`         | `Analyzer`   |

### Nucleotide sequence generation

| Name                          | Class                                      | Interfaces              | 
|-------------------------------|--------------------------------------------|-------------------------|
| DNA Chisel codon optimization | `evedesign.codons.DNAChiselCodonOptimizer` | `ProteinToDnaOptimizer` |


## Roadmap and contributing

We plan to continuously add more models, restraints, oracles and samplers to the framework, e.g. *de novo* 3D structure generation
with BoltzGen or BindCraft. 

We are actively looking for further contributors to develop our framework jointly with the community. 
If you are interested or feel like an important model is missing from the framework, please get in contact with us!

## License

*evedesign* is released under the MIT license.

## Contact

For general questions or inquiries about *evedesign* please reach out to [hello@evedesign.bio](mailto:hello@evedesign.bio).
