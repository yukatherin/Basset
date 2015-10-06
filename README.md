# Basset
#### Deep convolutional neural networks for predicting DNA sequence activity.

Basset provides researchers with tools to:

1. Train deep convolutional neural networks to learn highly accurate models of DNA sequence activity such as accessibility (via DNaseI-seq or ATAC-seq), protein binding (via ChIP-seq), and chromatin state.
2. Interpret the principles learned by the model.

---------------------------------------------------------------------------------------------------
### Installation

Basset has a few dependencies because it uses both Torch7 and Python and takes advantage of a variety of packages available for both.

First, I recommend installing Torch7 from [here](http://torch.ch/docs/getting-started.html). If you plan on training models on a GPU, make sure that you have CUDA installed and Torch should find it.

For the Python dependencies, I highly recommend the [Anaconda distribution](https://www.continuum.io/downloads). The only library missing is pysam, which you can install through Anaconda or manually from [here](https://code.google.com/p/pysam/).

To download and install the remaining dependencies, run
```
    ./install_dependencies.py
```

Basset relies on the environmental variable BASSETDIR to orient itself. In your startup script (e.g. .bashrc), write
```
    export BASSETDIR=the/dir/where/basset/is/installed
```

To make the code available for use in any directory, also write
```
    export PATH=$BASSETDIR/src:$PATH
    export LUAPATH=$BASSETDIR/src:$LUAPATH
    export PYTHONPATH=$BASSETDIR/src:$PYTHONPATH
```

To download and install additional useful data, like my best pre-trained model and public datasets, run
```
    ./install_data.py
```

The full requirement list is [here](docs/requirements.md).

---------------------------------------------------------------------------------------------------
### Documentation

Basset is under active development, so don't hesitate to ask for clarifications or additional features, documentation, or tutorials.

Each first order item should be a page.
Then each second order item should be a section on the page.
Ideally, the text here links to that page section.

- [File specifications](docs/file_specs.md)
  - [BED](docs/file_specs.md#bed)
  - [Table](docs/file_specs.md#table)
  - [HDF5](docs/file_specs.md#hdf5)
- [Preprocess](docs/preprocess.md)
  - [preprocess_features.py](docs/preprocess.md#preprocess_features.py)
  - [seq_hdf5.py](docs/preprocess.md#seq_hdf.py)
  - [basset_sample.py](docs/preprocess.md#basset_sample.py)
- Learning
  - [basset_train.lua](docs/learning.md#train)
  - [basset_test.lua](docs/learning.md#test)
  - [basset_predict.lua](docs/learning.md#predict)
- Visualization
  - [basset_motifs.py]
  - [basset_motifs_infl.py]
  - [basset_sat.py]
  - [basset_sat_vcf.py]
  - [basset_sad.py]

---------------------------------------------------------------------------------------------------
### Tutorials

- Preprocess
  - [Prepare the ENCODE and Epigenomics Roadmap compendium from scratch.](tutorials/prepare_compendium.ipynb)
  - [Prepare new dataset(s) by adding to a compendium.](tutorials/new_data_many.ipynb)
  - [Prepare new dataset(s) in isolation.](tutorials/new_data_iso.ipynb)
- Train
  - [Train a model.](tutorials/train.ipynb)
- Test
  - [Test a trained model.](tutorials/test.ipynb)
- Visualization
  - [Study the motifs learned by the model.](tutorials/motifs.ipynb)
  - [Execute an in silico saturated mutagenesis](tutorials/sat_mut.ipynb)
  - [Compute SNP Accessibility Difference profiles.](tutorials/sad.ipynb)