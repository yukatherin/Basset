# Basset
#### Convolutional neural network analysis for predicting DNA sequence activity.

Basset provides researchers with tools to:

1. Apply deep convolutional neural networks to learn accurate models to predict DNA sequence activity such as accessibility (via DNaseI-seq or ATAC-seq) or binding (via ChIP-seq).
2. Extract the information learned.

---------------------------------------------------------------------------------------------------
### Installation

Basset has a few dependencies because it uses both Torch7 and Python and takes advantage of a variety of packages available for both.

First, I recommend installing Torch7 from [here](http://torch.ch/docs/getting-started.html). If you plan on training models on a GPU, make sure that you have CUDA installed and Torch should find it.

For the Python dependencies, I highly recommend the [Anaconda distribution](https://www.continuum.io/downloads). The only library missing is pysam, which you can install through Anaconda or manually from [here](https://code.google.com/p/pysam/).

To download and install the remaining dependencies, run
```
    ./install_dependencies.py
```

Basset relies on the environmental variable BASSETDIR to orient itself. In your startup script (.e.g .bashrc), write
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

Each first order item should be a page.
Then each second order item should be a section on the page.
Ideally, the text here links to that page section.

- File specifications
  - [BED](docs/file_specs.md#bed)
  - Table
- Preprocess
  - preprocess_peaks.py
- Train
- Predict
- Visualization

---------------------------------------------------------------------------------------------------
### Tutorials

- Preprocess
  - [Prepare new dataset(s) by adding to a compendium.](tutorials/new_data_many.ipynb)
  - [Prepare new dataset(s) in isolation.](tutorials/new_data_iso.ipynb)
- Train
- Predict
- Visualization