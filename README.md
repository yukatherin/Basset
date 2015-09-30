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
    export BASSETDIR=the/dir/where/basset/is/installed

To make the code available for use in any directory, also write
    export PATH=$BASSETDIR/src:$PATH
    export LUAPATH=$BASSETDIR/src:$LUAPATH
    export PYTHONPATH=$BASSETDIR/src:$PYTHONPATH

To download and install additional useful data, like my best pre-trained model and public datasets, run
    ./install_data.py

Finally, if you're downloading this tool,

Full list of requirements:
- [Torch7](http://torch.ch/docs/getting-started.html)
  - nn
  - optim
  - cutorch
  - cunn
  - lfs
  - [hdf5](https://github.com/deepmind/torch-hdf5)
  - [dpnn](https://github.com/nicholas-leonard/dpnn)
  - [inn](https://github.com/szagoruyko/imagine-nn)
- Python
  - [numpy](http://www.numpy.org/)
  - [matplotlib](http://matplotlib.org/)
  - [seaborn](http://stanford.edu/~mwaskom/software/seaborn/index.html)
  - [pandas](http://pandas.pydata.org/)
  - [h5py](http://www.h5py.org/)
  - [sklearn](http://scikit-learn.org/stable/)
  - [pysam](https://code.google.com/p/pysam/)
- Bioinformatics
  - [bedtools](http://bedtools.readthedocs.org/en/latest/)
  - [Samtools](http://www.htslib.org/)
  - [WebLogo](http://weblogo.threeplusone.com/) (optional)
  - [Tomtom, from MEME Suite](http://meme-suite.org/doc/download.html) (optional)

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

Each second order item should be a page.

- Preprocess
  - preprocess_peaks.py
- Train
- Predict
- Visualization