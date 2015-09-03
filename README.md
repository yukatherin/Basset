# Basset
#### Convolutional artificial neural network analysis for DNA sequences.

Basset provides researchers with tools to:

1. Apply convolutional neural networks to learn accurate models for properties of DNA sequences such as accessibility (via DNaseI-seq or ATAC-seq) or binding (via ChIP-Seq).
2. Extract the information learned.

-------------------------------------------------------------------------------------------------------------------
### Installation

Requirements:
- BEDtools
- WebLogo
- Torch7
  - hdf5
  - nn
  - dpnn
  - inn
  - optim
  - cutorch
  - cunn
  - metrics
  - lfs
- Python
  - numpy
  - matplotlib
  - seaborn
  - pandas
  - h5py
  - sklearn (only for preprocessing.scale)
  - pysam (only for fasta)


-------------------------------------------------------------------------------------------------------------------
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

-------------------------------------------------------------------------------------------------------------------
### Tutorials

Each second order item should be a page.

- Preprocess
  - preprocess_peaks.py
- Train
- Predict
- Visualization