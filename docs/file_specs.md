### Basset
###### Convolutional artificial neural network analysis for DNA sequences.
--------------------------------------------------------------------------------
### File specifications

<a name="bed"/>
###### BED

- chrom
- start
- end
- name (unused)
- score (unused)
- strand
- accessibilities : comma-separated list of integer indexes

--------------------------------------------------------------------------------
<a name="table"/>
###### Table

- tab-separated
- row indexes are chrom:start:end:strand corresponding to [BED](#bed).
- columns are sample names.
- entries are 0/1 accessibility/binding/activity.

--------------------------------------------------------------------------------
<a name="hdf5"/>
###### HDF5

- https://www.hdfgroup.org/HDF5/
- Basset will look for datasets named:
  - train_in
  - train_out
  - valid_in
  - valid_out
  - test_in
  - test_out
  - test_headers
