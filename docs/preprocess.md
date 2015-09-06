### Basset
###### Convolutional artificial neural network analysis for DNA sequences.
--------------------------------------------------------------------------------
## Preprocess

<a name="preprocess_peaks.py"/>
#### preprocess_peaks.py

Merge a set of feature BED files for training.

- Input
  - [target_beds_file](../docs/file_specs.md#bed)
- Output
  - [BED](../docs/file_specs.md#bed)
  - [Table](../docs/file_specs.md#table)
- Options

| Option | Variable | Help |
| --- | --- | --- |
| -a | db_acc_file | Existing database of accessibility scores |
| -b | db_bed | Existing database of BED peaks. |
| -c |chrom_lengths_file | Table of chromosome lengths |
| -m | merge_overlap | Overlap required (after extension to peak_size) to merge features. Can be negative. [Default: 0] |
| -o | out_prefix | Output file prefix [Default: peak_size] |
| -s | peak_size | Peak extension size [Default: 600] |
| -y | ignore_y | Ignore Y chromsosome peaks [Default: False] |


--------------------------------------------------------------------------------
<a name="seq_hdf5.py"/>
#### seq_hdf5.py

Construct an HDF5 file, dividng the data into training, validation, and test subsets.

- Input
  - FASTA
  - [Table](../docs/file_specs.md#table)
  - Output HDF5
- Output
  - [HDF5](../docs/file_specs.md#hdf5)
- Options

| Option | Variable | Help |
| --- | --- | --- |
| -b | batch_size | Align sizes with batch size |
| -c | counts | Validation and training percentages are given as raw counts [Default: False] |
| -r | permute | Permute sequences [Default: False] |
| -s | random_seed | numpy.random seed [Default: 1] |
| -t | test_pct | Test % [Default: 0] |
| -v | valid_pct | Validation % [Default: 0] |


--------------------------------------------------------------------------------
<a name="sample_db.py"/>
#### sample_db.py

Sample sequences from an existing database.

- Input
  - sample_num[int]
  - [BED](../docs/file_specs.md#bed)
  - [Table](../docs/file_specs.md#table)
- Output
  - [BED](../docs/file_specs.md#bed)
  - [Table](../docs/file_specs.md#table)
- Options

| Option | Variable | Help |
| --- | --- | --- |
| -o | out_prefix | Output file prefix [Default: peak_size] |
