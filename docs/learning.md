### Basset
###### Convolutional artificial neural network analysis for DNA sequences.
--------------------------------------------------------------------------------
### Learning

<a name="basset_train"/>
###### basset_train.lua

Train a convolutional neural network on the given data.

- Input
  - [HDF5](../docs/file_specs.md#hdf5) with "train_in", "train_out", "valid_in", "valid_out" specified.
- Output
  - Convnet file
- Options

| Option | Help |
| --- | --- |
| -cuda | Run on GPU [Default: False] |
| -restart | Start training with the given saved model |
| -save | Prefix for saved models [Default: bassett] |
| -seed | Random number seed [Default: 1] |
| -spearmint | Spearmint job id |
| -stagnant_t | Allowed epochs with stagnant validation

--------------------------------------------------------------------------------
<a name="basset_test"/>
###### basset_test.lua

Report model performance on the given test data, producing files with AUC and points along the ROC curves for each sample.

- Input
  - Convnet file
  - [HDF5](../docs/file_specs.md#hdf5) with "test_in" specified.
  - Output directory
- Output
  - aucs.txt : AUCS for each sample.
  - rocX.txt : ROC points for all samples X.
- Options

| Option | Help |
| --- | --- |
| -cuda | Run on GPU [Default: False] |

--------------------------------------------------------------------------------
<a name="basset_predict"/>
###### basset_predict.lua

Predict activity for a new set of sequences.

- Input
  - Convnet file
  - [HDF5](../docs/file_specs.md#hdf5) with "test_in" specified.
  - Output file
- Output
  - HDF5?
- Options

| Option | Help |
| --- | --- |
| -cuda | Run on GPU [Default: False] |
