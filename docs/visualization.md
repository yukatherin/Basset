### Basset
###### Deep convolutional neural networks for DNA sequence analysis.
--------------------------------------------------------------------------------
### Visualization

<a name="motifs"/>
###### basset_motifs.py

Collect statistics and make plots to explore the first convolution layer of the given model using the given sequences.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Saved model to use |
| test_hdf5_file | [HDF5](../docs/file_specs.md#hdf5) | Test data |

| Options | Variable | Description |
| --- | --- | --- |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -o | out_dir | Output directory |
| -m | meme_db | MEME database used to annotate motifs |
| -s | sample | Sample sequences from the test set |
| -t | trim_filters | Trim uninformative positions off the filter ends |

--------------------------------------------------------------------------------
<a name="infl"/>
###### basset_motifs_infl.py

Collect statistics and make plots to explore the first convolution layer of the given model using the given sequences.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Saved model to use |
| test_hdf5_file | [HDF5](../docs/file_specs.md#hdf5) | Test data |

| Options | Variable | Description |
| --- | --- | --- |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -o | out_dir | Output directory |
| -m | meme_db | MEME database used to annotate motifs |
| -s | sample | Sample sequences from the test set |
| -t | trim_filters | Trim uninformative positions off the filter ends |

--------------------------------------------------------------------------------
<a name="sat"/>
###### basset_sat.py

Collect statistics and make plots to explore the first convolution layer of the given model using the given sequences.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Saved model to use |
| test_hdf5_file | [HDF5](../docs/file_specs.md#hdf5) | Test data |

| Options | Variable | Description |
| --- | --- | --- |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -o | out_dir | Output directory |
| -m | meme_db | MEME database used to annotate motifs |
| -s | sample | Sample sequences from the test set |
| -t | trim_filters | Trim uninformative positions off the filter ends |

--------------------------------------------------------------------------------
<a name="sat_vcf"/>
###### basset_sat_vcf.py

Collect statistics and make plots to explore the first convolution layer of the given model using the given sequences.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Saved model to use |
| test_hdf5_file | [HDF5](../docs/file_specs.md#hdf5) | Test data |

| Options | Variable | Description |
| --- | --- | --- |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -o | out_dir | Output directory |
| -m | meme_db | MEME database used to annotate motifs |
| -s | sample | Sample sequences from the test set |
| -t | trim_filters | Trim uninformative positions off the filter ends |

--------------------------------------------------------------------------------
<a name="sad"/>
###### basset_sad.py

Collect statistics and make plots to explore the first convolution layer of the given model using the given sequences.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Saved model to use |
| test_hdf5_file | [HDF5](../docs/file_specs.md#hdf5) | Test data |

| Options | Variable | Description |
| --- | --- | --- |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -o | out_dir | Output directory |
| -m | meme_db | MEME database used to annotate motifs |
| -s | sample | Sample sequences from the test set |
| -t | trim_filters | Trim uninformative positions off the filter ends |
