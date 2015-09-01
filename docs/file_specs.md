# Basset
#### Convolutional artificial neural network analysis for DNA sequences.
-------------------------------------------------------------------------------------------------------------------
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

-------------------------------------------------------------------------------------------------------------------
<a name="table"/>
###### Table

- tab-separated
- row indexes are chrom:start:end:strand corresponding to [BED](#bed).
- columns are sample names.
- entries are 0/1 accessibility/binding/activity.