#!/usr/bin/env python
from optparse import OptionParser
import h5py

import numpy as np

import dna_io

################################################################################
# hdf5_test_bed.py
#
# Extract a BED of test sequences from an HDF5 file with test_headers that
# specify the region.
################################################################################

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <hdf5_file>'
    parser = OptionParser(usage)
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide HDF file')
    else:
        hdf5_file = args[0]

    # get headers
    hdf5_in = h5py.File(hdf5_file, 'r')
    seq_headers = np.array(hdf5_in['test_headers'])
    hdf5_in.close()

    for si, header in enumerate(seq_headers):
        chrom = header[:header.find(':')]
        start = int(header[header.find(':')+1:header.find('-')])
        end = int(header[header.find('-')+1:])
        print '%s\t%d\%d' % (chrom,start,end)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
