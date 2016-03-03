#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import subprocess
import sys

import numpy as np

################################################################################
# preprocess_chromatin.py
#
# Annotate the sites in a BED file with a set of Wig/BigWig tracks.
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sites_bed> <sample_wigs_file> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='bin_size', default=None, type='int', help='Bin size to take the mean track value [Default: %default]')
    parser.add_option('-f', dest='function', default='mean', help='Function to compute in each bin [Default: %default]')
    parser.add_option('-n', dest='normalize', default=False, action='store_true', help='Normalize ')
    parser.add_option('-s', dest='span', default=200, type='int', help='Span of sequence to consider around each site Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide sites BED and file with Wig/BigWig labels and paths')
    else:
        sites_bed_file = args[0]
        sample_wigs_file = args[1]
        out_file = args[2]

    if options.bin_size is None:
        options.bin_size = options.span

    if options.span % options.bin_size != 0:
        parser.error('Bin size must evenly divide span')

    #################################################################
    # setup
    #################################################################
    # count sites
    num_sites = 0
    for line in open(sites_bed_file):
        num_sites += 1

    # count bins
    num_bins = options.span / options.bin_size

    # determine span on each side
    span_left = int(round(options.span/2.0))
    span_right = options.span/2

    # get wig files and labels
    target_wigs = OrderedDict()
    for line in open(sample_wigs_file):
        a = line.split()
        target_wigs[a[0]] = a[1]

    # label features
    feature_labels = []
    for target in target_wigs.keys():
        if num_bins == 1:
            feature_labels.append(target)
        else:
            bin_mid = -span_left + options.bin_size/2
            for bi in range(num_bins):
                feature_labels.append('%s_pos%d' % (target,bin_mid))
                bin_mid += options.bin_size

    # initialize features array
    site_features = np.zeros((num_sites,len(feature_labels)), dtype='float16')

    #################################################################
    # process wigs
    #################################################################
    wi = 0
    for line in open(sample_wigs_file):
        a = line.split()
        sample, wig_file = a[0], a[1]

        cmd = 'bwtool matrix %d:%d %s %s /dev/stdout' % (span_left, span_right, sites_bed_file, wig_file)
        print cmd
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

        si = 0
        for line in p.stdout:
            # fix nan's
            line = line.replace('NA', 'nan')

            # convert to floats
            span_values = np.array([float(v) for v in line.split()])

            # reshape into bins
            bin_values = span_values.reshape((num_bins,-1))

            # compute function in bins
            if options.function == 'max':
                bin_sum = np.nanmax(bin_values, axis=1)
            elif options.function == 'mean':
                bin_sum = np.nanmean(bin_values, axis=1, dtype='float64').astype('float16')
            else:
                print >> sys.stderr, 'Unrecognized function %s' % options.function
                exit()

            # copy into primary data structure
            for bi in range(num_bins):
                site_features[si][wi+bi] = bin_sum[bi]

            si += 1

        p.communicate()

        wi += num_bins

    #################################################################
    # normalize
    #################################################################
    if options.normalize:
        site_features = site_features - site_features.mean(axis=0)
        site_features = site_features / site_features.std(axis=0)

    #################################################################
    # output
    #################################################################
    sv_out = open(out_file, 'w')

    cols = [''] + feature_labels
    print >> sv_out, '\t'.join(cols)

    si = 0
    for line in open(sites_bed_file):
        a = line.split()
        site_id = '%s:%s-%s(%s)' % (a[0], a[1], a[2], a[5])
        cols = [site_id] + [str(v) for v in site_features[si]]
        print >> sv_out, '\t'.join(cols)
        si += 1

    sv_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
