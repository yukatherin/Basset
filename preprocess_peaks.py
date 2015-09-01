#!/usr/bin/env python
from optparse import OptionParser
import gzip
import os
import subprocess

import numpy as np

################################################################################
# preprocess_peaks.py
#
# Preprocess a set of peak BED files for Basset analysis, potentially adding
# them to an existing database of peaks, specified as a BED file with the
# sample accessibilities comma-separated in column 4 and a full accessibility
# table file.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sample_beds_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='db_acc_file', help='Existing database of accessibility scores')
    parser.add_option('-b', dest='db_bed', help='Existing database of BED peaks.')
    parser.add_option('-d', dest='merge_dist', default=0, help='Maximum distance between features allowed for features to be merged. [Default: %default]')
    parser.add_option('-o', dest='out_prefix', default='peaks', help='Output file prefix [Default: %default]')
    parser.add_option('-s', dest='peak_size', default=600, type='int', help='Peak extension size [Default: %default]')
    # parser.add_option('-y', dest='use_y', default=False, action='store_true', help='Use Y chromsosome peaks [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
    	parser.error('Must provide file labeling the samples and providing BED file paths.')
    else:
    	sample_beds_file = args[0]

    # determine whether we'll add to an existing DB
    db_samples = []
    db_add = False
    if (options.db_bed and not options.db_acc_file) or (not options.db_bed and options.db_acc_file):
    	parser.error('Must provide both BED file and accessibility table if \
                                    you want to add to an existing database')
    elif options.db_bed and options.db_acc_file:
    	db_add = True
    	db_acc_in = open(options.db_acc_file)
    	db_samples = db_acc_in.readline().split('\t')[1:]

    # read in samples and assign them indexes into the db
    sample_beds = []
    sample_dbi = []
    for line in open(sample_beds_file):
    	a = line.rstrip().split('\t')
    	sample_dbi.append(len(db_samples))
    	db_samples.append(a[0])
    	sample_beds.append(a[1])


    #################################################################
    # print peaks to chromosome-specific files
    #################################################################
    chrom_files = {}
    chrom_outs = {}

    peak_beds = sample_beds
    if db_add:
        peak_beds.append(options.db_bed)

    for bi in range(len(peak_beds)):
        if peak_beds[bi][-3:] == '.gz':
            peak_bed_in = gzip.open(peak_beds[bi])
        else:
            peak_bed_in = open(peak_beds[bi])

        for line in peak_bed_in:
            a = line.split('\t')
            a[-1] = a[-1].rstrip()

            chrom = a[0]
            strand = '+'
            if len(a) > 5 and a[5] in '+-':
                strand = a[5]
            chrom_key = (chrom,strand)

            # open chromosome file
            if chrom_key not in chrom_outs:
                chrom_files[chrom_key] = '%s_%s_%s.bed' % (options.out_prefix, chrom, strand)
                chrom_outs[chrom_key] = open(chrom_files[chrom_key], 'w')

            # if it's the db bed
            if db_add and bi == len(peak_beds)-1:
                print >> chrom_outs[chrom_key], line,

            # if it's a new bed
            else:
                # specify the sample index
                while len(a) < 7:
                    a.append('')
                a[5] = strand
                a[6] = str(sample_dbi[bi])
                print >> chrom_outs[chrom_key], '\t'.join(a)

        peak_bed_in.close()

    # close chromosome-specific files
    for chrom_key in chrom_outs:
        chrom_outs[chrom_key].close()


    #################################################################
    # sort chromosome-specific files
    #################################################################
    for chrom_key in chrom_files:
        chrom,strand = chrom_key
        chrom_sbed = '%s_%s_%s_sort.bed' % (options.out_prefix,chrom,strand)
        sort_cmd = 'sortBed -i %s > %s' % (chrom_files[chrom_key], chrom_sbed)
        subprocess.call(sort_cmd, shell=True)
        os.remove(chrom_files[chrom_key])
        chrom_files[chrom_key] = chrom_sbed


    #################################################################
    # parse chromosome-specific files
    #################################################################
    final_bed_out = open('%s.bed' % options.out_prefix, 'w')

    for chrom_key in chrom_files:
        chrom, strand = chrom_key

        open_peaks = []
        for line in open(chrom_files[chrom_key]):
            a = line.split('\t')
            a[-1] = a[-1].rstrip()

            # construct Peak
            peak_start = int(a[1])
            peak_end = int(a[2])
            peak_acc = acc_set(a[6])
            peak = Peak(peak_start, peak_end, peak_acc)

            if len(open_peaks) == 0:
                # initialize open peak
                open_end = peak_end
                open_peaks = [peak]

            else:
                # operate on exiting open peak

                # if beyond existing open peak
                if open_end + options.merge_dist < peak_start:
                    # close open peak
                    mpeak = merge_peaks(open_peaks, options.peak_size)

                    # print to file
                    print >> final_bed_out, mpeak.bed_str(chrom, strand)

                    # initialize open peak
                    open_end = peak_end
                    open_peaks = [peak]

                else:
                    # extend open peak
                    open_peaks.append(peak)
                    open_end = max(open_end, peak_end)

        if len(open_peaks) > 0:
            # close open peak
            mpeak = merge_peaks(open_peaks, options.peak_size)

            # print to file
            print >> final_bed_out, mpeak.bed_str(chrom, strand)

    final_bed_out.close()

    # clean
    for chrom_key in chrom_files:
        os.remove(chrom_files[chrom_key])


    #################################################################
    # construct/update accessibility table
    #################################################################
    final_acc_out = open('%s_acc.txt' % options.out_prefix, 'w')

    # print header
    cols = [''] + db_samples
    print >> final_acc_out, '\t'.join(cols)

    # print sequences
    for line in open('%s.bed' % options.out_prefix):
        a = line.split('\t')
        # index peak
        peak_id = '%s:%s-%s:%s' % (a[0], a[1], a[2], a[5])

        # construct full accessibility vector
        peak_acc = [0]*len(db_samples)
        for ai in a[6].split(','):
            peak_acc[int(ai)] = 1

        # print line
        cols = [peak_id] + peak_acc
        print >> final_acc_out, '\t'.join([str(c) for c in cols])

    final_acc_out.close()


def acc_set(acc_cs):
    ''' Return a set of ints from a comma-separated list of int strings.

    Attributes:
        acc_str (str) : comma-separated list of int strings

    Returns:
        set (int) : int's in the original string
    '''
    ai_strs = [ai for ai in acc_cs.split(',')]
    if ai_strs[-1] == '':
        ai_strs = ai_strs[:-1]
    return set([int(ai) for ai in ai_strs])


def merge_peaks(peaks, peak_size):
    ''' Merge the Peaks in the given list.

    Attributes:
        peaks (list[Peak]) : list of Peaks
        peak_size (int) : desired peak extension size

    Returns:
        Peak representing the merger
    '''
    # determine peak midpoints
    peak_mids = []
    for p in peaks:
        mid = (p.start + p.end - 1) / 2.0
        peak_mids.append(mid)

    # take the mean
    merge_mid = int(0.5+np.mean(peak_mids))

    # extend to the full size
    merge_start = merge_mid - peak_size/2
    merge_end = merge_mid + peak_size/2

    # merge accessibilities
    merge_acc = set()
    for p in peaks:
        merge_acc |= p.acc

    return Peak(merge_start, merge_end, merge_acc)


class Peak:
    ''' Peak representation

    Attributes:
        start (int)   : peak start
        end   (int)   : peak end
        acc   (set[int]) : set of sample indexes where this peak is accessible.
    '''
    def __init__(self, start, end, acc):
        self.start = start
        self.end = end
        self.acc = acc

    def bed_str(self, chrom, strand):
        ''' Return a BED-style line '''
        acc_str = ','.join([str(ai) for ai in sorted(list(self.acc))])
        cols = (chrom, str(self.start), str(self.end), '.', '.', strand, acc_str)
        return '\t'.join(cols)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
