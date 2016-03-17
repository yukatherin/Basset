#!/usr/bin/env python
from optparse import OptionParser
import copy
import math
import os
import random
import string
import subprocess
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

sns_colors = sns.color_palette('deep')

from dna_io import one_hot_set, vecs2dna

################################################################################
# basset_kmers.py
#
# Generate random sequences and study scores by k-mers.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='targets_file', default=None, help='File labelings targets in the second column [Default: %default]')
    parser.add_option('-c', dest='center_nt', default=50, help='Center nt to consider kmers from [Default: %default]')
    parser.add_option('-d', dest='model_out_file', default=None, help='Pre-computed model output table.')
    parser.add_option('-k', dest='kmer', default=8, type='int', help='K-mer length [Default: %default]')
    parser.add_option('-l', dest='seq_len', default=1000, type='int', help='Input sequence length [Default: %default]')
    parser.add_option('-n', dest='num_seqs', default=100000, type='int', help='Number of sequences to predict [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='.')
    parser.add_option('-r', dest='rc', default=False, action='store_true', help='Consider k-mers w/ their reverse complements [Default: %default]')
    parser.add_option('-s', dest='sample', default=None, type='int', help='Sequences to sample [Default: %default]')
    parser.add_option('-t', dest='targets', default=None, help='Comma-separated list of targets to analyze in more depth [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide Basset model file.')
    else:
        model_file = args[0]

    random.seed(2)

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # generate random sequences
    #################################################################
    # random sequences
    seq_vecs = np.zeros((options.num_seqs,4,1,options.seq_len), dtype='float16')
    for si in range(options.num_seqs):
        for li in range(options.seq_len):
            ni = random.randint(0,3)
            seq_vecs[si,ni,0,li] = 1

    # create a new HDF5 file
    seq_hdf5_file = '%s/seqs.h5' % options.out_dir
    seq_hdf5_out = h5py.File(seq_hdf5_file, 'w')
    seq_hdf5_out.create_dataset('test_in', data=seq_vecs)
    seq_hdf5_out.close()

    # get fasta
    seq_dna = vecs2dna(seq_vecs)


    #################################################################
    # Torch predict
    #################################################################
    if options.model_out_file is None:
        options.model_out_file = '%s/model_out.txt' % options.out_dir

        torch_cmd = 'basset_predict.lua -scores %s %s %s' % (model_file, seq_hdf5_file, options.model_out_file)
        print torch_cmd
        subprocess.call(torch_cmd, shell=True)

    # load scores
    seq_scores = np.loadtxt(options.model_out_file, dtype='float32')

    # read target labels
    if options.targets_file:
        target_labels = [line.split()[1] for line in open(options.targets_file)]
    else:
        target_labels = ['t%d'%(ti+1) for ti in range(seq_scores.shape[1])]

    if options.targets == None:
        options.targets = range(seq_scores.shape[1])


    #################################################################
    # process and output
    #################################################################
    kmers_start = (options.seq_len - options.center_nt) / 2

    for ti in options.targets:
        ##############################################
        # hash scores by k-mer
        ##############################################
        kmer_scores = {}

        for si in range(len(seq_dna)):
            # get score
            sscore = seq_scores[si,ti]

            # hash to each center kmer
            for ki in range(kmers_start, kmers_start + options.center_nt):
                kmer = seq_dna[si][ki:ki+options.kmer]
                if options.rc:
                    kmer = consider_rc(kmer)

                kmer_scores.setdefault(kmer,[]).append(sscore)


        ##############################################
        # print table
        ##############################################
        table_out = open('%s/table.txt' % options.out_dir, 'w')

        for kmer in kmer_scores:
            cols = (kmer, len(kmer_scores[kmer]), np.mean(kmer_scores[kmer]), np.std(kmer_scores[kmer])/math.sqrt(len(kmer_scores[kmer])))
            print >> table_out, '%s  %4d  %6.3f  %6.3f' % cols

        table_out.close()



def consider_rc(kmer):
    rc_kmer = rc(kmer)
    if rc_kmer < kmer:
        return rc_kmer
    else:
        return kmer


def rc(seq):
    ''' Reverse complement sequence'''
    return seq.translate(string.maketrans("ATCGatcg","TAGCtagc"))[::-1]


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
