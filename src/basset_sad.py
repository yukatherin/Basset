#!/usr/bin/env python
from optparse import OptionParser
import glob, os, subprocess, time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns

from dna_io import dna_one_hot

################################################################################
# basset_sad.py
#
# Compute SAD scores for SNPs in a VCF file.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_th> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='genome_fasta', default='%s/data/genomes/hg19.fa'%os.environ['BASSETDIR'], help='Genome FASTA from which sequences will be drawn [Default: %default]')
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=600, help='Sequence length provided to the model [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad', help='Output directory for tables and plots [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide Torch model and VCF file')
    else:
        model_th = args[0]
        vcf_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # load SNPs
    #################################################################
    vcf_in = open(vcf_file)

    # read through header
    line = vcf_in.readline()
    while line[0] == '#':
        line = vcf_in.readline()

    # read in SNPs
    snps = []
    for line in vcf_in:
        snps.append(SNP(line))

    #################################################################
    # prep input sequences
    #################################################################
    left_len = options.seq_len/2 - 1
    right_len = options.seq_len/2

    # open genome FASTA
    genome = pysam.Fastafile(options.genome_fasta)

    # initialize one hot coded vector list
    seq_vecs_list = []

    # name sequences
    seq_headers = []

    for snp in snps:
        # specify positions in GFF-style 1-based
        seq_start = snp.pos - left_len
        seq_end = snp.pos + right_len + len(snp.ref_allele) - snp.longest_alt()

        # extract sequence as BED style
        seq = genome.fetch(snp.chrom, seq_start-1, seq_end)

        # verify that ref allele matches ref sequence
        seq_ref = seq[left_len:left_len+len(snp.ref_allele)]
        if seq_ref != snp.ref_allele:
            print >> sys.stderr, 'WARNING: skipping %s because reference allele does not match reference genome: %s vs %s' % (snp.rsid, snp.ref_allele, seq_ref)
            continue

        # one hot code ref allele
        seq_vecs_list.append(dna_one_hot(seq[:options.seq_len], options.seq_len))

        # name ref allele
        seq_headers.append('%s_%s' % (snp.rsid, cap_allele(snp.ref_allele)))

        for alt_al in snp.alt_alleles:
            # remove ref allele and include alt allele
            seq_alt = seq[:left_len] + alt_al + seq[left_len+len(snp.ref_allele):]

            # one hot code
            seq_vecs_list.append(dna_one_hot(seq_alt, options.seq_len))

            # name
            seq_headers.append('%s_%s' % (snp.rsid, cap_allele(alt_al)))

    # stack
    seq_vecs = np.vstack(seq_vecs_list)

    # write to HDF5
    h5f = h5py.File('%s/model_in.h5'%options.out_dir, 'w')
    h5f.create_dataset('test_in', data=seq_vecs)
    h5f.close()


    #################################################################
    # predict in Torch
    #################################################################
    subprocess.call('basset_predict.lua -norm %s %s/model_in.h5 %s/model_out.txt' % (model_th, options.out_dir, options.out_dir), shell=True)

    # read in predictions
    seq_preds = []
    for line in open('%s/model_out.txt' % options.out_dir):
        seq_preds.append(np.array([float(p) for p in line.split()]))
    seq_preds = np.array(seq_preds)


    #################################################################
    # plot raw predictions
    #################################################################
    sns.set(style='white', font_scale=30.0/seq_preds.shape[0])
    plt.figure()
    sns.heatmap(seq_preds, xticklabels=False, yticklabels=seq_headers)
    plt.tight_layout()
    plt.savefig('%s/raw_heat.pdf' % options.out_dir)
    plt.close()

    #################################################################
    # plot SAD
    #################################################################
    sad_matrix = []
    sad_labels = []

    pi = 0
    for snp in snps:
        # get reference prediction
        ref_preds = seq_preds[pi,:]
        pi += 1

        for allele in snp.alt_alleles:
            # get alternate prediction
            alt_preds = seq_preds[pi,:]
            pi += 1

            # normalize by reference
            alt_sad = alt_preds - ref_preds
            sad_matrix.append(alt_sad)

            # label as mutation from reference
            alt_label = '%s_%s>%s' % (snp.rsid, cap_allele(snp.ref_allele), cap_allele(allele))
            sad_labels.append(alt_label)

    # convert fully to numpy array
    sad_matrix = np.array(sad_matrix)

    # plot heatmap
    sns.set(style='white', font_scale=30.0/sad_matrix.shape[0])
    plt.figure()
    sns.heatmap(sad_matrix, xticklabels=False, yticklabels=sad_labels)
    plt.tight_layout()
    plt.savefig('%s/sad_heat.pdf' % options.out_dir)
    plt.close()


def cap_allele(allele, cap=5):
    ''' Cap the length of an allele in the figures '''
    if len(allele) > cap:
        allele = allele[:cap] + '*'
    return allele


class SNP:
    ''' SNP

    Represent SNPs read in from a VCF file

    Attributes:
        vcf_line (str)
    '''
    def __init__(self, vcf_line):
        a = vcf_line.split()
        if a[0].startswith('chr'):
            self.chrom = a[0]
        else:
            self.chrom = 'chr%s' % a[0]
        self.pos = int(a[1])
        self.rsid = a[2]
        self.ref_allele = a[3]
        self.alt_alleles = a[4].split(',')
        self.index_snp = None
        if len(a) >= 6:
            self.index_snp = a[5]

    def get_alleles(self):
        ''' Return a list of all alleles '''
        alleles = [self.ref_allele] + self.alt_alleles
        return alleles

    def longest_alt(self):
        ''' Return the longest alt allele. '''
        return max([len(al) for al in self.alt_alleles])


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()