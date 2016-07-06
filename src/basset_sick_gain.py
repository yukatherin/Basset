#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import os
import subprocess

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import stats

################################################################################
# basset_sick_gain.py
#
# Shuffle SNPs outside of DNase sites and compare the SAD distributions.
#
# Todo:
#  -Control for GC% changes introduced by mutation shuffles.
#  -Control for positional changes within the DHS regions.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <vcf_file> <excl_bed_file> <model_file>'
    parser = OptionParser(usage)
    parser.add_option('-e', dest='add_excl_bed', default='%s/assembly/hg19_gaps.bed', help='Additional genomic regions to exclude from the shuffle [Default: %default]')
    parser.add_option('-g', dest='gpu', default=False, action='store_true', help='Run on GPU [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=600, help='Sequence length provided to the model [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad_shuffle', help='Output directory')
    parser.add_option('-r', dest='replot', default=False, action='store_true', help='Re-plot only, without re-computing [Default: %default]')
    parser.add_option('-s', dest='num_shuffles', default=1, type='int', help='Number of SNP shuffles [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide VCF file, sample BEDs file, and model file')
    else:
        vcf_file = args[0]
        excl_bed_file = args[1]
        model_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #########################################
    # supplement the excluded sites
    #########################################
    if options.add_excl_bed is not None:
        # copy exclusion BED file
        supp_excl_bed_file = '%s/excl.bed' % options.out_dir
        shutil.copy(excl_bed_file, supp_excl_bed_file)

        # add on additional sites
        supp_excl_bed_out = open(supp_excl_bed_out, 'a')
        for line in open(options.add_excl_bed):
            print(line, file=supp_excl_bed_out, end='')
        supp_excl_bed_out.close()

        excl_bed_file = supp_excl_bed_file

    #########################################
    # compute SAD
    #########################################
    # filter VCF to excluded SNPs
    excl_vcf_file = '%s/excl.vcf' % options.out_dir
    cmd = 'bedtools intersect -v -a %s -b %s > %s' % (vcf_file, excl_bed_file, excl_vcf_file)
    if not options.replot:
        subprocess.call(cmd, shell=True)

    # compute SADs
    true_sad = compute_sad(excl_vcf_file, model_file, '%s/excl_sad'%options.out_dir, options.seq_len, options.gpu, options.replot)

    #########################################
    # compute shuffled SAD
    #########################################
    shuffle_sad = np.zeros((true_sad.shape[0],true_sad.shape[1],options.num_shuffles))
    for ni in range(options.num_shuffles):
        # shuffle the SNPs
        shuf_vcf_file = '%s/shuf%d.vcf' % (options.out_dir, ni)
        cmd = 'bedtools shuffle -excl %s -i %s' % (excl_bed_file, excl_vcf_file, shuf_vcf_file)
        if not options.replot:
            subprocess.call(cmd, shell=True)

        # compute SAD scores for shuffled SNPs
        shuffle_sad[:,:,ni] = compute_sad(shuf_vcf_file, model_file, '%s/shuf%d_sad'%(options.out_dir,ni), options.seq_len, options.gpu, options.replot)

    #########################################
    # stats and plots
    #########################################
    mw_out = open('%s/mannwhitney.txt' % options.out_dir, 'w')

    for ti in range(true_sad.shape[1]):
        # plot CDFs
        sns_colors = sns.color_palette('deep')
        plt.figure()
        plt.hist(true_sad[:,ti], 1000, normed=1, histtype='step', cumulative=True, color=sns_colors[0], linewidth=1, label='SNPs')
        plt.hist(shuffle_sad[:,ti,:].flatten(), 1000, normed=1, histtype='step', cumulative=True, color=sns_colors[2], linewidth=1, label='Shuffle')
        ax = plt.gca()
        ax.grid(True, linestyle=':')
        ax.set_xlim(-.2, .2)
        plt.legend()
        plt.savefig('%s/t%d_cdf.pdf' % (options.out_dir,ti))
        plt.close()

        # compute Mann-Whitney
        mw_z, mw_p = stats.mannwhitneyu(true_sad[:,ti], shuffle_sad[:,ti,:].flatten())
        cols = (sample, true_sad.shape[0], true_sad[:,ti].mean(), shuffle_sad[:,ti,:].mean(), mw_z, mw_p)
        print('%-20s  %5d  %6.3f  %6.3f  %6.2f  %6.1e' % cols, file=mw_out)

    mw_out.close()


def compute_sad(vcf_file, model_file, out_dir, seq_len, gpu, replot):
    ''' Run basset_sad.py to compute scores. '''

    cuda_str = ''
    if gpu:
        cuda_str = '--cudnn'

    cmd = 'basset_sad.py %s -l %d -o %s %s %s' % (cuda_str, seq_len, out_dir, model_file, vcf_file)
    if not replot:
        subprocess.call(cmd, shell=True)

    num_targets = sad_targets('%s/sad_table.txt'%out_dir)

    sad_table = []
    sad_table_in = open('%s/sad_table.txt' % out_dir)
    sad_table_in.readline()
    last_snpid = None
    for line in sad_table_in:
        a = line.split()
        snpid = a[0]
        sad = float(a[-1])

        if last_snpid == snpid:
            sad_table[-1].append(sad)
        else:
            sad_table.append([sad])

        last_snpid = snpid

    return np.array(sad)


def sad_targets(sad_table_file):
    ''' Determine how many targets there are in a SAD table.'''
    sad_in = open(sad_table_file)
    sad_in.readline()

    line = sad_in.readline()
    snp_id = line.split()[0]
    targets = 1

    line = sad_in.readline()
    while snp_id == line.split()[0]:
        targets += 1
        line = sad_in.readline()

    return targets


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
